# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


"""Utilities for extracting pandapower VA diff simulation results per contingency."""

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from beartype.typing import Optional
from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.result_constants import ResultConstants
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
)
from toop_engine_grid_helpers.pandapower.outage_group import (
    OUTAGE_GROUP_SEPARATOR,
    ConnectivityGraphCache,
    elem_node_id,
    get_node_table_id,
    is_node_of_element_type,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    SEPARATOR,
)

# Flat polars schema mirroring VADiffResultSchema (index levels as ordinary columns).
_VA_DIFF_RESULT_SCHEMA = {
    "timestep": pl.Int64,
    "contingency": pl.Utf8,
    "element": pl.Utf8,
    "va_diff": pl.Float64,
    "element_name": pl.Utf8,
    "contingency_name": pl.Utf8,
}


def _empty_va_diff_polars() -> pl.DataFrame:
    """Empty va-diff result frame with the flat polars schema."""
    return pl.DataFrame(schema=_VA_DIFF_RESULT_SCHEMA)


def _get_bus_va_series(net: pandapowerNet) -> pd.Series:
    """
    Return a Series of bus voltage angles indexed by bus id.

    Extracts the ``va_degree`` column from ``net.res_bus`` and drops buses
    without a solved voltage angle (NaN).

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network with loadflow results available in ``net.res_bus``.

    Returns
    -------
    pd.Series
        Series of voltage angles in degrees, indexed by bus id, containing
        only entries with non-null ``va_degree`` values.
    """
    res_bus = net.res_bus.dropna(subset=["va_degree"])
    return res_bus["va_degree"]


def _compute_va_diff_both_ends(
    open_cb: pl.DataFrame,
    va_frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute voltage angle differences for open circuit breakers where

    voltage angle results are available on both sides.

    Parameters
    ----------
    open_cb : pl.DataFrame
        Open circuit breakers, columns ``switch_id`` / ``bus`` / ``element``.

    va_frame : pl.DataFrame
        Bus voltage angles, columns ``bus`` (id) / ``va`` (degrees), containing only
        buses with a solved angle.

    Returns
    -------
    va_diff_both : pl.DataFrame
        ``switch`` / ``va_diff`` = ``|va(bus) - va(element)|`` for open switches with a
        solved angle on **both** connected buses.

    open_cb_rest : pl.DataFrame
        Subset of ``open_cb`` (same columns) whose switches lack an angle on both sides.
        These are handled by one-sided or outage-group inference logic.
    """
    va_bus = va_frame.rename({"va": "va_bus"})
    va_element = va_frame.rename({"bus": "element", "va": "va_element"})
    joined = open_cb.join(va_bus, on="bus", how="left").join(va_element, on="element", how="left")

    both_mask = pl.col("va_bus").is_not_null() & pl.col("va_element").is_not_null()
    va_diff_both = joined.filter(both_mask).select(
        pl.col("switch_id").alias("switch"),
        (pl.col("va_bus") - pl.col("va_element")).abs().alias("va_diff"),
    )
    open_cb_rest = joined.filter(~both_mask).select(open_cb.columns)
    return va_diff_both, open_cb_rest


def _build_side(
    df: pl.DataFrame,
    deenergized_sw_side: str,
    energized_sw_side: str,
    node_to_component: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build a normalized (component, switch, bus, deenergized_sw_side) table for one side of a bus-bus switch.

    This helper maps the de-energized side of a bus-bus connection to the connected
    component it belongs to. The energized side is the bus where voltage angle results
    (`va_res`) are available. These values are later used to compute voltage angle
    differences (`va_diff`) to buses in that component.

    It is used to construct per-switch connectivity rows in a vectorized manner.

    Parameters
    ----------
    df : pl.DataFrame
        Open bus-bus switches with columns ``switch_id`` and the two referenced by
        `deenergized_sw_side` and `energized_sw_side`.

    deenergized_sw_side : str
        Name of the column identifying the bus on the de-energized side of the switch.
        Values are converted into component lookup keys of the form `"b<sep><bus_id>"` and
        joined against `node_to_component`.

    energized_sw_side : str
        Name of the column identifying the bus on the energized side of the switch.
        Values populate the output `bus` column (the side with available `va_res`).

    node_to_component : pl.DataFrame
        Lookup frame with columns ``comp_key`` (e.g. ``"b<sep>12"``) and ``comp``.
        Rows whose lookup key is not present are dropped.

    Returns
    -------
    pl.DataFrame
        Columns:

        - comp: connected component identifier of the de-energized-side bus
        - switch: switch identifier
        - bus: energized-side bus index (where `va_res` is available)
        - deenergized_sw_side: de-energized-side bus index (opposite side)

        Rows without a matching component are excluded.
    """
    return (
        df.with_columns(
            (pl.lit("b" + OUTAGE_GROUP_SEPARATOR) + pl.col(deenergized_sw_side).cast(pl.Utf8)).alias("comp_key"),
        )
        .join(node_to_component, on="comp_key", how="inner")
        .select(
            pl.col("comp"),
            pl.col("switch_id").alias("switch"),
            pl.col(energized_sw_side).cast(pl.Int64).alias("bus"),
            pl.col(deenergized_sw_side).cast(pl.Int64).alias("deenergized_sw_side"),
        )
    )


def _build_node_to_component(connected_components: list) -> dict:
    """Map every node to the index of the connected component it belongs to."""
    return {node: cmp_idx for cmp_idx, component in enumerate(connected_components) for node in component}


def get_outage_groups_with_pst(net: pandapowerNet, connected_components: list) -> tuple[list, list[int]]:
    """
    Find connected component groups that contain at least one element with a PST.

    Parameters
    ----------
    net : pandapowerNet
        Network used to detect which elements are associated with PSTs.
    connected_components : list
        Collection of outage groups (e.g., lists/sets of element IDs).

    Returns
    -------
    tuple[list, list]
        - List of connected components that contain at least one PST-related element.
        - List of their corresponding indices in `connected_components`.
    """
    all_els = set(get_elements_with_pst(net))  # O(1) membership checks
    out_gr_with_pst = []
    out_gr_with_pst_ids = []

    for ind, cc in enumerate(connected_components):
        if any(e in all_els for e in cc):
            out_gr_with_pst.append(cc)
            out_gr_with_pst_ids.append(ind)

    return out_gr_with_pst, out_gr_with_pst_ids


def _make_one_sided_rows(
    net: pandapowerNet,
    open_cb_rest: pl.DataFrame,
    va_frame: pl.DataFrame,
    connected_components: list,
    graph_cache: Optional[ConnectivityGraphCache] = None,
) -> pl.DataFrame:
    """
    Build the intermediate row table for "one-sided" VA-diff inference.

    A one-sided case is a switch where only one end (bus or element side) has a solved
    voltage angle available in ``va_deg``. For each such switch, this function produces
    rows with:
      - ``switch``: switch id
      - ``bus``: the energized/known-VA bus id (the side present in ``va_deg``)
      - ``comp``: connected-component index of the opposite (de-energized) side

    Additionally, the output includes a boolean ``pst`` flag indicating whether the
    de-energized component contains at least one PST element (as determined by
    ``get_outage_groups_with_pst``).

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network used to detect PST elements within connected components.
    open_cb_rest : pl.DataFrame
        Open circuit breakers to consider, columns ``switch_id`` / ``bus`` / ``element``.
    va_frame : pl.DataFrame
        Bus voltage angles, columns ``bus`` (id) / ``va`` (degrees). Presence of a bus id
        determines which side is considered energized/known.
    connected_components : list
        List of connected components, where each component is an iterable of node ids.
        The position in the list is used as the component index (``comp``).
    graph_cache : ConnectivityGraphCache, optional
        Cache used to share the node-to-component map and the PST component ids across
        contingencies. ``None`` recomputes both on every call.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['switch', 'bus', 'comp', 'pst']`` suitable for subsequent
        VA-diff computation. ``pst`` is True if the opposite-side component contains a PST.
    """
    # Both of these derive purely from the connected components (PST membership is a
    # structural property of the trafo tables, not of the current tap position), so they
    # are shared across contingencies whenever the graph itself is.
    if graph_cache is None:
        node_to_component = _build_node_to_component(connected_components)
        out_gr_with_pst_ids = get_outage_groups_with_pst(net, connected_components)[1]
    else:
        node_to_component = graph_cache.derive(
            net, "node_to_component", lambda: _build_node_to_component(connected_components)
        )
        out_gr_with_pst_ids = graph_cache.derive(
            net, "pst_component_ids", lambda: get_outage_groups_with_pst(net, connected_components)[1]
        )

    # node -> component index as a polars lookup frame (built once, joined for both sides).
    node_to_component_pl = pl.DataFrame(
        {"comp_key": list(node_to_component.keys()), "comp": list(node_to_component.values())},
        schema={"comp_key": pl.Utf8, "comp": pl.Int64},
    )

    va_bus_ids = va_frame.get_column("bus")
    cb_bus_side = open_cb_rest.filter(pl.col("bus").is_in(va_bus_ids))  # bus side has va
    cb_element_side = open_cb_rest.filter(pl.col("element").is_in(va_bus_ids))  # element side has va

    # element-side: known va bus is r.element; component from OTHER end r.bus
    cb_element_side = _build_side(
        cb_element_side,
        deenergized_sw_side="bus",
        energized_sw_side="element",
        node_to_component=node_to_component_pl,
    )

    # bus-side: known va bus is r.bus; component from OTHER end r.element
    cb_bus_side = _build_side(
        cb_bus_side,
        deenergized_sw_side="element",
        energized_sw_side="bus",
        node_to_component=node_to_component_pl,
    )
    result = pl.concat([cb_bus_side, cb_element_side], how="vertical")
    return result.with_columns(pl.col("comp").is_in(out_gr_with_pst_ids).alias("pst"))


def _filter_components_with_enough_va(df_rows: pl.DataFrame) -> pl.DataFrame:
    """
    Filter rows to components that have at least two distinct buses with voltage-angle data.

    Ensures that voltage-angle differences can be computed within each component by
    keeping only those connected components (``comp``) that contain two or more unique
    bus ids.

    Parameters
    ----------
    df_rows : pl.DataFrame
        Frame containing at least the columns:
        - ``comp``: connected-component identifier
        - ``bus``: bus id associated with each row

    Returns
    -------
    pl.DataFrame
        Filtered frame containing only rows belonging to components with
        at least two distinct buses. Returns the input unchanged if empty.
    """
    if df_rows.is_empty():
        return df_rows
    valid_comps = (
        df_rows.group_by("comp").agg(pl.col("bus").n_unique().alias("n_bus")).filter(pl.col("n_bus") >= 2).select("comp")
    )
    return df_rows.join(valid_comps, on="comp", how="inner")


def _max_abs_diff_per_bus_within_component(
    buses: np.ndarray,
    va_by_bus: dict[int, float],
) -> dict[int, float]:
    """
    Compute the maximum absolute voltage-angle difference per bus within a component.

    For a given set of buses, calculates for each bus the maximum absolute difference
    in voltage angle (degrees) relative to all other buses in the same set:

        max_{other bus in set} |va(bus) - va(other)|

    Parameters
    ----------
    buses : np.ndarray
        Array of bus ids belonging to the same connected component.
    va_by_bus : dict[int, float]
        Bus voltage angles (in degrees), keyed by bus id.

    Returns
    -------
    dict[int, float]
        Mapping from bus id to the maximum absolute voltage-angle difference (degrees)
        with any other bus in the provided set.
    """
    va = np.array([va_by_bus[int(b)] for b in buses], dtype=float)  # (m,)
    diff = np.abs(va[:, None] - va[None, :])  # (m, m)
    np.fill_diagonal(diff, -np.inf)
    max_per_bus = diff.max(axis=1)
    return dict(zip(map(int, buses), map(float, max_per_bus), strict=True))


def _compute_va_diff_one_side(
    df_rows: pl.DataFrame,
    va_by_bus: dict[int, float],
) -> pl.DataFrame:
    """
    Compute inferred voltage-angle differences for one-sided switch cases.

    Filters component rows to those with sufficient voltage-angle (va) availability, then for each
    connected component computes a per-bus va-difference metric (via
    ``_max_abs_diff_per_bus_within_component``) and maps it back to switches.

    Parameters
    ----------
    df_rows : pl.DataFrame
        Rows describing one-sided switch candidates. Must contain at least the columns:
        - ``comp``: connected-component identifier
        - ``bus``: bus id associated with the row
        - ``switch``: switch id
    va_by_bus : dict[int, float]
        Bus voltage angles (degrees), keyed by bus id.

    Returns
    -------
    pl.DataFrame
        ``switch`` / ``va_diff`` with the inferred va-diff values. Empty if no valid rows.
    """
    empty = pl.DataFrame(schema={"switch": pl.Int64, "va_diff": pl.Float64})
    if df_rows.is_empty():
        return empty

    df_rows = _filter_components_with_enough_va(df_rows)
    if df_rows.is_empty():
        return empty

    bus_va_res: dict[int, float] = {}
    for _, g in df_rows.group_by("comp", maintain_order=True):
        buses = g.get_column("bus").unique().to_numpy()
        bus_va_res.update(_max_abs_diff_per_bus_within_component(buses, va_by_bus))

    va_by_bus_frame = pl.DataFrame(
        {"bus": list(bus_va_res.keys()), "va_diff": list(bus_va_res.values())},
        schema={"bus": pl.Int64, "va_diff": pl.Float64},
    )
    return df_rows.join(va_by_bus_frame, on="bus", how="left").select("switch", "va_diff")


def _apply_tabular_angle(net: pandapowerNet, trafo_df: pd.DataFrame) -> None:
    """
    Apply tabular tap-changer phase shifts to ``trafo_df['angle_deg']`` in-place.

    For transformers with ``tap_changer_type == 'Tabular'``, this function looks up the
    tap position in ``net.trafo_characteristic_table`` (matching
    ``id_characteristic_table`` ↔ ``id_characteristic`` and ``tap_pos`` ↔ ``step``)
    and writes the corresponding ``angle_deg`` into the transformer table.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing ``net.trafo_characteristic_table`` with columns
        ``['id_characteristic', 'step', 'angle_deg']``.
    trafo_df : pd.DataFrame
        Transformer table to update (e.g. ``net.trafo`` or ``net.trafo3w``). Must contain
        ``tap_changer_type``, ``id_characteristic_table`` and ``tap_pos`` columns, and an
        ``angle_deg`` column (will be assigned).

    Returns
    -------
    None
        Modifies ``trafo_df`` in-place.
    """
    mask = trafo_df["tap_changer_type"].eq("Tabular")
    if not mask.any():
        return

    tab = trafo_df.loc[mask].copy()
    tab["trafo_id"] = tab.index
    tab = tab[["trafo_id", "id_characteristic_table", "tap_pos"]].merge(
        net.trafo_characteristic_table[["id_characteristic", "step", "angle_deg"]],
        how="left",
        left_on=["id_characteristic_table", "tap_pos"],
        right_on=["id_characteristic", "step"],
    )

    trafo_df.loc[tab.trafo_id, "angle_deg"] = tab["angle_deg"].to_numpy()


def _apply_ideal_angle(trafo_df: pd.DataFrame) -> None:
    """
    Apply ideal tap-changer phase shifts to ``trafo_df['angle_deg']`` in-place.

    For transformers with ``tap_changer_type == 'Ideal'``, the phase shift is computed as::

        angle_deg = (tap_pos - tap_neutral) * tap_step_degree

    and written to the corresponding rows in ``trafo_df``.

    Parameters
    ----------
    trafo_df : pd.DataFrame
        Transformer table to update (e.g. ``net.trafo`` or ``net.trafo3w``). Must contain
        the columns ``tap_changer_type``, ``tap_pos``, ``tap_neutral``, and
        ``tap_step_degree``, as well as an ``angle_deg`` column to be updated.

    Returns
    -------
    None
        Modifies ``trafo_df`` in-place.
    """
    mask = trafo_df["tap_changer_type"].eq("Ideal")
    if not mask.any():
        return

    ideal = trafo_df.loc[mask].copy()
    ideal["tap_diff"] = ideal["tap_pos"] - ideal["tap_neutral"]
    ideal["angle_deg"] = ideal["tap_diff"] * ideal["tap_step_degree"]

    trafo_df.loc[ideal.index, "angle_deg"] = ideal["angle_deg"].to_numpy()


def _select_vn_kv(df: pd.DataFrame) -> np.ndarray:
    """
    Select the nominal voltage (kV) of the tapped winding side.

    Returns an array containing the nominal voltage corresponding to the
    transformer tap side specified in ``tap_side`` ("hv", "mv", or "lv").
    Supports both 2-winding transformers (no ``vn_mv_kv`` column) and
    3-winding transformers (with ``vn_mv_kv``).

    Parameters
    ----------
    df : pd.DataFrame
        Transformer table containing ``tap_side`` and nominal voltage columns
        (``vn_hv_kv``, ``vn_lv_kv``, and optionally ``vn_mv_kv``).

    Returns
    -------
    np.ndarray
        Array of nominal voltages (in kV) corresponding to the tapped side for
        each row; NaN where the tap side is missing or not applicable.
    """
    has_mv = "vn_mv_kv" in df.columns
    mv = df["vn_mv_kv"] if has_mv else np.nan

    return np.select(
        [df["tap_side"].eq("hv"), df["tap_side"].eq("mv"), df["tap_side"].eq("lv")],
        [df["vn_hv_kv"], mv, df["vn_lv_kv"]],
        default=np.nan,
    )


def _apply_ratio_angle(trafo_df: pd.DataFrame) -> None:
    """
    Apply ratio tap-changer phase shifts to ``trafo_df['angle_deg']`` in-place.

    For transformers with ``tap_changer_type == 'Ratio'``, computes the phase shift
    introduced by the tap position based on the nominal voltage of the tapped side,
    tap position offset, tap step percent, and tap step angle. The calculation uses:

        du = vn_kv * (tap_pos - tap_neutral) * (tap_step_percent / 100)

        angle_deg = atan2(
            du * sin(tap_step_degree),
            vn_kv + du * cos(tap_step_degree)
        )  [converted to degrees]

    where ``vn_kv`` is selected from the tapped winding side (hv/mv/lv).

    Parameters
    ----------
    trafo_df : pd.DataFrame
        Transformer table to update (e.g. ``net.trafo`` or ``net.trafo3w``). Must contain:
        - ``tap_changer_type``, ``tap_pos``, ``tap_neutral``
        - ``tap_step_percent``, ``tap_step_degree``
        - ``tap_side``
        - Nominal voltage columns: ``vn_hv_kv``, ``vn_lv_kv`` (and optionally ``vn_mv_kv``)
        - ``angle_deg`` column to be updated.

    Returns
    -------
    None
        Modifies ``trafo_df`` in-place.
    """
    mask = trafo_df["tap_changer_type"].eq("Ratio")
    if not mask.any():
        return

    ratio = trafo_df.loc[mask].copy()
    ratio["tap_diff"] = ratio["tap_pos"] - ratio["tap_neutral"]
    ratio["vn_kv"] = _select_vn_kv(ratio)

    ratio["du"] = ratio["vn_kv"] * ratio["tap_diff"] * (ratio["tap_step_percent"] / 100.0)

    step_rad = np.deg2rad(ratio["tap_step_degree"])
    num = ratio["du"] * np.sin(step_rad)
    den = ratio["vn_kv"] + ratio["du"] * np.cos(step_rad)

    ratio["angle_deg"] = np.rad2deg(np.arctan2(num, den))
    trafo_df.loc[ratio.index, "angle_deg"] = ratio["angle_deg"].to_numpy()


def _apply_symmetrical_angle(trafo_df: pd.DataFrame) -> None:
    """
    Apply symmetrical tap-changer phase shifts to ``trafo_df['angle_deg']`` in-place.

    For transformers with ``tap_changer_type == 'Symmetrical'``, computes the phase shift
    introduced by the tap position using the same geometric formulation as the ratio model:

        du = vn_kv * (tap_pos - tap_neutral) * (tap_step_percent / 100)

        angle_deg = atan2(
            du * sin(tap_step_degree),
            vn_kv + du * cos(tap_step_degree)
        )  [converted to degrees]

    where ``vn_kv`` is the nominal voltage of the tapped winding side (hv/mv/lv), selected
    via :func:`_select_vn_kv`.

    Parameters
    ----------
    trafo_df : pd.DataFrame
        Transformer table to update (e.g. ``net.trafo`` or ``net.trafo3w``). Must contain:
        - ``tap_changer_type``, ``tap_pos``, ``tap_neutral``
        - ``tap_step_percent``, ``tap_step_degree``
        - ``tap_side``
        - Nominal voltage columns: ``vn_hv_kv``, ``vn_lv_kv`` (and optionally ``vn_mv_kv``)
        - ``angle_deg`` column to be updated.

    Returns
    -------
    None
        Modifies ``trafo_df`` in-place.
    """
    mask = trafo_df["tap_changer_type"].eq("Symmetrical")
    if not mask.any():
        return

    sym = trafo_df.loc[mask].copy()
    sym["tap_diff"] = sym["tap_pos"] - sym["tap_neutral"]
    sym["vn_kv"] = _select_vn_kv(sym)

    sym["du"] = sym["vn_kv"] * sym["tap_diff"] * (sym["tap_step_percent"] / 100.0)

    step_rad = np.deg2rad(sym["tap_step_degree"])
    num = sym["du"] * np.sin(step_rad)
    den = sym["vn_kv"] + sym["du"] * np.cos(step_rad)

    sym["angle_deg"] = np.rad2deg(np.arctan2(num, den))
    trafo_df.loc[sym.index, "angle_deg"] = sym["angle_deg"].to_numpy()


def populate_angle_deg_from_tap(net: pandapowerNet, trafo_df: pd.DataFrame) -> None:
    """
    Populate the ``angle_deg`` column in-place for a transformer table based on tap changer settings.

    Computes the effective phase shift introduced by tap changers for both 2-winding
    (``net.trafo``) and 3-winding (``net.trafo3w``) transformer tables by applying
    tabular, ideal, ratio, and symmetrical tap models. Missing values are filled with 0°.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network providing characteristic tables and context for tabular taps.
    trafo_df : pd.DataFrame
        Transformer table to update (either ``net.trafo`` or ``net.trafo3w``). Must contain
        tap-related columns used by the applied models.

    Returns
    -------
    None
        The input dataframe is modified in-place; no value is returned.
    """
    if trafo_df.empty:
        return

    # Ensure column exists
    trafo_df["angle_deg"] = 0.0

    _apply_tabular_angle(net, trafo_df)
    _apply_ideal_angle(trafo_df)
    _apply_ratio_angle(trafo_df)
    _apply_symmetrical_angle(trafo_df)

    trafo_df["angle_deg"] = trafo_df["angle_deg"].fillna(0)


def _elements_from_index(df: pd.DataFrame, element_type: str) -> list[str]:
    """
    Build globally unique element identifiers from a dataframe index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose index contains the pandapower table ids of the elements.
    element_type : str
        Pandapower element type (e.g. "trafo", "trafo3w", "line") used to build the identifier.

    Returns
    -------
    list[str]
        List of element unique ids generated from the index values using ``elem_node_id``.
    """
    return [elem_node_id("elem", int(idx), element_type) for idx in df.index]


def get_pst_elements_tabular(net: pandapowerNet) -> list[str]:
    """
    Return identifiers of tabular PST elements (2W and 3W) with non-zero tabular angle.

    Selects characteristic IDs from ``net.trafo_characteristic_table`` where ``angle_deg != 0``,
    then returns element ids for ``net.trafo`` and ``net.trafo3w`` that have
    ``tap_changer_type == 'Tabular'`` and reference those characteristic IDs via
    ``id_characteristic_table``.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing ``trafo``, ``trafo3w``, and
        ``trafo_characteristic_table``.

    Returns
    -------
    list[str]
        List of element identifiers (via ``_elements_from_index``) for matching
        2-winding and 3-winding transformers.
    """
    id_with_angle = net.trafo_characteristic_table.loc[
        net.trafo_characteristic_table.angle_deg != 0, "id_characteristic"
    ].unique()

    trafo = net.trafo.loc[
        (net.trafo.tap_changer_type == "Tabular") & (net.trafo.id_characteristic_table.isin(id_with_angle))
    ]
    trafo3w = net.trafo3w.loc[
        (net.trafo3w.tap_changer_type == "Tabular") & (net.trafo3w.id_characteristic_table.isin(id_with_angle))
    ]

    return _elements_from_index(trafo, "trafo") + _elements_from_index(trafo3w, "trafo3w")


def notna_and_nonzero(s: pd.Series) -> pd.Series:
    """
    Return a boolean mask indicating values that are neither NaN nor equal to zero.

    Parameters
    ----------
    s : pd.Series
        Input series to evaluate.

    Returns
    -------
    pd.Series
        Boolean mask with True where values are non-null and non-zero, False otherwise.
    """
    return s.notna() & (s != 0)


def get_pst_elements_ratio(net: pandapowerNet) -> list[str]:
    """
    Return identifiers of ratio-tap PST elements (2W and 3W) with active angle capability.

    Selects transformers from ``net.trafo`` and ``net.trafo3w`` where:
      - ``tap_changer_type == 'Ratio'``
      - ``tap_step_percent`` is non-null and non-zero
      - ``tap_step_degree`` is non-null and non-zero

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing ``trafo`` and ``trafo3w`` tables.

    Returns
    -------
    list[str]
        List of element identifiers (via ``_elements_from_index``) for matching
        2-winding and 3-winding transformers.
    """
    trafo = net.trafo.loc[
        (net.trafo.tap_changer_type == "Ratio")
        & notna_and_nonzero(net.trafo.tap_step_percent)
        & notna_and_nonzero(net.trafo.tap_step_degree)
    ]
    trafo3w = net.trafo3w.loc[
        (net.trafo3w.tap_changer_type == "Ratio")
        & notna_and_nonzero(net.trafo3w.tap_step_percent)
        & notna_and_nonzero(net.trafo3w.tap_step_degree)
    ]

    return _elements_from_index(trafo, "trafo") + _elements_from_index(trafo3w, "trafo3w")


def get_pst_elements_symmetrical(net: pandapowerNet) -> list[str]:
    """
    Return identifiers of symmetrical-tap PST elements (2W and 3W) with active angle capability.

    Selects transformers from ``net.trafo`` and ``net.trafo3w`` where:
      - ``tap_changer_type == 'Symmetrical'``
      - ``tap_step_percent`` is non-null and non-zero
      - ``tap_step_degree`` is non-null and non-zero

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing ``trafo`` and ``trafo3w`` tables.

    Returns
    -------
    list[str]
        List of element identifiers (via ``_elements_from_index``) for matching
        2-winding and 3-winding transformers.
    """
    trafo = net.trafo.loc[
        (net.trafo.tap_changer_type == "Symmetrical")
        & notna_and_nonzero(net.trafo.tap_step_percent)
        & notna_and_nonzero(net.trafo.tap_step_degree)
    ]
    trafo3w = net.trafo3w.loc[
        (net.trafo3w.tap_changer_type == "Symmetrical")
        & notna_and_nonzero(net.trafo3w.tap_step_percent)
        & notna_and_nonzero(net.trafo3w.tap_step_degree)
    ]

    return _elements_from_index(trafo, "trafo") + _elements_from_index(trafo3w, "trafo3w")


def get_pst_elements_ideal(net: pandapowerNet) -> list[str]:
    """
    Return identifiers of ideal-tap PST elements (2W and 3W) with non-zero angle steps.

    Selects transformers from ``net.trafo`` and ``net.trafo3w`` where:
      - ``tap_changer_type == 'Ideal'``
      - ``tap_step_degree`` is non-null and non-zero

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing ``trafo`` and ``trafo3w`` tables.

    Returns
    -------
    list[str]
        List of element identifiers (via ``_elements_from_index``) for matching
        2-winding and 3-winding transformers.
    """
    trafo = net.trafo.loc[(net.trafo.tap_changer_type == "Ideal") & notna_and_nonzero(net.trafo.tap_step_degree)]
    trafo3w = net.trafo3w.loc[(net.trafo3w.tap_changer_type == "Ideal") & notna_and_nonzero(net.trafo3w.tap_step_degree)]

    return _elements_from_index(trafo, "trafo") + _elements_from_index(trafo3w, "trafo3w")


def get_elements_with_pst(net: pandapowerNet) -> list[str]:
    """
    Return all element identifiers that have PST-relevant tap changer characteristics.

    Includes:
        - Tabular (non-zero angle_deg in characteristic table)
        - Ratio (tap_step_percent and tap_step_degree are non-zero OR NaN)
        - Symmetrical (tap_step_percent and tap_step_degree are non-zero OR NaN)
        - Ideal (tap_step_degree is non-zero OR NaN)

    Returns
    -------
        List[str]: element IDs like 'e_trafo_12' or 'e_trafo3w_3'.
    """
    if "trafo_characteristic_table" not in net:
        return []
    return (
        get_pst_elements_tabular(net)
        + get_pst_elements_ratio(net)
        + get_pst_elements_symmetrical(net)
        + get_pst_elements_ideal(net)
    )


def get_va_change_for_trafo(net: pandapowerNet, trafo_id: int, from_bus: int) -> float:
    """
    Compute the voltage angle change contributed by a 2-winding transformer (PST)

    when traversing it from a given bus.

    The returned value represents the signed phase shift encountered when moving
    through the transformer starting from `from_bus`. The sign depends on:
    - the traversal direction (HV → LV or LV → HV),
    - which side the tap changer is located on ("hv" or "lv"),
    - the PST angle (`angle_deg`) and the fixed shift (`shift_degree`).

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing the transformer table (`net.trafo`).
    trafo_id : int
        Index of the transformer in `net.trafo`.
    from_bus : int
        Bus id from which the path enters the transformer. This determines
        the traversal direction and therefore the sign of the angle change.

    Returns
    -------
    float
        Signed voltage angle difference (in degrees) introduced by this
        transformer when traversed from `from_bus`.
    """
    # Get transformer row (contains hv_bus, lv_bus, tap_side, angle_deg, shift_degree)
    trafo = net.trafo.loc[trafo_id]

    va_diff = 0.0

    # Traversal direction: entering from HV side (HV → LV)
    if from_bus == trafo.hv_bus:
        if trafo.tap_side == "hv":
            # Va_hv - Va_lv = angle + shift
            va_diff = trafo.angle_deg + trafo.shift_degree
        elif trafo.tap_side == "lv":
            # Va_hv - Va_lv = -angle + shift
            va_diff = -trafo.angle_deg + trafo.shift_degree

    # Traversal direction: entering from LV side (LV → HV)
    elif from_bus == trafo.lv_bus:
        if trafo.tap_side == "hv":
            # Va_lv - Va_hv = -angle - shift
            va_diff = -trafo.angle_deg - trafo.shift_degree
        elif trafo.tap_side == "lv":
            # Va_lv - Va_hv = angle - shift
            va_diff = trafo.angle_deg - trafo.shift_degree

    return va_diff


def get_va_change_tr3_hv_lv(net: pandapowerNet, trafo3w_id: int, from_bus: int, to_bus: int) -> Optional[float]:
    """
    Compute the voltage-angle change when traversing a 3-winding transformer between HV and LV sides.

    Returns the signed angle difference (Va_from - Va_to) for HV ↔ LV traversal, accounting for
    the tap position (``angle_deg``), tap location (``tap_side``), and fixed winding phase shift
    (``shift_lv_degree``). If the provided bus pair does not correspond to an HV-LV connection,
    returns ``None``.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing the ``trafo3w`` table.
    trafo3w_id : int
        Index of the 3-winding transformer in ``net.trafo3w``.
    from_bus : int
        Bus id where traversal enters the transformer.
    to_bus : int
        Bus id where traversal exits the transformer.

    Returns
    -------
    Optional[float]
        Signed voltage-angle difference in degrees for HV ↔ LV traversal, or ``None`` if the
        bus pair does not match this side combination.
    """
    trafo3w = net.trafo3w.loc[trafo3w_id]
    va_diff = None

    if from_bus == trafo3w.hv_bus and to_bus == trafo3w.lv_bus:
        if trafo3w.tap_side == "hv":
            va_diff = trafo3w.angle_deg + trafo3w.shift_lv_degree
        elif trafo3w.tap_side == "mv":
            va_diff = trafo3w.shift_lv_degree
        elif trafo3w.tap_side == "lv":
            va_diff = -trafo3w.angle_deg + trafo3w.shift_lv_degree

    elif from_bus == trafo3w.lv_bus and to_bus == trafo3w.hv_bus:
        if trafo3w.tap_side == "hv":
            va_diff = -trafo3w.angle_deg - trafo3w.shift_lv_degree
        elif trafo3w.tap_side == "mv":
            va_diff = -trafo3w.shift_lv_degree
        elif trafo3w.tap_side == "lv":
            va_diff = trafo3w.angle_deg - trafo3w.shift_lv_degree

    return va_diff


def get_va_change_tr3_hv_mv(net: pandapowerNet, trafo3w_id: int, from_bus: int, to_bus: int) -> Optional[float]:
    """
    Compute the voltage-angle change when traversing a 3-winding transformer between HV and MV sides.

    Returns the signed angle difference (Va_from - Va_to) for HV ↔ MV traversal, accounting for
    the tap-induced phase shift (``angle_deg``), the tap location (``tap_side``), and the fixed
    winding phase shift (``shift_mv_degree``). If the provided bus pair does not correspond to
    an HV-MV connection, returns ``None``.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing the ``trafo3w`` table.
    trafo3w_id : int
        Index of the 3-winding transformer in ``net.trafo3w``.
    from_bus : int
        Bus id where traversal enters the transformer.
    to_bus : int
        Bus id where traversal exits the transformer.

    Returns
    -------
    Optional[float]
        Signed voltage-angle difference in degrees for HV ↔ MV traversal, or ``None`` if the
        bus pair does not match this side combination.
    """
    trafo3w = net.trafo3w.loc[trafo3w_id]
    va_diff = None

    if from_bus == trafo3w.hv_bus and to_bus == trafo3w.mv_bus:
        if trafo3w.tap_side == "hv":
            va_diff = trafo3w.angle_deg + trafo3w.shift_mv_degree
        elif trafo3w.tap_side == "mv":
            va_diff = -trafo3w.angle_deg + trafo3w.shift_mv_degree
        elif trafo3w.tap_side == "lv":
            va_diff = trafo3w.shift_mv_degree

    elif from_bus == trafo3w.mv_bus and to_bus == trafo3w.hv_bus:
        if trafo3w.tap_side == "hv":
            va_diff = -trafo3w.angle_deg - trafo3w.shift_mv_degree
        elif trafo3w.tap_side == "mv":
            va_diff = trafo3w.angle_deg - trafo3w.shift_mv_degree
        elif trafo3w.tap_side == "lv":
            va_diff = -trafo3w.shift_mv_degree

    return va_diff


def get_va_change_tr3_mv_lv(net: pandapowerNet, trafo3w_id: int, from_bus: int, to_bus: int) -> Optional[float]:
    """
    Compute the voltage-angle change when traversing a 3-winding transformer between MV and LV sides.

    Returns the signed angle difference (Va_from - Va_to) for MV ↔ LV traversal, accounting for
    the tap-induced phase shift (``angle_deg``), the tap location (``tap_side``), and the fixed
    winding phase shifts (``shift_mv_degree`` and ``shift_lv_degree``). If the provided bus pair
    does not correspond to an MV-LV connection, returns ``None``.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing the ``trafo3w`` table.
    trafo3w_id : int
        Index of the 3-winding transformer in ``net.trafo3w``.
    from_bus : int
        Bus id where traversal enters the transformer.
    to_bus : int
        Bus id where traversal exits the transformer.

    Returns
    -------
    Optional[float]
        Signed voltage-angle difference in degrees for MV ↔ LV traversal, or ``None`` if the
        bus pair does not match this side combination.
    """
    trafo3w = net.trafo3w.loc[trafo3w_id]
    va_diff = None

    if from_bus == trafo3w.mv_bus and to_bus == trafo3w.lv_bus:
        if trafo3w.tap_side == "hv":
            va_diff = trafo3w.shift_lv_degree - trafo3w.shift_mv_degree
        elif trafo3w.tap_side == "mv":
            va_diff = trafo3w.angle_deg + trafo3w.shift_lv_degree - trafo3w.shift_mv_degree
        elif trafo3w.tap_side == "lv":
            va_diff = -trafo3w.angle_deg + trafo3w.shift_lv_degree - trafo3w.shift_mv_degree

    elif from_bus == trafo3w.lv_bus and to_bus == trafo3w.mv_bus:
        if trafo3w.tap_side == "hv":
            va_diff = -trafo3w.shift_lv_degree + trafo3w.shift_mv_degree
        elif trafo3w.tap_side == "mv":
            va_diff = -trafo3w.angle_deg - trafo3w.shift_lv_degree + trafo3w.shift_mv_degree
        elif trafo3w.tap_side == "lv":
            va_diff = trafo3w.angle_deg - trafo3w.shift_lv_degree + trafo3w.shift_mv_degree

    return va_diff


def get_va_change_for_tr3(net: pandapowerNet, trafo3w_id: int, from_bus: int, to_bus: int) -> float:
    """
    Compute the voltage angle change introduced by a 3-winding transformer (PST)

    when traversed between two specific sides (HV/MV/LV).

    The function returns the signed phase shift encountered when moving through
    the transformer from `from_bus` to `to_bus`. The value depends on:
    - traversal direction (e.g., HV → LV vs LV → HV),
    - which side the tap changer is located on ("hv", "mv", "lv"),
    - the tap-induced angle (`angle_deg`),
    - fixed phase shifts between windings (`shift_mv_degree`, `shift_lv_degree`).

    Sign convention:
    - The result represents: Va_from_bus - Va_to_bus.
    - Reversing the traversal direction negates the angle difference.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing the 3-winding transformer table (`net.trafo3w`).
    trafo3w_id : int
        Index of the transformer in `net.trafo3w`.
    from_bus : int
        Bus id where the path enters the transformer.
    to_bus : int
        Bus id where the path exits the transformer.

    Returns
    -------
    float
        Signed voltage angle difference (in degrees) introduced by the transformer
        when moving from `from_bus` to `to_bus`.
    """
    for fn in (get_va_change_tr3_hv_lv, get_va_change_tr3_hv_mv, get_va_change_tr3_mv_lv):
        va = fn(net, trafo3w_id, from_bus, to_bus)
        if va is not None:
            return va

    return 0.0


def get_va_change(net: pandapowerNet, elements: list[str]) -> float:
    """
    Compute cumulative voltage angle change introduced by PST transformers

    along a path in the network graph.

    The function walks through the sequence of nodes representing a path
    (typically produced by a shortest-path search). For each element node
    (e.g., trafo or trafo3w), it determines the direction of traversal using
    the previous and next nodes in the path and accumulates the corresponding
    phase shift contribution.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing transformer data used to determine
        PST phase shifts.
    elements : list[str]
        Ordered list of graph node identifiers representing a path
        (e.g., bus → element → bus → element → bus).

    Returns
    -------
    float
        Total voltage angle change (in degrees) introduced by PST transformers
        encountered along the path.
    """
    total_va_change = 0.0

    # Iterate over path nodes by index so we can access neighbors
    # (previous and next nodes determine direction through the element)
    for ind in range(1, len(elements) - 1):
        el = elements[ind]

        # Extract element id from the current node
        el_id = get_node_table_id(el)

        # If this node represents a 2-winding transformer,
        # accumulate its PST angle contribution
        if is_node_of_element_type(el, "trafo"):
            # Previous and next nodes in the path correspond to buses
            # surrounding the current element node
            from_bus = get_node_table_id(elements[ind - 1])
            total_va_change += get_va_change_for_trafo(net, el_id, from_bus)

        # If this node represents a 3-winding transformer,
        # accumulate its PST angle contribution
        if is_node_of_element_type(el, "trafo3w"):
            # Previous and next nodes in the path correspond to buses
            # surrounding the current element node
            from_bus = get_node_table_id(elements[ind - 1])
            to_bus = get_node_table_id(elements[ind + 1])
            total_va_change += get_va_change_for_tr3(net, el_id, from_bus, to_bus)

    return total_va_change


def _build_va_change_df_for_group(
    net: pandapowerNet,
    component_group: pd.DataFrame,
    graph: nx.Graph,
    elements_with_pst: set,
) -> pd.DataFrame:
    """
    Construct a matrix of PST-induced voltage angle changes between buses

    within a single connected component.

    For each pair of buses in the component, the function:
    1. Finds the shortest path between their corresponding "another_side" nodes.
    2. Checks whether the path contains any elements with phase-shifting
       transformers (PST).
    3. If a PST is present on the path, computes the cumulative voltage angle
       change along that path using `get_va_change`.
    4. Stores the resulting angle correction in a bus-to-bus matrix.

    Returns
    -------
    pd.DataFrame
        Square DataFrame indexed and columned by bus ids from the component.
        Each cell (i, j) contains the cumulative PST-induced voltage angle
        change along the shortest path between the corresponding buses.
    """
    # All buses participating in this connected component
    buses = component_group["bus"].unique()

    # Initialize square matrix (bus x bus) with zeros.
    # Will store cumulative PST angle changes between each bus pair.
    va_change_df = pd.DataFrame(
        0.0,
        index=buses,
        columns=buses,
    )

    # Materialize rows to avoid recreating the generator for nested iteration
    rows = list(component_group.itertuples())

    # Outer loop: treat each row as a source
    for i_row in rows:
        # Convert "another_side" bus to a graph node id
        src = elem_node_id(kind="bus", idx=int(i_row.deenergized_sw_side))

        # Compute shortest paths from this source to all reachable nodes once
        # Result: dict[dst_node] -> [path nodes]
        paths = nx.single_source_shortest_path(graph, src)

        # Inner loop: treat each row as a destination
        for j_row in rows:
            dst = elem_node_id(kind="bus", idx=int(j_row.deenergized_sw_side))

            # Skip self-pair
            if src == dst:
                continue

            # Get shortest path from src to dst (if reachable)
            path = paths.get(dst)
            if path is None:
                # No path exists in the graph
                continue

            # Check whether the path crosses any PST element
            if any(n in elements_with_pst for n in path):
                # Compute total phase shift introduced along this path
                va_change = get_va_change(net, path)

                # Store result using original bus ids as matrix indices
                va_change_df.loc[i_row.bus, j_row.bus] = va_change

    return va_change_df


def _compute_va_diff_pst(
    net: pandapowerNet,
    df_rows: pd.DataFrame,
    va_deg: pd.Series,
    graph: nx.Graph,
) -> pd.Series:
    """
    Compute voltage angle difference (va_diff) per switch for one-sided cases,

    accounting for phase-shifting transformers (PST).

    For each connected component:
    1. Take all buses belonging to that component.
    2. Build a pairwise voltage angle difference matrix using `va_deg`.
    3. Build a correction matrix (`va_change_df`) representing cumulative
       angle shifts introduced by PST elements along shortest paths between
       buses in the component.
    4. Subtract PST-induced shifts from the raw angle differences.
    5. For each bus, take the maximum absolute corrected difference.
    6. Map that value back to switches connected to the bus.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network. PST phase shifts are applied in-place before
        computing corrections.
    df_rows : pd.DataFrame
        Table describing one-sided switch cases. Must contain:
        - "switch": switch id
        - "bus": bus id associated with the switch
        - "another_side": opposite-side bus id used for path search
        - "comp": connected component identifier
    va_deg : pd.Series
        Bus voltage angles in degrees, indexed by bus id.
    graph : nx.Graph
        Topology graph containing buses and elements. Used to compute
        shortest paths between buses and detect PST elements along paths.

    Returns
    -------
    pd.Series
        Series indexed by switch id with inferred voltage angle difference
        ("va_diff") after compensating for PST-induced angle shifts.
    """
    # Nothing to compute
    if df_rows.empty:
        return pd.Series(dtype=float, name="va_diff")

    # Apply PST phase shifts to the network model so angle corrections are consistent
    populate_angle_deg_from_tap(net, net.trafo)
    populate_angle_deg_from_tap(net, net.trafo3w)

    # Will store the final max corrected angle difference per bus
    bus_va_res: dict[int, float] = {}

    # Precompute elements that introduce phase shift (for fast membership checks)
    elements_with_pst = set(get_elements_with_pst(net))

    # Process each connected component independently
    for _, comp_group in df_rows.groupby("comp", sort=False):
        # Buses belonging to this component
        buses = comp_group["bus"].unique()

        # Build pairwise voltage angle difference matrix:
        # diff[i, j] = va[i] - va[j]
        va = va_deg.loc[buses].to_numpy()
        diff = va[:, None] - va[None, :]
        re_diff = pd.DataFrame(diff, index=buses, columns=buses)

        # Build matrix of PST-induced angle changes along shortest paths
        # between buses within this component
        va_change_df = _build_va_change_df_for_group(
            net=net,
            component_group=comp_group,
            graph=graph,
            elements_with_pst=elements_with_pst,
        )

        # Correct raw angle differences by subtracting PST contributions
        res_df = re_diff - va_change_df

        # For each bus, take the maximum absolute corrected difference
        # across all other buses in the same component
        row_max = res_df.abs().max(axis=1)

        # Store results keyed by bus id
        bus_va_res.update(row_max.to_dict())

    # Map bus-level results back to switches and return a Series indexed by switch id
    return df_rows.assign(va_diff=df_rows["bus"].map(bus_va_res)).set_index("switch")["va_diff"].rename("va_diff")


def _combine_switch_va_diffs(*frames: Optional[pl.DataFrame]) -> pl.DataFrame:
    """
    Combine multiple ``switch`` / ``va_diff`` frames into one, removing duplicate switch ids.

    Concatenates all provided frames (ignoring ``None``/empty), then de-duplicates by
    ``switch``, keeping the first occurrence.

    Parameters
    ----------
    *frames : Optional[pl.DataFrame]
        Frames with columns ``switch`` / ``va_diff``. ``None`` and empty frames are ignored.

    Returns
    -------
    pl.DataFrame
        A single ``switch`` / ``va_diff`` frame. Empty if all inputs are empty or ``None``.
    """
    non_empty = [f for f in frames if f is not None and not f.is_empty()]
    if not non_empty:
        return pl.DataFrame(schema={"switch": pl.Int64, "va_diff": pl.Float64})
    return pl.concat(non_empty, how="vertical").unique(subset="switch", keep="first", maintain_order=True)


def _format_switch_va_diff_output(
    va_diff_by_switch: pl.DataFrame,
    timestep: int,
    contingency: PandapowerContingency,
) -> pl.DataFrame:
    """
    Format per-switch VA-diff values into a flat, schema-aligned polars frame.

    Turns a ``switch`` / ``va_diff`` frame into ``timestep`` / ``contingency`` / ``element`` /
    ``va_diff``, where ``element`` is the globally unique switch identifier
    (``"<switch_id><sep>switch"``).

    Parameters
    ----------
    va_diff_by_switch : pl.DataFrame
        Frame with columns ``switch`` / ``va_diff``.
    timestep : int
        Timestep key to include.
    contingency : PandapowerContingency
        Contingency providing the ``unique_id``.

    Returns
    -------
    pl.DataFrame
        Columns ``timestep`` / ``contingency`` / ``element`` / ``va_diff``. Empty (with those
        columns) if the input is empty.
    """
    return va_diff_by_switch.select(
        pl.lit(timestep, dtype=pl.Int64).alias("timestep"),
        pl.lit(contingency.unique_id).alias("contingency"),
        (pl.col("switch").cast(pl.Utf8) + f"{SEPARATOR}switch").alias("element"),
        pl.col("va_diff"),
    )


def _apply_contingency_va_diff_info(
    net: pandapowerNet,
    va_diff_df: pd.DataFrame,
    timestep: int,
    contingency: PandapowerContingency,
) -> pd.DataFrame:
    """
    Apply explicit per-switch va_diff overrides/additions from contingency.va_diff_info.

    Preserves your sign convention (to-side gets -va_diff).

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network with loadflow results available in ``net.res_bus['va_degree']``.
    va_diff_df : pd.DataFrame
        Target dataframe to update. Must be indexed by ``(timestep, contingency, element)``
        where ``element`` is the switch unique id, and contain columns ``['va_diff', 'element_name']``.
    timestep : int
        Timestep key used in the multi-index of ``va_diff_df``.
    contingency : PandapowerContingency
        Contingency providing ``unique_id`` and optionally ``va_diff_info`` (a list of ``VADiffInfo``),
        where each entry contains ``from_bus``, ``to_bus``, ``power_switches_from`` and ``power_switches_to``.

    Returns
    -------
    pd.DataFrame
        The updated ``va_diff_df`` with the contingency-specific va-diff values applied.
    """
    if not getattr(contingency, "va_diff_info", None):
        return va_diff_df

    for va_diff_info in contingency.va_diff_info:
        va_diff = net.res_bus.loc[va_diff_info.from_bus, "va_degree"] - net.res_bus.loc[va_diff_info.to_bus, "va_degree"]

        for switch_id, switch_name in va_diff_info.power_switches_from.items():
            va_diff_df.loc[
                (timestep, contingency.unique_id, switch_id),
                ["va_diff", "element_name"],
            ] = va_diff, switch_name

        for switch_id, switch_name in va_diff_info.power_switches_to.items():
            va_diff_df.loc[
                (timestep, contingency.unique_id, switch_id),
                ["va_diff", "element_name"],
            ] = -va_diff, switch_name

    return va_diff_df


def get_va_diff_results(
    net: pandapowerNet,
    timestep: int,
    contingency: PandapowerContingency,
    result_constants: ResultConstants,
) -> pl.DataFrame:
    """Get the voltage angle difference results for the given network and contingency.

    Parameters
    ----------
    net : pandapowerNet
        The network to compute the voltage angle difference results for
    timestep : int
        The timestep of the results
    contingency : PandapowerContingency
        The contingency to compute the voltage angle difference results for.
        Will also calculate the va_diff of the outaged elements if they are lines or transformers
    result_constants : ResultConstants
        Per-run constants: the ANGLE-scoped switch ids and switch name map (projections of the
        monitored-element table) plus the shared connectivity-graph cache. Building the graph
        dominates this function's runtime and the projections would otherwise be recomputed
        for every outage.

    Returns
    -------
    pl.DataFrame
        The voltage angle difference results as a flat polars frame (``timestep``/
        ``contingency``/``element`` are ordinary columns). The graph traversal and PST tap
        math remain pandas/networkx internally; only the small output is returned as polars.

    """
    # VA diff is computed for all open CBs in the network before filtering to monitored switches.
    # This is required for correctness: an outage group may contain multiple open switches, and
    # the algorithm picks the worst-case angle difference across all of them. If we computed only
    # for monitored switches, a group with 4 switches where only 2 are monitored would miss the
    # contribution of the unmonitored 2, producing an incorrect (too low) result for the monitored ones.
    # Boundary read of pandapower's pandas tables (net.switch, net.res_bus), converted once:
    # the open-CB selection and the va lookup then flow through the polars/numpy logic.
    switch_pl = pl.from_pandas(net.switch[["bus", "element", "type", "closed", "et"]].rename_axis("switch_id").reset_index())
    open_cb = switch_pl.filter((pl.col("type") == "CB") & (~pl.col("closed")) & (pl.col("et") == "b")).select(
        "switch_id", "bus", "element"
    )

    va_deg = _get_bus_va_series(net)  # pandas Series; consumed only by the PST/networkx core
    va_frame = pl.DataFrame(
        {"bus": va_deg.index.to_numpy(), "va": va_deg.to_numpy()},
        schema={"bus": pl.Int64, "va": pl.Float64},
    )
    va_by_bus = {int(bus): float(va) for bus, va in va_deg.items()}

    va_diff_both, open_cb_rest = _compute_va_diff_both_ends(open_cb, va_frame)
    # Normally a cache hit: outages only open CB switches and de-energize elements, neither
    # of which changes this graph.
    graph_cache = result_constants.graph_cache
    graph, connected_components = graph_cache.get(net)
    rows_df = _make_one_sided_rows(net, open_cb_rest, va_frame, connected_components, graph_cache)

    not_pst_rows_df = rows_df.filter(~pl.col("pst"))
    va_diff_one_side = _compute_va_diff_one_side(not_pst_rows_df, va_by_bus)

    # The PST path is a networkx shortest-path traversal with per-element tap math that reads
    # the pandas net tables; it stays pandas/networkx. Hand it the rows as pandas and take its
    # per-switch Series back into polars.
    pst_rows_df = rows_df.filter(pl.col("pst"))
    va_diff_pst_series = _compute_va_diff_pst(net, pst_rows_df.to_pandas(), va_deg, graph)
    va_diff_pst = pl.DataFrame(
        {"switch": va_diff_pst_series.index.to_numpy(), "va_diff": va_diff_pst_series.to_numpy()},
        schema={"switch": pl.Int64, "va_diff": pl.Float64},
    )

    va_diff_by_switch = _combine_switch_va_diffs(va_diff_both, va_diff_one_side, va_diff_pst)
    va_diff_by_switch = va_diff_by_switch.filter(pl.col("switch").is_in(result_constants.angle_switch_table_ids.tolist()))

    out = _format_switch_va_diff_output(va_diff_by_switch, timestep, contingency)
    if out.is_empty():
        return _empty_va_diff_polars()

    # Attach the monitored switch display name (keyed by globally unique element id) and the
    # contingency name, then return the flat polars frame directly - no pandas round-trip.
    name_frame = pl.DataFrame(
        {
            "element": list(result_constants.switch_name_map.keys()),
            "element_name": list(result_constants.switch_name_map.values()),
        },
        schema={"element": pl.Utf8, "element_name": pl.Utf8},
    )
    return (
        out.join(name_frame, on="element", how="left")
        .with_columns(
            pl.col("element_name").fill_null(""),
            pl.lit(contingency.name or "").alias("contingency_name"),
        )
        .select("timestep", "contingency", "element", "va_diff", "element_name", "contingency_name")
    )


def get_failed_va_diff_results(
    timestep: int, contingency: PandapowerContingency, result_constants: ResultConstants
) -> pl.DataFrame:
    """Get the voltage angle difference results for the given network and contingency when the loadflow failed.

    This will return NaN for all elements that were monitored and the contingency.

    Parameters
    ----------
    timestep : int
        The timestep of the results
    contingency : PandapowerContingency
        The contingency to compute the voltage angle difference results for.
    result_constants : ResultConstants
        Per-run constants; supplies the globally unique ids of the ANGLE-scoped switches.

    Returns
    -------
    pl.DataFrame
        The voltage angle difference results for the given network and contingency when the loadflow failed.
        This will return NaN for all elements that were monitored and the contingency.
    """
    monitored_switches = result_constants.angle_switch_element_ids
    all_power_switches = {}
    for va_diff_info in contingency.va_diff_info:
        all_power_switches.update(va_diff_info.power_switches_from)
        all_power_switches.update(va_diff_info.power_switches_to)
    outaged_elements = list(all_power_switches.keys())
    elements = [*monitored_switches, *outaged_elements]
    failed_va_diff_results = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [[timestep], [contingency.unique_id], elements],
            names=["timestep", "contingency", "element"],
        )
    ).assign(va_diff=np.nan)
    # fill missing columns with NaN
    failed_va_diff_results["element_name"] = (
        failed_va_diff_results.index.get_level_values("element").map(all_power_switches).fillna("")
    )
    failed_va_diff_results["contingency_name"] = ""
    if failed_va_diff_results.empty:
        return _empty_va_diff_polars()
    return pl.from_pandas(failed_va_diff_results.reset_index())
