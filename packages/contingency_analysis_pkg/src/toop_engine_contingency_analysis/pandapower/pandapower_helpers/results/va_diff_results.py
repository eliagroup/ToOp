# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


"""Utilities for extracting pandapower VA diff simulation results per contingency."""

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
    PandapowerMonitoredElementSchema,
)
from toop_engine_grid_helpers.pandapower.outage_group import (
    build_connectivity_graph_for_contingency,
    elem_node_id,
    get_node_table_id,
    is_node_of_element_type,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    get_globally_unique_id_from_index,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import (
    VADiffResultSchema,
)


def _select_monitored_open_cb_bus_bus_switches(
    net: pandapowerNet,
    monitored_elements: pd.DataFrame,
) -> pd.DataFrame:
    monitored_switch_ids = monitored_elements.query("kind == 'switch'")["table_id"]
    switches = net.switch.loc[net.switch.index.isin(monitored_switch_ids)]

    open_cb = switches.loc[(switches["type"] == "CB") & (~switches["closed"]) & (switches["et"] == "b")]
    return open_cb


def _get_bus_va_series(net: pandapowerNet) -> pd.Series:
    """Series indexed by bus id with va_degree, dropping NaNs."""
    res_bus = net.res_bus.dropna(subset=["va_degree"])
    return res_bus["va_degree"]


def _compute_va_diff_both_ends(
    open_cb: pd.DataFrame,
    va_deg: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Compute voltage angle differences for open circuit breakers where

    voltage angle results are available on both sides.

    Parameters
    ----------
    open_cb : pandas.DataFrame
        DataFrame in the same format as ``net.switch`` containing **only
        open circuit breakers**. The following columns are required:
        The DataFrame index is assumed to be the switch index.

    va_deg : pandas.Series
        Voltage angle results in degrees, in the same format as
        ``net.res_bus.va_degree``. The Series index must be bus indices
        and the values are voltage angles in degrees.

    Returns
    -------
    va_diff : pandas.Series
        Voltage angle difference ``va(bus) - va(element)`` for all open
        switches where voltage angle results are available on **both**
        connected buses. The Series is indexed by switch index.

    open_cb_rest : pandas.DataFrame
     Subset of ``open_cb`` containing switches for which voltage angle
     results are **not available on both sides**. These switches may
     be handled by one-sided or outage-group inference logic.
    """
    mask_both = open_cb["bus"].isin(va_deg.index) & open_cb["element"].isin(va_deg.index)
    cb_both = open_cb.loc[mask_both].copy()

    if cb_both.empty:
        va_diff_both = pd.Series(dtype=float, name="va_diff")
    else:
        a = va_deg.loc[cb_both["bus"]].to_numpy()
        b = va_deg.loc[cb_both["element"]].to_numpy()
        va_diff_both = pd.Series(a - b, index=cb_both.index, name="va_diff")

    open_cb_rest = open_cb.loc[~mask_both]
    return va_diff_both, open_cb_rest


def _build_node_to_component_map(connected_components: list) -> dict[str, int]:
    return {node: comp_idx for comp_idx, component in enumerate(connected_components) for node in component}


def _build_side(df: pd.DataFrame, key_from: str, bus_from: str, node_to_component: dict) -> pd.DataFrame:
    """
    Build a normalized (component, switch, bus) table for one side of a bus-bus switch.

    This helper maps the *other* end of a bus-bus connection into a connected
    component, while keeping the known bus on the current side. It is used to
    construct per-switch connectivity rows in a vectorized manner.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame representing one side of bus-bus switches.
        Must contain columns referenced by `key_from` and `bus_from`.
        The DataFrame index is assumed to represent the switch identifier.

    key_from : str
        Name of the column whose values identify the **other-end bus** of the
        switch. These values are converted into component lookup keys of the
        form `"b_<bus_id>"` and mapped via `node_to_component`.

    bus_from : str
        Name of the column whose values identify the **known bus on this side**
        of the switch. These values are used to populate the output `bus` column.

    node_to_component : dict[str, Any]
        Mapping from node identifiers (e.g. `"b_12"`) to connected component
        identifiers. Rows whose lookup key is not present in the mapping are
        dropped.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:

        - comp: connected component identifier
        - switch: switch identifier (copied from `df.index`)
        - bus: bus index on the current side of the switch

        Rows without a matching component are excluded.
    """
    out = df.copy()

    # build component lookup
    out["comp_key"] = "b_" + out[key_from].astype(str)
    out["comp"] = out["comp_key"].map(node_to_component)

    # keep switch id from index
    out["switch"] = out.index

    out["another_side"] = out[key_from].astype(int)

    # output bus
    out["bus"] = out[bus_from].astype(int)

    # drop those without a component match
    out = out.dropna(subset=["comp"])

    return out[["comp", "switch", "bus", "another_side"]]


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
    open_cb_rest: pd.DataFrame,
    va_deg: pd.Series,
    connected_components: list,
) -> pd.DataFrame:
    """
    Build rows for one-sided cases.

    Output columns: switch, bus (the bus that HAS va), comp (component of the OTHER side).
    """
    node_to_component = _build_node_to_component_map(connected_components)
    mask_bus_side = open_cb_rest["bus"].isin(va_deg.index)  # bus side has va
    mask_element_side = open_cb_rest["element"].isin(va_deg.index)  # element side has va

    cb_bus_side = open_cb_rest.loc[mask_bus_side]
    cb_element_side = open_cb_rest.loc[mask_element_side]

    # element-side: known va bus is r.element; component from OTHER end r.bus
    cb_element_side = _build_side(
        cb_element_side,
        key_from="bus",
        bus_from="element",
        node_to_component=node_to_component,
    )

    # bus-side: known va bus is r.bus; component from OTHER end r.element
    cb_bus_side = _build_side(
        cb_bus_side,
        key_from="element",
        bus_from="bus",
        node_to_component=node_to_component,
    )
    _, out_gr_with_pst_ids = get_outage_groups_with_pst(net, connected_components)

    result = pd.concat([cb_bus_side, cb_element_side], ignore_index=True)
    result["pst"] = result["comp"].isin(out_gr_with_pst_ids)
    return result


def _filter_components_with_enough_va(df_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep only components where we have >= 2 distinct buses with va."""
    if df_rows.empty:
        return df_rows
    valid_comps = df_rows.groupby("comp")["bus"].nunique().loc[lambda s: s >= 2].index
    return df_rows.loc[df_rows["comp"].isin(valid_comps)]


def _max_abs_diff_per_bus_within_component(
    buses: np.ndarray,
    va_deg: pd.Series,
) -> dict[int, float]:
    """
    For a set of buses (with va), compute for each bus:

      max_{other bus in set} |va(bus) - va(other)|
    """
    va = va_deg.loc[buses].to_numpy()  # (m,)
    diff = np.abs(va[:, None] - va[None, :])  # (m, m)
    np.fill_diagonal(diff, -np.inf)
    max_per_bus = diff.max(axis=1)
    return dict(zip(map(int, buses), map(float, max_per_bus), strict=True))


def _compute_va_diff_one_side(
    df_rows: pd.DataFrame,
    va_deg: pd.Series,
) -> pd.Series:
    """Return Series indexed by switch id with inferred va_diff for one-sided cases."""
    if df_rows.empty:
        return pd.Series(dtype=float, name="va_diff")

    df_rows = _filter_components_with_enough_va(df_rows)
    if df_rows.empty:
        return pd.Series(dtype=float, name="va_diff")

    bus_va_res: dict[int, float] = {}

    for _, g in df_rows.groupby("comp", sort=False):
        buses = g["bus"].unique()
        bus_va_res.update(_max_abs_diff_per_bus_within_component(buses, va_deg))

    return df_rows.assign(va_diff=df_rows["bus"].map(bus_va_res)).set_index("switch")["va_diff"].rename("va_diff")


def _apply_tabular_angle(net: pandapowerNet, trafo_df: pd.DataFrame) -> None:
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
    mask = trafo_df["tap_changer_type"].eq("Ideal")
    if not mask.any():
        return

    ideal = trafo_df.loc[mask].copy()
    ideal["tap_diff"] = ideal["tap_pos"] - ideal["tap_neutral"]
    ideal["angle_deg"] = ideal["tap_diff"] * ideal["tap_step_degree"]

    trafo_df.loc[ideal.index, "angle_deg"] = ideal["angle_deg"].to_numpy()


def _select_vn_kv(df: pd.DataFrame) -> np.ndarray:
    """
    Select nominal voltage of tapped side (hv/mv/lv).

    Works for both 2W (no vn_mv_kv) and 3W (vn_mv_kv present).
    """
    has_mv = "vn_mv_kv" in df.columns
    mv = df["vn_mv_kv"] if has_mv else np.nan

    return np.select(
        [df["tap_side"].eq("hv"), df["tap_side"].eq("mv"), df["tap_side"].eq("lv")],
        [df["vn_hv_kv"], mv, df["vn_lv_kv"]],
        default=np.nan,
    )


def _apply_ratio_angle(trafo_df: pd.DataFrame) -> None:
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
    Populate `angle_deg` for a transformer table (2-winding or 3-winding) in-place.

    Supports both:
      - net.trafo   (2-winding transformers)
      - net.trafo3w (3-winding transformers)
    """
    if trafo_df.empty:
        return

    # Ensure column exists
    trafo_df["angle_deg"] = 0.0

    _apply_tabular_angle(net, trafo_df)
    _apply_ideal_angle(trafo_df)
    _apply_ratio_angle(trafo_df)
    _apply_symmetrical_angle(trafo_df)


def _elements_from_index(df: pd.DataFrame, element_type: str) -> list[str]:
    """Build element identifiers from a dataframe index."""
    return [elem_node_id("elem", int(idx), element_type) for idx in df.index]


def get_pst_elements_tabular(net: pandapowerNet) -> list[str]:
    """
    Return PST elements with tap_changer_type == 'Tabular' for trafo and trafo3w.

    Logic:
        - Uses trafo_characteristic_table: selects characteristic IDs where angle_deg != 0
        - Selects trafos / trafo3w that use those characteristic table IDs
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

    return _elements_from_index(trafo, "e_trafo_") + _elements_from_index(trafo3w, "e_trafo3w_")


def notna_and_nonzero(s: pd.Series) -> pd.Series:
    """Return a boolean mask where values are not NaN and not equal to 0."""
    return s.notna() & (s != 0)


def get_pst_elements_ratio(net: pandapowerNet) -> list[str]:
    """
    Return PST elements with tap_changer_type == 'Ratio' for trafo and trafo3w.

    Logic:
        - tap_step_percent is non-zero OR NaN
        - tap_step_degree  is non-zero OR NaN
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
    Return PST elements with tap_changer_type == 'Symmetrical' for trafo and trafo3w.

    Logic:
        - tap_step_percent is non-zero OR NaN
        - tap_step_degree  is non-zero OR NaN
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

    return _elements_from_index(trafo, "e_trafo_") + _elements_from_index(trafo3w, "e_trafo3w_")


def get_pst_elements_ideal(net: pandapowerNet) -> list[str]:
    """
    Return PST elements with tap_changer_type == 'Ideal' for trafo and trafo3w.

    Logic:
        - tap_step_degree is non-zero OR NaN
    """
    trafo = net.trafo.loc[(net.trafo.tap_changer_type == "Ideal") & notna_and_nonzero(net.trafo.tap_step_degree)]
    trafo3w = net.trafo3w.loc[(net.trafo3w.tap_changer_type == "Ideal") & notna_and_nonzero(net.trafo3w.tap_step_degree)]

    return _elements_from_index(trafo, "e_trafo_") + _elements_from_index(trafo3w, "e_trafo3w_")


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
    """Voltage angle change for 3W transformer when traversing HV ↔ LV (Va_from - Va_to)."""
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
    """Voltage angle change for 3W transformer when traversing HV ↔ MV (Va_from - Va_to)."""
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
    """Voltage angle change for 3W transformer when traversing MV ↔ LV (Va_from - Va_to)."""
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
        src = elem_node_id(kind="bus", idx=int(i_row.another_side))

        # Compute shortest paths from this source to all reachable nodes once
        # Result: dict[dst_node] -> [path nodes]
        paths = nx.single_source_shortest_path(graph, src)

        # Inner loop: treat each row as a destination
        for j_row in rows:
            dst = elem_node_id(kind="bus", idx=int(j_row.another_side))

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


def _combine_switch_va_diffs(*series: pd.Series) -> pd.Series:
    """Concat and de-duplicate by switch id (keep first)."""
    s = pd.concat([x for x in series if x is not None], axis=0)
    if s.empty:
        return pd.Series(dtype=float, name="va_diff")
    s = s[~s.index.duplicated(keep="first")]
    s.name = "va_diff"
    return s


def _format_switch_va_diff_output(
    va_diff_by_switch: pd.Series,
    timestep: int,
    contingency: PandapowerContingency,
) -> pd.DataFrame:
    """
    Shape into schema-compatible multi-index DF with index:

      (timestep, contingency, element)
    where element is "{switch_id}%%switch"
    """
    if va_diff_by_switch.empty:
        out = pd.DataFrame(columns=["va_diff"])
    else:
        out = va_diff_by_switch.to_frame()

    out["timestep"] = timestep
    out["contingency"] = contingency.unique_id
    out["element"] = get_globally_unique_id_from_index(out.index, "switch")
    out = out.set_index(["timestep", "contingency", "element"])[["va_diff"]]
    return out


def _apply_contingency_va_diff_info(
    net: pandapowerNet,
    va_diff_df: pd.DataFrame,
    timestep: int,
    contingency: PandapowerContingency,
) -> pd.DataFrame:
    """
    Apply explicit per-switch va_diff overrides/additions from contingency.va_diff_info.

    Preserves your sign convention (to-side gets -va_diff).
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


@pa.check_types
def get_va_diff_results(
    net: pandapowerNet,
    timestep: int,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    contingency: PandapowerContingency,
) -> pat.DataFrame[VADiffResultSchema]:
    """Get the voltage angle difference results for the given network and contingency.

    Parameters
    ----------
    net : pandapowerNet
        The network to compute the voltage angle difference results for
    timestep : int
        The timestep of the results
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
        The dataframe containing the monitored elements
    contingency : PandapowerContingency
        The contingency to compute the voltage angle difference results for.
        Will also calculate the va_diff of the outaged elements if they are lines or transformers

    Returns
    -------
    pat.DataFrame[VADiffResultSchema]
        The voltage angle difference results for the given network and contingency

    """
    open_cb = _select_monitored_open_cb_bus_bus_switches(net, monitored_elements)
    va_deg = _get_bus_va_series(net)

    va_diff_both, open_cb_rest = _compute_va_diff_both_ends(open_cb, va_deg)
    graph = build_connectivity_graph_for_contingency(net)
    connected_components = list(nx.connected_components(graph))
    rows_df = _make_one_sided_rows(net, open_cb_rest, va_deg, connected_components)

    not_pst_rows_df = rows_df[~rows_df["pst"]]
    va_diff_one_side = _compute_va_diff_one_side(not_pst_rows_df, va_deg)

    pst_rows_df = rows_df[rows_df["pst"]]
    va_diff_pst = _compute_va_diff_pst(net, pst_rows_df, va_deg, graph)

    va_diff_by_switch = _combine_switch_va_diffs(va_diff_both, va_diff_one_side, va_diff_pst)

    out = _format_switch_va_diff_output(va_diff_by_switch, timestep, contingency)

    va_diff_df = get_empty_dataframe_from_model(VADiffResultSchema)
    if out.empty:
        return va_diff_df
    va_diff_df = pd.concat([va_diff_df, out], axis=0)

    va_diff_df = _apply_contingency_va_diff_info(net, va_diff_df, timestep, contingency)

    return va_diff_df


@pa.check_types
def get_failed_va_diff_results(
    timestep: int, monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema], contingency: PandapowerContingency
) -> pat.DataFrame[VADiffResultSchema]:
    """Get the voltage angle difference results for the given network and contingency when the loadflow failed.

    This will return NaN for all elements that were monitored and the contingency.

    Parameters
    ----------
    timestep : int
        The timestep of the results
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
        The list of monitored elements including switches
    contingency : Contingency
        The contingency to compute the voltage angle difference results for.
    contingency: PandapowerContingency

    Returns
    -------
    pat.DataFrame[VADiffResultSchema]
        The voltage angle difference results for the given network and contingency when the loadflow failed.
        This will return NaN for all elements that were monitored and the contingency.
    """
    monitored_switches = monitored_elements.query("kind == 'switch'").index.to_list()
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
    return failed_va_diff_results
