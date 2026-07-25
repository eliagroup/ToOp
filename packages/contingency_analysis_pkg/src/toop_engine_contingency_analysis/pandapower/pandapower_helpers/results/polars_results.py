# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Polars result extraction for branch and node results.

Branch and node results are rebuilt for every contingency, but most of what goes into
them never changes: element ids, rated currents, bus voltage levels and the base-case
voltages all depend on the static network only. :class:`ResultConstants` computes those
once per job; the per-outage functions then only touch the numbers the load flow actually
changed.

The frames are polars with ``timestep`` / ``contingency`` / ``element`` (/ ``side``) as
plain columns and stay that way through the whole pipeline: results are concatenated in
polars and converted to pandas exactly once, at the end of the run (see
``convert_polars_loadflow_results_to_pandas``). Building the pandas MultiIndex is by far
the most expensive step, so it is never paid per outage.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl
from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.branch_res_power_columns import (
    branch_res_power_columns,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
)
from toop_engine_grid_helpers.pandapower.outage_group import ConnectivityGraphCache
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id_from_index
from toop_engine_interfaces.loadflow_results import BranchSide
from toop_engine_interfaces.nminus1_definition import SwitchMonitoringScope

BRANCH_TYPES = ("line", "trafo", "trafo3w", "impedance")

#: ``res_*`` tables snapshotted to polars after each converged power flow.
RES_TABLES_FOR_POLARS = ("res_line", "res_trafo", "res_trafo3w", "res_impedance", "res_bus")

MAX_AMOUNT_OF_SIDES = 3

#: Voltage deviation treated as fully loaded when scaling ``vm_loading``.
MAX_ALLOWED_VM_DEVIATION = 0.2


def cache_res_tables_as_polars(net: pandapowerNet) -> None:
    """Snapshot the ``res_*`` tables of *net* as polars frames under ``res_*_polars``.

    Called right after a converged power flow so the result extractors can read polars
    directly instead of converting the pandas tables on every access. The pandas tables
    are left untouched: pandapower and the SpPS engine keep using them.
    """
    for table in RES_TABLES_FOR_POLARS:
        df = net[table]
        # reset_index(drop=True) keeps row order, which is what the precomputed element
        # id arrays are aligned to; polars has no index of its own.
        net[f"{table}_polars"] = pl.from_pandas(df.reset_index(drop=True))


def _rated_current(net_table: pd.DataFrame, sn_col: str, vn_col: str, i_limit_col: str) -> np.ndarray:
    """Maximum of the current implied by rated power/voltage and the CurrentLimit value."""
    if i_limit_col in net_table.columns:
        i_limit = net_table[i_limit_col].fillna(0).to_numpy() / 1000  # already in A
    else:
        i_limit = np.zeros(len(net_table))

    i_rated = net_table[sn_col].to_numpy() / (np.sqrt(3) * net_table[vn_col].to_numpy())  # convert kA to A
    return np.maximum(i_rated, i_limit)


@dataclass(frozen=True)
class ResultConstants:
    """Per-job constants for branch, node and switch result extraction.

    Everything here depends on the static network (and the base-case load flow) only, so it
    is computed once - via :meth:`from_network` - and reused for every contingency. The
    class is frozen: these are the outage-invariant inputs, and rebuilding them per outage
    is exactly the cost it exists to avoid.
    """

    #: Switch-to-element mapping as polars. The mapping runs to hundreds of thousands of
    #: rows, so it is converted once here rather than per contingency.
    switch_element_mapping_pl: pl.DataFrame
    #: Globally unique ids of every monitored element, used to filter branch/node results.
    monitored_element_ids: pl.Series
    #: Monitored-element display names, keyed by globally unique id.
    element_name_map: dict

    #: Pandapower switch indices of ANGLE-scoped switches (va-diff is keyed by table id).
    angle_switch_table_ids: np.ndarray
    #: Globally unique ids of the same switches (the failed path reports them by element id).
    angle_switch_element_ids: list
    #: Names of monitored switches, keyed by globally unique id.
    switch_name_map: dict

    # Element ids used to build the NaN result rows when a load flow does not converge.
    monitored_trafo3w_ids: list
    monitored_branch_ids: list
    monitored_bus_ids: list

    #: Globally unique element ids per branch type and bus, aligned to the res_* row order.
    element_uids: dict
    #: Rated currents per (branch type, side); used to turn currents into loadings.
    i_max: dict

    #: Bus voltage levels in kV, and the base-case per-unit voltages for the per-bus deviation.
    voltage_levels: np.ndarray
    basecase_vm: np.ndarray

    #: Shared across outages: the va-diff connectivity graph is rebuilt only when the bus-bus
    #: switch states actually change. Mutated internally across outages - the frozen field
    #: holds a stable reference to a cache that memoizes; it is never reassigned.
    graph_cache: ConnectivityGraphCache = field(default_factory=ConnectivityGraphCache)

    @classmethod
    def from_network(
        cls,
        net: pandapowerNet,
        basecase_net: pandapowerNet,
        monitored_elements: pd.DataFrame,
        switch_element_mapping: pd.DataFrame,
    ) -> "ResultConstants":
        """Compute the per-job constants from the static network and monitored elements."""
        # Projections of the monitored-element table. Each was previously recomputed on every
        # outage - the scope filter runs a Python callable over every monitored element, so
        # doing it once matters on large nets.
        angle_scope = monitored_elements["monitoring_scope"].apply(
            lambda scope: scope is not None and SwitchMonitoringScope.ANGLE in scope
        )

        element_uids = {
            table: get_globally_unique_id_from_index(net[table].index, element_type=table).to_numpy(dtype=object)
            for table in (*BRANCH_TYPES, "bus")
        }

        voltage_levels = net.bus["vn_kv"].to_numpy()

        # Base-case voltages for the per-bus deviation; zeros would divide by zero.
        basecase_vm = basecase_net.res_bus["vm_pu"].to_numpy(dtype=np.float64)
        if len(basecase_vm) != len(voltage_levels):
            # No base-case load flow was run (or it covered a different bus set), so there
            # is nothing to compare against and the deviation is undefined. pandas used to
            # produce NaN here by index alignment; polars needs the length made explicit.
            basecase_vm = np.full(len(voltage_levels), np.nan)
        basecase_vm = np.where(basecase_vm == 0, np.nan, basecase_vm)

        return cls(
            switch_element_mapping_pl=pl.from_pandas(switch_element_mapping),
            monitored_element_ids=pl.Series("element", monitored_elements.index.to_numpy(), dtype=pl.String),
            element_name_map=monitored_elements["name"].to_dict(),
            angle_switch_table_ids=monitored_elements.loc[angle_scope, "table_id"].to_numpy(),
            angle_switch_element_ids=monitored_elements.index[angle_scope].to_list(),
            switch_name_map=monitored_elements.query("kind == 'switch'")["name"].to_dict(),
            monitored_trafo3w_ids=monitored_elements.query("table == 'trafo3w'").index.to_list(),
            monitored_branch_ids=monitored_elements.query("kind == 'branch' & table != 'trafo3w'").index.to_list(),
            monitored_bus_ids=monitored_elements.query("kind == 'bus'").index.to_list(),
            element_uids=element_uids,
            i_max={
                ("line", 0): net.line["max_i_ka"].to_numpy(),
                ("line", 1): net.line["max_i_ka"].to_numpy(),
                ("trafo", 0): _rated_current(net.trafo, "sn_mva", "vn_hv_kv", "CurrentLimit.value_hv"),
                ("trafo", 1): _rated_current(net.trafo, "sn_mva", "vn_lv_kv", "CurrentLimit.value_lv"),
                ("trafo3w", 0): _rated_current(net.trafo3w, "sn_hv_mva", "vn_hv_kv", "CurrentLimit.value_hv"),
                ("trafo3w", 1): _rated_current(net.trafo3w, "sn_mv_mva", "vn_mv_kv", "CurrentLimit.value_mv"),
                ("trafo3w", 2): _rated_current(net.trafo3w, "sn_lv_mva", "vn_lv_kv", "CurrentLimit.value_lv"),
            },
            voltage_levels=voltage_levels,
            basecase_vm=basecase_vm,
        )


def get_branch_results_polars(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
    constants: ResultConstants,
) -> pl.DataFrame:
    """Branch results for one contingency as a polars frame.

    One row per branch terminal, with ``timestep``/``contingency``/``element``/``side`` as
    columns. ``p`` and ``q`` are blanked wherever ``i`` is null, matching the convention
    that an unsupplied terminal reports no flow.
    """
    frames = []
    for branch_type in BRANCH_TYPES:
        res_table = net[f"res_{branch_type}_polars"]
        if res_table.is_empty():
            continue

        uids = constants.element_uids[branch_type]
        for side in range(MAX_AMOUNT_OF_SIDES):
            try:
                columns = branch_res_power_columns(branch_type, side=side)
            except IndexError:
                break  # this branch type has no further sides

            present = [column for column in columns if column in res_table.columns]
            if not present:
                continue

            frame = res_table.select(present).rename(
                dict(zip(present, ["p", "q", "i", "loading"], strict=False)),
            )
            frame = frame.with_columns(
                pl.lit(timestep, dtype=pl.Int64).alias("timestep"),
                pl.lit(contingency.unique_id).alias("contingency"),
                pl.Series("element", uids, dtype=pl.String),
                pl.lit(side + 1, dtype=pl.Int64).alias("side"),
            )

            if "i" in frame.columns:
                # pandapower reports kA; the schema wants A.
                frame = frame.with_columns((pl.col("i") * 1000).alias("i"))
                i_max = constants.i_max.get((branch_type, side))
                if i_max is not None:
                    frame = frame.with_columns((pl.col("i") / pl.Series("i_max", i_max * 1000)).alias("loading"))
                # A terminal with no current carries no meaningful power either.
                frame = frame.with_columns(
                    pl.when(pl.col("i").is_null()).then(None).otherwise(pl.col("p")).alias("p"),
                    pl.when(pl.col("i").is_null()).then(None).otherwise(pl.col("q")).alias("q"),
                )

            frames.append(frame)

    branch_results = pl.concat(frames, how="diagonal")
    return branch_results.with_columns(
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    )


def get_node_results_polars(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
    constants: ResultConstants,
) -> pl.DataFrame:
    """Node (bus) results for one contingency as a polars frame."""
    res_bus = net["res_bus_polars"]
    basecase_vm = pl.Series("bc_vm", constants.basecase_vm)

    node_results = res_bus.rename({"vm_pu": "vm", "va_degree": "va", "p_mw": "p", "q_mvar": "q"})
    return node_results.with_columns(
        pl.lit(timestep, dtype=pl.Int64).alias("timestep"),
        pl.lit(contingency.unique_id).alias("contingency"),
        pl.Series("element", constants.element_uids["bus"], dtype=pl.String),
        # Deviation from the base-case voltage, in percent. Still per-unit at this point.
        (((pl.col("vm") - basecase_vm).abs() / basecase_vm) * 100).alias("vm_basecase_deviation"),
        ((pl.col("vm") - 1) / MAX_ALLOWED_VM_DEVIATION).alias("vm_loading"),
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    ).with_columns(
        # Scale to kV only after the per-unit quantities above have been derived.
        (pl.col("vm") * pl.Series("vn_kv", constants.voltage_levels)).alias("vm"),
    )


def _failed_key_frame(
    timestep: int,
    contingencies: list[str],
    elements: list[str],
    sides: list[int] | None,
) -> pl.DataFrame:
    """Cross product of ``contingencies`` x ``elements`` (x ``sides``) as key columns.

    Returns a frame with ``timestep``/``contingency``/``element`` (and ``side`` when *sides*
    is given). An empty input on any axis yields a 0-row frame with the columns and dtypes
    preserved, so the caller can build result rows on it unconditionally.
    """
    frame = pl.DataFrame({"contingency": pl.Series(contingencies, dtype=pl.String)}).join(
        pl.DataFrame({"element": pl.Series(elements, dtype=pl.String)}),
        how="cross",
    )
    if sides is not None:
        frame = frame.join(pl.DataFrame({"side": pl.Series(sides, dtype=pl.Int64)}), how="cross")
    return frame.with_columns(pl.lit(timestep, dtype=pl.Int64).alias("timestep"))


def get_failed_branch_results_polars(
    timestep: int,
    failed_outages: list[str],
    monitored_2_end_branches: list[str],
    monitored_3_end_branches: list[str],
) -> pl.DataFrame:
    """All-null branch results for outages whose load flow did not converge.

    Native-polars counterpart of :func:`get_branch_results_polars`: one row per monitored
    branch terminal (two sides for lines/2W trafos, three for 3W trafos), with the electrical
    columns null. Same flat layout, so the two concatenate directly.
    """
    two_end = _failed_key_frame(
        timestep, failed_outages, monitored_2_end_branches, [BranchSide.ONE.value, BranchSide.TWO.value]
    )
    three_end = _failed_key_frame(
        timestep,
        failed_outages,
        monitored_3_end_branches,
        [BranchSide.ONE.value, BranchSide.TWO.value, BranchSide.THREE.value],
    )
    return pl.concat([two_end, three_end], how="vertical").select(
        "timestep",
        "contingency",
        "element",
        "side",
        pl.lit(None, dtype=pl.Float64).alias("p"),
        pl.lit(None, dtype=pl.Float64).alias("q"),
        pl.lit(None, dtype=pl.Float64).alias("i"),
        pl.lit(None, dtype=pl.Float64).alias("loading"),
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    )


def get_failed_node_results_polars(
    timestep: int,
    failed_outages: list[str],
    monitored_nodes: list[str],
) -> pl.DataFrame:
    """All-null node results for outages whose load flow did not converge.

    Native-polars counterpart of :func:`get_node_results_polars`: one row per monitored bus
    with the electrical columns null. Same flat layout, so the two concatenate directly.
    """
    return _failed_key_frame(timestep, failed_outages, monitored_nodes, sides=None).select(
        "timestep",
        "contingency",
        "element",
        pl.lit(None, dtype=pl.Float64).alias("vm"),
        pl.lit(None, dtype=pl.Float64).alias("va"),
        pl.lit(None, dtype=pl.Float64).alias("p"),
        pl.lit(None, dtype=pl.Float64).alias("q"),
        pl.lit(None, dtype=pl.Float64).alias("vm_basecase_deviation"),
        pl.lit(None, dtype=pl.Float64).alias("vm_loading"),
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    )


def filter_to_monitored(results: pl.DataFrame, monitored_element_ids: pl.Series) -> pl.DataFrame:
    """Keep only rows whose ``element`` is monitored."""
    return results.filter(pl.col("element").is_in(monitored_element_ids))
