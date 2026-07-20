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
plain columns. Building the pandas MultiIndex is by far the most expensive step in these
functions, so it is deferred to the pipeline boundary (see :func:`to_pandas_results`)
and paid once per outage on the filtered frames instead of nine times on the full ones.
"""

from typing import Optional

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

BRANCH_TYPES = ("line", "trafo", "trafo3w", "impedance")

#: ``res_*`` tables snapshotted to polars after each converged power flow.
RES_TABLES_FOR_POLARS = ("res_line", "res_trafo", "res_trafo3w", "res_impedance", "res_bus")

MAX_AMOUNT_OF_SIDES = 3

#: Voltage deviation treated as fully loaded when scaling ``vm_loading``.
MAX_ALLOWED_VM_DEVIATION = 0.2

BRANCH_RESULT_INDEX = ["timestep", "contingency", "element", "side"]
NODE_RESULT_INDEX = ["timestep", "contingency", "element"]


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


class ResultConstants:
    """Per-job constants for branch and node result extraction.

    Everything here depends on the static network (and the base-case load flow) only, so it
    is computed once and reused for every contingency.
    """

    def __init__(
        self,
        net: pandapowerNet,
        basecase_net: pandapowerNet,
        monitored_elements: Optional[pd.DataFrame] = None,
        switch_element_mapping: Optional[pd.DataFrame] = None,
    ) -> None:
        # The switch mapping and the monitored-element table are identical for every outage
        # in a run, but converting them is not free: the mapping runs to hundreds of
        # thousands of rows. Convert once here instead of per contingency.
        self.switch_element_mapping_pl = (
            pl.from_pandas(switch_element_mapping) if switch_element_mapping is not None else None
        )
        if monitored_elements is not None:
            self.monitored_element_ids = pl.Series("element", monitored_elements.index.to_numpy(), dtype=pl.String)
            self.element_name_map = monitored_elements["name"].to_dict()
        else:
            self.monitored_element_ids = None
            self.element_name_map = None

        self.element_uids = {
            table: get_globally_unique_id_from_index(net[table].index, element_type=table).to_numpy(dtype=object)
            for table in (*BRANCH_TYPES, "bus")
        }

        # Rated currents per branch side; used to turn currents into loadings.
        self.i_max = {
            ("line", 0): net.line["max_i_ka"].to_numpy(),
            ("line", 1): net.line["max_i_ka"].to_numpy(),
            ("trafo", 0): _rated_current(net.trafo, "sn_mva", "vn_hv_kv", "CurrentLimit.value_hv"),
            ("trafo", 1): _rated_current(net.trafo, "sn_mva", "vn_lv_kv", "CurrentLimit.value_lv"),
            ("trafo3w", 0): _rated_current(net.trafo3w, "sn_hv_mva", "vn_hv_kv", "CurrentLimit.value_hv"),
            ("trafo3w", 1): _rated_current(net.trafo3w, "sn_mv_mva", "vn_mv_kv", "CurrentLimit.value_mv"),
            ("trafo3w", 2): _rated_current(net.trafo3w, "sn_lv_mva", "vn_lv_kv", "CurrentLimit.value_lv"),
        }

        # Shared across outages: the va-diff connectivity graph is rebuilt only if the
        # bus-bus switch states actually change.
        self.graph_cache = ConnectivityGraphCache()

        self.voltage_levels = net.bus["vn_kv"].to_numpy()

        # Base-case voltages for the per-bus deviation; zeros would divide by zero.
        basecase_vm = basecase_net.res_bus["vm_pu"].to_numpy(dtype=np.float64)
        if len(basecase_vm) != len(self.voltage_levels):
            # No base-case load flow was run (or it covered a different bus set), so there
            # is nothing to compare against and the deviation is undefined. pandas used to
            # produce NaN here by index alignment; polars needs the length made explicit.
            basecase_vm = np.full(len(self.voltage_levels), np.nan)
        self.basecase_vm = np.where(basecase_vm == 0, np.nan, basecase_vm)


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


def filter_to_monitored(results: pl.DataFrame, monitored_element_ids: pl.Series) -> pl.DataFrame:
    """Keep only rows whose ``element`` is monitored."""
    return results.filter(pl.col("element").is_in(monitored_element_ids))


def to_pandas_results(results: pl.DataFrame, index_columns: list[str]) -> pd.DataFrame:
    """Convert a polars result frame to the indexed pandas frame the schemas expect.

    This is where the MultiIndex is finally built - once per outage, on the already
    filtered frame.
    """
    return results.to_pandas().set_index(index_columns)
