# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Per-job constants and shared helpers for polars result extraction.

Result tables are rebuilt for every contingency, but most of what goes into them never
changes: element ids, rated currents, bus voltage levels and the base-case voltages all
depend on the static network only. :class:`ResultConstants` computes those once per job (via
:meth:`ResultConstants.from_network`); the per-outage extractors in ``branch_results`` /
``node_results`` / ``va_diff_results`` / ``switch_results`` then only touch the numbers the
load flow actually changed.

This module also holds the ``res_*`` polars snapshot (:func:`cache_res_tables_as_polars`)
that those extractors read from.

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
from toop_engine_grid_helpers.pandapower.outage_group import ConnectivityGraphCache
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id_from_index
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
