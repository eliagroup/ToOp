# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Per-outage copy of a pandapower net.

Every contingency runs on its own copy, so this is paid once per outage. Most of a ``deepcopy``
goes into name, mrid and description columns that no outage ever writes, so we copy only the
writable columns and share the rest - about 5x faster.
"""

from copy import deepcopy

import pandapower as pp
import pandas as pd

#: Columns an outage can write, per table. A listed table is copied column-wise: these columns
#: get data of their own, all others share the source net's memory. Unlisted tables are
#: deep-copied whole, which is what ``gen``/``sgen`` (rows get added and dropped), the ``res_*``
#: tables, the ``_ppc`` internals and any newly added table need.
#:
#: Writing a *shared* column would corrupt the source net and every later outage. After changing
#: this map, run ``check_shared_tables_readonly.py``: it makes such a write raise instead.
MUTABLE_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    # Outaging an element clears its in_service flag.
    "bus": ("in_service",),
    "line": ("in_service",),
    "impedance": ("in_service",),
    # pandapower resolves tap-dependent impedance by overwriting these on every solve
    # (build_branch._get_vk_values_from_table), for two- and three-winding trafos alike.
    "trafo": ("in_service", "vk_percent", "vkr_percent"),
    "trafo3w": (
        "in_service",
        "vk_hv_percent",
        "vkr_hv_percent",
        "vk_mv_percent",
        "vkr_mv_percent",
        "vk_lv_percent",
        "vkr_lv_percent",
    ),
    # Opened by an outage, by busbar isolation, by an SpPS action or by a cascade trip.
    # Switches have no in_service column - closing them is how they go out of service.
    "switch": ("closed",),
    # SpPS setpoint actions, plus in_service when the element itself is outaged.
    "load": ("in_service", "p_mw", "q_mvar"),
    "shunt": ("in_service", "p_mw", "q_mvar"),
    "ward": ("in_service", "ps_mw", "qs_mvar"),
    "xward": ("in_service", "ps_mw", "qs_mvar"),
    # Reference data from the CGMES import: never written on the outage path.
    "measurement": (),
    "trafo_characteristic_table": (),
    "shunt_characteristic_table": (),
    "q_capability_characteristic": (),
    "q_capability_curve_table": (),
}


def _copy_table_columns(frame: pd.DataFrame, mutable_columns: tuple[str, ...]) -> pd.DataFrame:
    """Copy *frame*, giving only *mutable_columns* data of their own.

    ``.copy(deep=False)`` shares the column data but gives the copy its own block manager.
    Reassigning a column from a real copy then detaches it, so writes to it stay local.
    """
    copied = frame.copy(deep=False)
    for column in mutable_columns:
        if column in copied.columns:
            copied[column] = frame[column].copy(deep=True)
    return copied


def copy_net_for_outage(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Copy *net* so one outage can run on it without touching the original.

    Tables in :data:`MUTABLE_COLUMNS_BY_TABLE` are copied column-wise; everything else is
    deep-copied.
    """
    copied = pp.pandapowerNet({})
    for key, value in net.items():
        if isinstance(value, pd.DataFrame) and isinstance(key, str):
            if key in MUTABLE_COLUMNS_BY_TABLE:
                copied[key] = _copy_table_columns(value, MUTABLE_COLUMNS_BY_TABLE[key])
                continue
            if key.startswith("_empty_res_"):
                copied[key] = value.copy(deep=False)
                continue
        copied[key] = deepcopy(value)
    return copied
