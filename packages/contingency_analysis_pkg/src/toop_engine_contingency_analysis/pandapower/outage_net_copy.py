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

import logging
from copy import deepcopy

import numpy as np
import pandapower as pp
import pandas as pd

_logger = logging.getLogger(__name__)

#: Columns an outage may write, per table. Listed tables are copied column-wise - these columns
#: get data of their own, the rest share the source net's memory. Unlisted tables are deep-copied.
#:
#: This is also the specification the write barrier enforces: with
#: ``ContingencyAnalysisConfig.freeze_net_columns`` on, :func:`freeze_net_columns` makes every
#: column outside this map read-only, so writing one raises instead of landing silently.
MUTABLE_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    # Outaging an element clears its in_service flag.
    "bus": ("in_service",),
    "line": ("in_service",),
    "impedance": ("in_service",),
    # pandapower resolves tap-dependent impedance by overwriting these on every solve
    # (build_branch._get_vk_values_from_table), for two- and three-winding trafos alike.
    "trafo": ("in_service", "vk_percent", "vkr_percent", "tap_pos"),
    "trafo3w": (
        "in_service",
        "tap_pos",
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

#: Tables whose *rows* an outage adds or drops, so they must be deep-copied whole.
#:
#: ``slack_allocation.replace_sgen_by_gen`` promotes an ``sgen`` to a ``gen`` when picking an
#: island's slack, and runs on every outage copy. A column-wise copy shares the index, so no
#: entry in :data:`MUTABLE_COLUMNS_BY_TABLE` can make these tables safe - hence the check below.
#: The write barrier cannot help either: inserting a row reallocates instead of writing in
#: place, so nothing raises.
ROW_MUTATING_TABLES: frozenset[str] = frozenset({"gen", "sgen"})


def _copy_table_columns(frame: pd.DataFrame, mutable_columns: tuple[str, ...]) -> pd.DataFrame:
    """Copy *frame*, giving only *mutable_columns* data of their own.

    ``.copy(deep=False)`` shares the column data; reassigning a column from a deep copy then
    detaches it, so writes to that column stay local.

    Parameters
    ----------
    frame : pd.DataFrame
        Table to copy.
    mutable_columns : tuple[str, ...]
        Columns to detach. Names absent from *frame* are ignored.

    Returns
    -------
    pd.DataFrame
        Copy whose *mutable_columns* own their data and whose other columns share *frame*'s.
    """
    copied = frame.copy(deep=False)
    for column in mutable_columns:
        if column in copied.columns:
            copied[column] = frame[column].copy(deep=True)
    return copied


def _freeze_values(values: object) -> bool:
    """Make one block's storage read-only.

    A numpy block is frozen through its ``flags``; an ``ExtensionArray`` has none, so the numpy
    arrays it wraps are frozen instead. Arrow-backed arrays expose neither and cannot be frozen.

    Parameters
    ----------
    values : object
        Block backing to freeze, either a numpy array or an ``ExtensionArray``.

    Returns
    -------
    bool
        True when the storage is now read-only, False when it cannot be frozen at all.
    """
    try:
        values.flags.writeable = False
        return True
    except (AttributeError, ValueError):
        pass

    # Clearing the flag always succeeds - only setting it back to True can fail - so unlike the
    # attempt above this one needs no guard.
    frozen = False
    for attr in ("_data", "_ndarray", "_mask"):
        inner = getattr(values, attr, None)
        if isinstance(inner, np.ndarray):
            inner.flags.writeable = False
            frozen = True
    return frozen


def _freeze_table_columns(frame: pd.DataFrame, mutable_columns: tuple[str, ...]) -> tuple[int, list[str], list[str]]:
    """Freeze every column of one table that is not declared mutable.

    Parameters
    ----------
    frame : pd.DataFrame
        Table to freeze.
    mutable_columns : tuple[str, ...]
        Columns an outage is allowed to write.

    Returns
    -------
    tuple[int, list[str], list[str]]
        Blocks frozen, columns that could not be frozen, and columns left writable because they
        share a block with a mutable column.
    """
    mutable = set(mutable_columns)

    frozen = 0
    unfreezable: list[str] = []
    skipped: list[str] = []
    for block in frame._mgr.blocks:
        columns = [str(frame.columns[loc]) for loc in block.mgr_locs]
        if not columns or all(column in mutable for column in columns):
            continue
        if any(column in mutable for column in columns):
            # Freezing would also block the legitimate write to the mutable column in this
            # block. ``_copy_table_columns`` normally splits those out, so this is rare.
            skipped.extend(column for column in columns if column not in mutable)
            continue
        if _freeze_values(block.values):
            frozen += 1
        else:
            unfreezable.extend(columns)
    return frozen, unfreezable, skipped


def freeze_net_columns(net: pp.pandapowerNet) -> int:
    """Freeze every column of the whole net that an outage may not write, in place.

    Call once per process, after the base-case load flow and before the outage loop. Each outage
    copy then inherits the barrier for free, because it shares those columns; the mutable ones
    are reassigned from a deep copy and stay writable.

    Freezing a deep-copied table (``gen``, ``sgen``, ``res_*``) protects only this net, since the
    copy gets writable storage of its own. That is also why freezing them is harmless.

    Read-only-ness does not survive pickling.

    Parameters
    ----------
    net : pp.pandapowerNet
        Network the outage copies are made from. Mutated in place.

    Returns
    -------
    int
        Number of blocks frozen.

    Raises
    ------
    RuntimeError
        If a shared column cannot be made read-only, or if nothing could be frozen at all - a
        barrier covering nothing would report success for a net it never checked.
    """
    frozen = 0
    unguarded: list[str] = []
    uncovered: list[str] = []
    for key, value in net.items():
        if not isinstance(key, str) or not isinstance(value, pd.DataFrame) or value.empty:
            continue
        if key in ROW_MUTATING_TABLES:
            continue
        table_frozen, unfreezable, skipped = _freeze_table_columns(value, MUTABLE_COLUMNS_BY_TABLE.get(key, ()))
        frozen += table_frozen
        # Only a table the copies share can carry a write back into this net; an unfreezable
        # column on a deep-copied table is merely outside the barrier's reach.
        target = unguarded if key in MUTABLE_COLUMNS_BY_TABLE else uncovered
        target.extend(f"{key}.{column}" for column in unfreezable)
        uncovered.extend(f"{key}.{column}" for column in skipped)

    if unguarded:
        raise RuntimeError(
            "outage copies share these columns and they cannot be made read-only, so a write "
            f"into them would corrupt this net undetected: {sorted(unguarded)}. Add them to "
            "MUTABLE_COLUMNS_BY_TABLE so they get data of their own, or drop their table from "
            "the map so it is deep-copied."
        )
    if not frozen:
        raise RuntimeError(
            "the write barrier froze nothing, so it proves nothing. The pandas block layout it "
            "relies on has probably changed; fix _freeze_table_columns rather than trusting "
            "a green run."
        )
    if uncovered:
        _logger.debug("write barrier: %s columns left writable: %s", len(uncovered), sorted(uncovered))
    _logger.debug("write barrier: froze %s blocks", frozen)
    return frozen


def copy_net_for_outage(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Copy *net* so one outage can run on it without touching the original.

    Tables in :data:`MUTABLE_COLUMNS_BY_TABLE` are copied column-wise; everything else is
    deep-copied.

    Parameters
    ----------
    net : pp.pandapowerNet
        Network to copy. Not modified.

    Returns
    -------
    pp.pandapowerNet
        Copy for one outage to run on. Columns shared with *net* keep the read-only flag
        :func:`freeze_net_columns` gave them.
    """
    copied = pp.pandapowerNet({})
    for key, value in net.items():
        if isinstance(value, pd.DataFrame) and isinstance(key, str):
            if key in ROW_MUTATING_TABLES:
                # Deep-copied on purpose, not for want of classification - see the constant.
                copied[key] = deepcopy(value)
                continue
            if key in MUTABLE_COLUMNS_BY_TABLE:
                copied[key] = _copy_table_columns(value, MUTABLE_COLUMNS_BY_TABLE[key])
                continue
            if key.startswith("_empty_res_"):
                copied[key] = value.copy(deep=False)
                continue
        copied[key] = deepcopy(value)
    return copied
