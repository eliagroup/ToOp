# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Names of pandapower ``res_*`` active/reactive power columns per branch side.

This replaces ``pandapower.toolbox.res_power_columns`` for branch extraction only:
explicit tables aligned with pandapower result tables (see ``results_branch.py``).

Side indices match pandapower's numeric convention (0-based): line/impedance use
from/to; two-winding transformers hv/lv; three-winding hv/mv/lv.
"""

from __future__ import annotations

from typing import Final

# (active_power_column, reactive_power_column) per side, in side order.
_BRANCH_RES_POWER_COLUMNS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "line": (
        ("p_from_mw", "q_from_mvar", "i_from_ka", "loading_percent_from"),
        ("p_to_mw", "q_to_mvar", "i_to_ka", "loading_percent_to"),
    ),
    "impedance": (
        ("p_from_mw", "q_from_mvar", "i_from_ka"),
        ("p_to_mw", "q_to_mvar", "i_to_ka"),
    ),
    "trafo": (
        ("p_hv_mw", "q_hv_mvar", "i_hv_ka", "loading_percent_hv"),
        ("p_lv_mw", "q_lv_mvar", "i_lv_ka", "loading_percent_lv"),
    ),
    "trafo3w": (
        ("p_hv_mw", "q_hv_mvar", "i_hv_ka", "loading_percent_hv"),
        ("p_mv_mw", "q_mv_mvar", "i_mv_ka", "loading_percent_mv"),
        ("p_lv_mw", "q_lv_mvar", "i_lv_ka", "loading_percent_lv"),
    ),
}


def branch_res_power_columns(branch_type: str, *, side: int) -> list[str]:
    """Return ``res_<branch_type>`` column names for P and Q at the given branch side.

    Parameters
    ----------
    branch_type
        One of ``line``, ``impedance``, ``trafo``, ``trafo3w``.
    side
        Zero-based side index (from/hv → 0, etc.).

    Returns
    -------
    list[str]
        Two entries: active power column, then reactive power column.

    Raises
    ------
    KeyError
        If ``branch_type`` is unknown or ``side`` is out of range for that type.
    """
    return _BRANCH_RES_POWER_COLUMNS[branch_type][side]
