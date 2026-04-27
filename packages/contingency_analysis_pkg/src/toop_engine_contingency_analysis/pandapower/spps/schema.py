"""Data contracts for the SpPS engine.

This module declares the two stable *shapes* that travel in and out of
:func:`spps.engine.run_spps`:

* ``SppsConditionsPandapowerSchema`` and ``SppsActionsPandapowerSchema`` (Pandera) — input tables.
* :class:`SppsResult` — the typed return bundle (Pydantic).

It also holds the static lookup tables that describe which pandapower
columns to read (:data:`RESULT_COLUMNS`) and write (:data:`ACTION_COLUMNS`)
for each element type, plus the compound-uid helpers used to reference
failed elements across tables.
"""

from typing import Final

import pandapower as pp
from pydantic import BaseModel, ConfigDict

ELEMENT_TABLES: Final[tuple[str, ...]] = (
    "line",
    "trafo",
    "trafo3w",
    "impedance",
    "bus",
    "switch",
    "gen",
    "load",
    "sgen",
    "shunt",
    "ward",
    "xward",
)

RESULT_COLUMNS: Final[dict[str, dict[str, dict[str, str]]]] = {
    "trafo3w": {
        "Current": {
            "Primary": "i_hv_ka",
            "Secondary": "i_mv_ka",
            "Tertiary": "i_lv_ka",
        },
        "Active power": {
            "Primary": "p_hv_mw",
            "Secondary": "p_mv_mw",
            "Tertiary": "p_lv_mw",
        },
        "Reactive power": {
            "Primary": "q_hv_mvar",
            "Secondary": "q_mv_mvar",
            "Tertiary": "q_lv_mvar",
        },
    },
    "trafo": {
        "Current": {"Primary": "i_hv_ka", "Secondary": "i_lv_ka"},
        "Active power": {"Primary": "p_hv_mw", "Secondary": "p_lv_mw"},
        "Reactive power": {"Primary": "q_hv_mvar", "Secondary": "q_lv_mvar"},
    },
    "line": {
        "Current": {"Primary": "i_from_ka", "Secondary": "i_to_ka"},
        "Active power": {"Primary": "p_from_mw", "Secondary": "p_to_mw"},
        "Reactive power": {"Primary": "q_from_mvar", "Secondary": "q_to_mvar"},
    },
}

ACTION_COLUMNS: Final[dict[str, dict[str, str]]] = {
    "gen": {"Active power": "p_mw", "Voltage": "vm_pu"},
    "sgen": {"Active power": "p_mw", "Reactive power": "q_mvar"},
    "load": {"Active power": "p_mw", "Reactive power": "q_mvar"},
    "shunt": {"Active power": "p_mw", "Reactive power": "q_mvar"},
    "ward": {"Active power": "ps_mw", "Reactive power": "qs_mvar"},
    "xward": {"Active power": "ps_mw", "Reactive power": "qs_mvar"},
}


class SppsResult(BaseModel):
    """Typed result bundle returned by :func:`spps.engine.run_spps`.

    ``arbitrary_types_allowed=True`` is set so Pydantic can wrap the
    (non-Pydantic) :class:`pandapower.pandapowerNet` instance without
    trying to validate its internals.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    net: pp.pandapowerNet
    """The pandapower network after all SpPS actions have been applied.

    Mutated *in place* — this is the same object the caller passed in.
    """

    iterations: int
    """Number of iterations actually executed (1-based count).

    Equal to ``len(activated_schemes_per_iter)``.
    """

    activated_schemes_per_iter: list[list[str]]
    """Per-iteration list of scheme names that fired (order preserved).

    ``activated_schemes_per_iter[i]`` is the sorted list of unique scheme
    names activated on iteration ``i + 1``. The last entry is empty iff the
    loop exited early because no more schemes triggered.
    """

    max_iterations_reached: bool
    """``True`` when the engine ran out of iteration budget.

    Specifically: the loop reached ``max_iterations`` while the final
    iteration was still producing new scheme activations. ``False`` means the
    loop exited cleanly (either no scheme triggered, or a PF failure stopped
    it). This is **not** a convergence indicator — see
    :attr:`power_flow_failed` for that.
    """

    power_flow_failed: bool = False
    """``True`` if an iteration-level power flow failed under
    ``on_power_flow_error="keep_previous"``.

    When ``True``, the engine restored the ``res_*`` tables on :attr:`net`
    from the snapshot taken *before* the failing solver call, so the result
    tables reflect the last successful iteration. Setpoint changes applied
    during the failing iteration (switch openings, gen/load adjustments) are
    left in place.
    """
