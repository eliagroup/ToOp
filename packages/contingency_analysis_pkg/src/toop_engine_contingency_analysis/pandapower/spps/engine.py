"""Core rule-engine algorithm: ``run_spps`` and its private helpers.

Mental model
------------
A *scheme* is a group of rows sharing the same ``scheme_name`` across the
*conditions* and *actions* DataFrames. A scheme is activated on an iteration
when **all** of its condition rows evaluate to ``True``; activation applies
**every** action row with that ``scheme_name`` to the network, and the loop
re-runs a power flow so the next iteration sees the new state.

Side effects
------------
The engine mutates ``net`` in place (setpoint changes, switch states,
``res_*`` tables). It also mutates the supplied ``conditions`` DataFrame by
adding auxiliary columns (``energized``, ``failed``, ``condition_element_value``,
``is_condition``) and mutates ``failed_elements`` in place as switches are
opened. Pass ``.copy()``'d inputs if immutability is required.

Concurrency
-----------
The engine is not reentrant: it relies on mutating ``net`` between iterations.
"""

from __future__ import annotations

import logging
from typing import Any, Final, Literal

import pandapower as pp
import pandas as pd
import pandera.typing as pat
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    SppsActionsPandapowerSchema,
    SppsConditionsPandapowerSchema,
)
from toop_engine_contingency_analysis.pandapower.spps.errors import SppsPowerFlowError
from toop_engine_contingency_analysis.pandapower.spps.schema import (
    ACTION_COLUMNS,
    ELEMENT_TABLES,
    RESULT_COLUMNS,
    SppsResult,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helpers — state population
# --------------------------------------------------------------------------- #


def _populate_energized(rules: pd.DataFrame, net: pp.pandapowerNet) -> None:
    """Write an ``energized`` boolean column onto *rules* from the network state.

    For each rule row, look up the condition element in its pandapower table
    and read a single "is it electrically alive?" flag:

    * switches → ``net.switch.closed``
    * everything else → ``net.<table>.in_service``

    The resulting ``energized`` column is used by both the ``"De-energized"``
    check (``energized == False``) and the NaN-based failure auto-detection
    inside :func:`_populate_failed` (only energized elements can legitimately
    be flagged as "failed in place").

    Rules whose ``condition_element_table`` does not match any registered
    table keep a ``pd.NA`` (no value). The column dtype is pandas'
    ``"boolean"`` (nullable).

    Mutates *rules* in place; does not touch *net*.
    """
    rules["energized"] = pd.Series(pd.NA, index=rules.index, dtype="boolean")
    for table in ELEMENT_TABLES:
        mask = rules["condition_element_table"] == table
        if not mask.any():
            continue
        ids = rules.loc[mask, "condition_element_table_id"]
        element_df = getattr(net, table)
        col = "closed" if table == "switch" else "in_service"
        rules.loc[mask, "energized"] = element_df.loc[ids, col].to_numpy()


_FAILURE_RESULT_TABLES: Final[tuple[str, ...]] = (
    "line",
    "trafo",
    "trafo3w",
    "impedance",
    "bus",
    "gen",
    "sgen",
    "load",
    "shunt",
    "ward",
    "xward",
)


def _populate_failed(
    rules: pd.DataFrame,
    failed_elements: set[str],
    net: pp.pandapowerNet,
) -> None:
    """Write a ``failed`` boolean column onto *rules* for ``"Failed"`` checks.

    Only rows whose ``condition_check_type == "Failed"`` are evaluated; every
    other row gets ``failed = False`` and is ignored by
    :func:`_evaluate_conditions`'s "Failed" branch.

    A condition element is flagged as failed when **either**:

    1. Its compound uid (``f"{id}{SEPARATOR}{table}"``) is present in
       *failed_elements*. This is the explicit/caller-supplied path, and also
       how switches opened earlier in the run are recycled as failures.
    2. It is energized (``in_service`` / ``closed`` == ``True``) yet its row
       in ``net.res_<table>`` contains at least one NaN — meaning pandapower
       couldn't produce a valid power-flow result for it. This auto-detection
       catches elements that became islanded after earlier actions.

    Parameters
    ----------
    rules
        Rules DataFrame. Must already have ``energized`` populated (see
        :func:`_populate_energized`). Mutated in place.
    failed_elements
        Set of compound uids (see :func:`spps.schema.make_element_uid`).
    net
        Pandapower network; only ``res_*`` tables are read.
    """
    rules["failed"] = False
    failed_type_mask = rules["condition_check_type"] == "Failed"
    if not failed_type_mask.any():
        return

    uids = rules["condition_element_table_id"].astype(str) + SEPARATOR + rules["condition_element_table"].astype(str)
    rules.loc[failed_type_mask, "failed"] = uids.loc[failed_type_mask].isin(failed_elements).to_numpy()

    energized = rules["energized"].fillna(False).astype(bool)

    for table in _FAILURE_RESULT_TABLES:
        mask = (rules["condition_element_table"] == table) & failed_type_mask
        if not mask.any():
            continue
        res = getattr(net, f"res_{table}", None)
        if res is None or res.empty:
            continue

        ids = rules.loc[mask, "condition_element_table_id"]
        has_nan = res.reindex(ids).isna().any(axis=1).to_numpy()
        auto = energized.loc[mask].to_numpy() & has_nan
        rules.loc[mask, "failed"] = rules.loc[mask, "failed"].to_numpy() | auto


# --------------------------------------------------------------------------- #
# Internal helpers — measurement extraction
# --------------------------------------------------------------------------- #


def _extract_res_values(
    rules: pd.DataFrame,
    net: pp.pandapowerNet,
    element_type: str,
    col_map: dict[str, dict[str, str]],
) -> None:
    """Extract ``res_<element_type>`` values into ``rules.condition_element_value``.

    For every combination of ``(condition_type, condition_side)`` supported
    by *col_map*, copy the corresponding absolute value from
    ``net.res_<element_type>`` into the rule's ``condition_element_value``.
    The special ``condition_side == "Maximum value"`` picks the largest
    absolute value across all sides (e.g. max of ``i_from_ka`` and ``i_to_ka``
    for a line) — handy for "overload at either end" rules.

    Absolute values are used so that sign conventions (direction of flow)
    don't flip a condition.

    Parameters
    ----------
    rules
        Rules DataFrame. Mutated in place.
    net
        Pandapower network (``res_<element_type>`` must exist and be filled).
    element_type
        Pandapower table name (``"line"``, ``"trafo"``, ``"trafo3w"``).
    col_map
        Mapping ``condition_type -> {condition_side -> res_column}``; one of
        the values of :data:`spps.schema.RESULT_COLUMNS`.
    """
    res_table = getattr(net, f"res_{element_type}")
    base_mask = rules["condition_element_table"] == element_type
    if not base_mask.any():
        return

    for cond_type, sides in col_map.items():
        type_mask = base_mask & (rules["condition_type"] == cond_type)
        if not type_mask.any():
            continue
        all_cols = list(dict.fromkeys(sides.values()))

        max_mask = type_mask & (rules["condition_side"] == "Maximum value")
        if max_mask.any():
            ids = rules.loc[max_mask, "condition_element_table_id"]
            vals = res_table.loc[ids, all_cols].abs()
            merged = vals.iloc[:, 0] if len(all_cols) == 1 else vals.max(axis=1)
            rules.loc[max_mask, "condition_element_value"] = merged.to_numpy()

        for side, col in sides.items():
            mask = type_mask & (rules["condition_side"] == side)
            if not mask.any():
                continue
            ids = rules.loc[mask, "condition_element_table_id"]
            rules.loc[mask, "condition_element_value"] = res_table.loc[ids, col].abs().to_numpy()


def _extract_bus_voltage(rules: pd.DataFrame, net: pp.pandapowerNet) -> None:
    """Copy bus voltage magnitudes (p.u.) into ``condition_element_value``.

    Applies only to rows where ``condition_element_table == "bus"`` and
    ``condition_type == "Voltage"``. The engine works entirely in p.u. for
    voltage — callers must convert ``condition_limit_value`` to p.u.
    up-front (see :func:`spps.preprocessing.convert_voltage_rules_to_pu`)
    so the comparison in :func:`_evaluate_conditions` is unit-consistent.

    Mutates *rules* in place.
    """
    mask = (rules["condition_element_table"] == "bus") & (rules["condition_type"] == "Voltage")
    if not mask.any():
        return
    ids = rules.loc[mask, "condition_element_table_id"]
    rules.loc[mask, "condition_element_value"] = net.res_bus.loc[ids, "vm_pu"].to_numpy()


def _extract_condition_values(rules: pd.DataFrame, net: pp.pandapowerNet) -> None:
    """Populate ``condition_element_value`` with the current power-flow result.

    Initialises the column to NaN, then fills it from ``net.res_*`` tables:

    * lines / trafos / trafo3w via :data:`spps.schema.RESULT_COLUMNS` and
      :func:`_extract_res_values`;
    * bus voltages via :func:`_extract_bus_voltage`.

    Rows whose ``(condition_element_table, condition_type)`` combination is
    not supported keep NaN, which then evaluates to ``False`` in every
    numeric check inside :func:`_evaluate_conditions`.

    Mutates *rules* in place.
    """
    rules["condition_element_value"] = pd.Series(float("nan"), index=rules.index, dtype="float64")
    for element_type, col_map in RESULT_COLUMNS.items():
        _extract_res_values(rules, net, element_type, col_map)
    _extract_bus_voltage(rules, net)


# --------------------------------------------------------------------------- #
# Internal helpers — condition evaluation & activation
# --------------------------------------------------------------------------- #


def _evaluate_conditions(rules: pd.DataFrame) -> None:
    """Evaluate every rule's check and write the result into ``is_condition``.

    Supported ``condition_check_type`` values and their semantics:

    * ``">"``  — ``condition_element_value > condition_limit_value``
    * ``"<"``  — ``condition_element_value < condition_limit_value``
    * ``"="``  — ``condition_element_value == condition_limit_value``
    * ``"Failed"``       — the ``failed`` column populated by
      :func:`_populate_failed`
    * ``"De-energized"`` — the ``energized`` column populated by
      :func:`_populate_energized` is explicitly ``False``

    Any NaN (e.g. unsupported element type, missing power-flow result)
    evaluates to ``False`` — missing data is never treated as a passing
    condition. Rows with an unknown check type silently keep
    ``is_condition = False``; consider adding stricter validation upstream.

    Mutates *rules* in place.
    """
    rules["is_condition"] = False

    check = rules["condition_check_type"]
    value = rules["condition_element_value"]
    limit = rules["condition_limit_value"]

    gt_mask = check == ">"
    rules.loc[gt_mask, "is_condition"] = (value[gt_mask] > limit[gt_mask]).fillna(False)

    lt_mask = check == "<"
    rules.loc[lt_mask, "is_condition"] = (value[lt_mask] < limit[lt_mask]).fillna(False)

    eq_mask = check == "="
    rules.loc[eq_mask, "is_condition"] = (value[eq_mask] == limit[eq_mask]).fillna(False)

    failed_mask = check == "Failed"
    rules.loc[failed_mask, "is_condition"] = rules.loc[failed_mask, "failed"].fillna(False)

    de_mask = check == "De-energized"
    rules.loc[de_mask, "is_condition"] = (rules.loc[de_mask, "energized"] == False).fillna(False)  # noqa: E712


def _satisfied_scheme_names(conditions: pd.DataFrame) -> set[str]:
    """Return the names of schemes for which every condition row passes.

    A scheme is satisfied when **all** of its condition rows have
    ``is_condition == True`` (one row per :class:`SppsConditionsPandapowerSchema`
    line).

    Parameters
    ----------
    conditions
        Conditions DataFrame with ``is_condition`` populated by
        :func:`_evaluate_conditions`.
    """
    satisfied = conditions["is_condition"].fillna(False).groupby(conditions["scheme_name"]).all()
    return set(satisfied[satisfied].index.tolist())


# --------------------------------------------------------------------------- #
# Internal helpers — action application
# --------------------------------------------------------------------------- #


def _apply_switch_actions(actions: pd.DataFrame, net: pp.pandapowerNet) -> None:
    """Apply every switch-targeted action by writing ``net.switch.closed``.

    ``measure_value`` for switch rows is the string ``"Open"`` or
    ``"Closed"``. Anything other than the literal string ``"Closed"`` is
    interpreted as *open* (``closed = False``). Consider validating the
    value upstream if you want strict error reporting on typos.

    Mutates *net* in place.
    """
    sw = actions[actions["measure_element_table"] == "switch"]
    if sw.empty:
        return
    ids = sw["measure_element_table_id"]
    closed = (sw["measure_value"] == "Closed").to_numpy()
    net.switch.loc[ids, "closed"] = closed
    logger.debug("Applied %d switch actions (ids=%s)", len(sw), ids.tolist())


def _apply_actions(actions: pd.DataFrame, net: pp.pandapowerNet) -> None:
    """Apply every action row in *actions* to *net*.

    Dispatches as follows:

    * Switch rows → :func:`_apply_switch_actions` (string ``"Open"`` /
      ``"Closed"``).
    * Everything else → looked up in :data:`spps.schema.ACTION_COLUMNS` by
      ``(measure_element_table, measure_type)`` and written as a numeric
      value to the corresponding pandapower column (``p_mw``, ``q_mvar``,
      ``vm_pu``, ...).

    Voltage ``measure_value`` must already be in per-unit — call
    :func:`spps.preprocessing.convert_voltage_rules_to_pu` during
    preprocessing. Rows whose ``(table, measure_type)`` pair isn't in
    :data:`spps.schema.ACTION_COLUMNS` are silently skipped (no-op).

    Mutates *net* in place.
    """
    if actions.empty:
        return
    _apply_switch_actions(actions, net)

    for table, type_to_col in ACTION_COLUMNS.items():
        df = actions[actions["measure_element_table"] == table]
        if df.empty:
            continue
        target = getattr(net, table)
        for measure_type, col in type_to_col.items():
            sub = df[df["measure_type"] == measure_type]
            if sub.empty:
                continue
            ids = sub["measure_element_table_id"]
            target.loc[ids, col] = pd.to_numeric(sub["measure_value"], errors="coerce").to_numpy()
            logger.debug(
                "Applied %d %s '%s' updates to net.%s.%s",
                len(sub),
                table,
                measure_type,
                table,
                col,
            )


# --------------------------------------------------------------------------- #
# Power-flow helpers
# --------------------------------------------------------------------------- #


def _snapshot_res_tables(net: pp.pandapowerNet) -> dict[str, pd.DataFrame]:
    """Return a name→copy mapping of every ``res_*`` DataFrame on *net*.

    Each value is a full (deep) pandas ``DataFrame.copy()`` so subsequent
    solver calls cannot mutate it. Non-DataFrame entries under the ``res_``
    prefix (if any) are ignored. Only used when
    ``on_power_flow_error == "keep_previous"`` so the normal path pays no
    copy cost.
    """
    return {k: v.copy() for k, v in net.items() if k.startswith("res_") and isinstance(v, pd.DataFrame)}


def _restore_res_tables(net: pp.pandapowerNet, snapshot: dict[str, pd.DataFrame]) -> None:
    """Write every DataFrame from *snapshot* back onto *net* under its key.

    Used to undo a failed in-loop power-flow call so callers see the last
    successful ``res_*`` state. Does **not** touch non-``res_`` keys or keys
    present on *net* but missing from *snapshot* (e.g. tables created
    mid-flight by the failed solver call).
    """
    for k, v in snapshot.items():
        net[k] = v


def _run_power_flow(
    net: pp.pandapowerNet,
    method: Literal["ac", "dc"],
    runpp_kwargs: dict[str, Any],
) -> None:
    """Dispatch to the right pandapower solver.

    * ``method="ac"`` → :func:`pandapower.runpp` (Newton AC power flow).
    * ``method="dc"`` → :func:`pandapower.rundcpp` (linear DC power flow).

    *runpp_kwargs* is forwarded verbatim. This helper exists so the
    pre-loop call and every in-loop call go through the exact same code
    path.
    """
    if method == "dc":
        pp.rundcpp(net, **runpp_kwargs)
    else:
        pp.runpp(net, **runpp_kwargs)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def run_spps(
    net: pp.pandapowerNet,
    conditions: pat.DataFrame[SppsConditionsPandapowerSchema],
    actions: pat.DataFrame[SppsActionsPandapowerSchema],
    failed_elements: set[str],
    method: Literal["ac", "dc"] = "ac",
    max_iterations: int = 1,
    runpp_kwargs: dict[str, Any] | None = None,
    on_power_flow_error: Literal["raise", "keep_previous"] = "raise",
) -> SppsResult:
    """Execute the SpPS rule engine.

    Parameters
    ----------
    net
        Pandapower network. Mutated in place.
    conditions
        Condition rows: :class:`SppsConditionsPandapowerSchema`.
    actions
        Action rows sharing ``scheme_name`` with *conditions*:
        :class:`SppsActionsPandapowerSchema`.
    failed_elements
        Compound uids of elements considered failed (for ``"Failed"`` checks).
        Each uid must follow the format ``f"{elem_id}{SEPARATOR}{elem_type}"``
        (see :func:`spps.schema.make_element_uid`) and can reference
        ``switch``, ``line``, ``bus``, ``trafo``, ``trafo3w``, ``impedance``,
        ``ward``, ``xward``, etc. The set is mutated in-place as switches are
        opened by activated schemes.
    method
        ``"ac"`` (default) runs ``pp.runpp``; ``"dc"`` runs ``pp.rundcpp``.
    max_iterations
        Upper bound on iterations. The loop stops early when no scheme triggers.
    runpp_kwargs
        Keyword arguments forwarded to the power-flow solver on every call.
    on_power_flow_error
        Strategy for power-flow failures *inside* the iteration loop
        (the initial PF always raises, as there is no previous state to fall
        back on):

        * ``"raise"`` (default) — wrap the underlying exception in
          :class:`spps.errors.SppsPowerFlowError` and re-raise.
        * ``"keep_previous"`` — log a warning, restore ``res_*`` tables from a
          snapshot taken before the failing PF so the returned ``net`` still
          contains the last successful power-flow results, stop the loop, and
          set :attr:`SppsResult.power_flow_failed` to ``True``. Setpoint
          changes applied earlier in the iteration (switch openings, etc.)
          are left in place.

    Returns
    -------
    SppsResult
        See class docstring.

    Raises
    ------
    SppsPowerFlowError
        If the initial power flow fails, or an iteration-level power flow
        fails while ``on_power_flow_error="raise"``.
    """
    schemes_per_iter: list[list[str]] = []
    activated_scheme_names: set[str] = set()
    max_iterations_reached = False
    power_flow_failed = False
    iterations = 0
    runpp_kwargs = runpp_kwargs or {}

    # IMPORTANT:
    # Must be executed once before the loop.
    # Energization is treated as static; elements de-energized during the process
    # are handled as "failed" instead of updating energized status dynamically.
    _populate_energized(conditions, net)

    try:
        _run_power_flow(net, method, runpp_kwargs)
    except Exception as exc:
        raise SppsPowerFlowError(f"Initial power flow failed; cannot run SpPS: {exc}") from exc

    for iteration in range(1, max_iterations + 1):
        iterations = iteration

        _populate_failed(conditions, failed_elements, net)
        _extract_condition_values(conditions, net)
        _evaluate_conditions(conditions)
        candidate_schemes = _satisfied_scheme_names(conditions)
        new_schemes = sorted(n for n in candidate_schemes if n not in activated_scheme_names)
        active_actions = actions[actions["scheme_name"].isin(new_schemes)].copy() if new_schemes else actions.iloc[0:0]
        active_actions = active_actions.assign(_iteration=iteration)

        schemes_per_iter.append(new_schemes)
        logger.info(
            "Iteration %d: activated %d scheme(s), %d action row(s): %s",
            iteration,
            len(new_schemes),
            len(active_actions),
            new_schemes,
        )

        if not new_schemes:
            logger.info("No new schemes triggered — stopping.")
            break

        switch_activations = active_actions[active_actions["measure_element_table"] == "switch"]
        if not switch_activations.empty:
            opened_uids = switch_activations["measure_element_table_id"].astype(int).astype(str) + SEPARATOR + "switch"
            failed_elements.update(opened_uids.tolist())

        activated_scheme_names.update(new_schemes)
        _apply_actions(active_actions, net)

        if iteration < max_iterations:
            snapshot = _snapshot_res_tables(net) if on_power_flow_error == "keep_previous" else None
            try:
                _run_power_flow(net, method, runpp_kwargs)
            except Exception as exc:
                if on_power_flow_error == "raise":
                    raise SppsPowerFlowError(f"Power flow failed on iteration {iteration}: {exc}") from exc
                logger.warning(
                    "Power flow failed on iteration %d; keeping previous res_* tables and stopping loop: %s",
                    iteration,
                    exc,
                )
                if snapshot is not None:
                    _restore_res_tables(net, snapshot)
                power_flow_failed = True
                break
        else:
            max_iterations_reached = True

    if max_iterations_reached:
        logger.warning(
            "SpPS exhausted max_iterations=%d while still activating schemes.",
            max_iterations,
        )

    return SppsResult(
        net=net,
        iterations=iterations,
        activated_schemes_per_iter=schemes_per_iter,
        max_iterations_reached=max_iterations_reached,
        power_flow_failed=power_flow_failed,
    )
