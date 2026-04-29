# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Core rule-engine algorithm: ``run_spps`` and its private helpers.

Mental model
------------
A *scheme* is a group of rows sharing the same ``scheme_name`` across the
*conditions* and *actions* DataFrames. A scheme is activated on an iteration
when its condition rows pass according to each row's ``condition_logic``:
``all`` requires **every** condition to be true; ``any`` requires **at least
one**. Activation applies **every** action row with that ``scheme_name`` to
the network, and the loop re-runs a power flow so the next iteration sees the
new state.

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
from toop_engine_interfaces.spps_parameters import (
    SppsConditionCheckType,
    SppsConditionLogic,
    SppsConditionSide,
    SppsConditionType,
    SppsPowerFlowFailurePolicy,
    SppsSwitchActionTarget,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helpers — state population
# --------------------------------------------------------------------------- #


def _populate_energized(conditions: pat.DataFrame[SppsConditionsPandapowerSchema], net: pp.pandapowerNet) -> None:
    """Write an ``energized`` boolean column onto *conditions* from the network state.

    For each row, the condition element is looked up in its pandapower table
    and a single "electrically alive" flag is read:

    * ``switch`` → ``net.switch.closed`` (``True`` = closed = energized)
    * any other table in :data:`ELEMENT_TABLES` → ``net.<table>.in_service``

    The ``energized`` column is used for the wire value ``de_energized`` in
    :func:`_evaluate_conditions` and for NaN-based auto-detection in
    :func:`_populate_failed` (only energized elements can be marked failed
    in place from result tables). Tables not in :data:`ELEMENT_TABLES` leave
    ``pd.NA`` for that row. Column dtype is nullable pandas ``"boolean"``.

    Parameters
    ----------
    conditions : pat.DataFrame[SppsConditionsPandapowerSchema]
        Condition rows; must include ``condition_element_table`` and
        ``condition_element_table_id``. Mutated in place.
    net : pandapower.pandapowerNet
        Network whose static element tables (``switch`` / ``in_service``) are
        read. Not modified.

    Returns
    -------
    None
        *conditions* gains a new ``energized`` column; *net* is unchanged.
    """
    conditions["energized"] = pd.Series(pd.NA, index=conditions.index, dtype="boolean")
    for table in ELEMENT_TABLES:
        mask = conditions["condition_element_table"] == table
        if not mask.any():
            continue
        ids = conditions.loc[mask, "condition_element_table_id"]
        element_df = getattr(net, table)
        col = "closed" if table == "switch" else "in_service"
        conditions.loc[mask, "energized"] = element_df.loc[ids, col].to_numpy()


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
    conditions: pat.DataFrame[SppsConditionsPandapowerSchema],
    failed_elements: set[str],
    net: pp.pandapowerNet,
) -> None:
    """Write a ``failed`` boolean column onto *conditions* for ``"failed"`` checks.

    Only rows whose ``condition_check_type == "failed"`` are evaluated; every
    other row gets ``failed = False`` and is ignored by
    :func:`_evaluate_conditions`'s ``"failed"`` branch.

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
    conditions
        Conditions DataFrame. Must already have ``energized`` populated (see
        :func:`_populate_energized`). Mutated in place.
    failed_elements
        Set of compound uids (see :func:`spps.schema.make_element_uid`).
    net
        Pandapower network; only ``res_*`` tables are read.
    """
    conditions["failed"] = False
    failed_type_mask = conditions["condition_check_type"] == SppsConditionCheckType.FAILED
    if not failed_type_mask.any():
        return

    uids = (
        conditions["condition_element_table_id"].astype(str) + SEPARATOR + conditions["condition_element_table"].astype(str)
    )

    conditions.loc[failed_type_mask, "failed"] = uids.loc[failed_type_mask].isin(failed_elements).to_numpy()

    energized = conditions["energized"].fillna(False).astype(bool)

    for table in _FAILURE_RESULT_TABLES:
        mask = (conditions["condition_element_table"] == table) & failed_type_mask
        if not mask.any():
            continue
        res = getattr(net, f"res_{table}", None)
        if res is None or res.empty:
            continue

        ids = conditions.loc[mask, "condition_element_table_id"]
        has_nan = res.reindex(ids).isna().any(axis=1).to_numpy()
        auto = energized.loc[mask].to_numpy() & has_nan
        conditions.loc[mask, "failed"] = conditions.loc[mask, "failed"].to_numpy() | auto


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

        max_mask = type_mask & (rules["condition_side"] == SppsConditionSide.MAXIMUM_VALUE)
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


def _extract_bus_voltage(conditions: pat.DataFrame[SppsConditionsPandapowerSchema], net: pp.pandapowerNet) -> None:
    """Copy bus voltage magnitudes (p.u.) into ``condition_element_value``.

    Only rows with ``condition_element_table == "bus"`` and
    ``condition_type`` equal to :attr:`SppsConditionType.VOLTAGE` are updated;
    ``vm_pu`` is read from ``net.res_bus``. All voltages in this path are in
    per-unit; ``condition_limit_value`` for those rows should already be in
    p.u. (see :func:`spps.preprocessing.convert_voltage_rules_to_pu`) so
    :func:`_evaluate_conditions` compares like units.

    Parameters
    ----------
    conditions : pat.DataFrame[SppsConditionsPandapowerSchema]
        Rule rows; must already have ``condition_element_value`` (typically
        pre-filled by :func:`_extract_condition_values`). Mutated in place.
    net : pandapower.pandapowerNet
        Network with ``res_bus`` containing ``vm_pu`` for the current solve.
        Not modified.

    Returns
    -------
    None
        Matching rows in *conditions* get ``condition_element_value`` from
        ``res_bus``; other rows are left unchanged by this helper.
    """
    mask = (conditions["condition_element_table"] == "bus") & (conditions["condition_type"] == SppsConditionType.VOLTAGE)
    if not mask.any():
        return
    ids = conditions.loc[mask, "condition_element_table_id"]
    conditions.loc[mask, "condition_element_value"] = net.res_bus.loc[ids, "vm_pu"].to_numpy()


def _extract_condition_values(conditions: pat.DataFrame[SppsConditionsPandapowerSchema], net: pp.pandapowerNet) -> None:
    """Populate ``condition_element_value`` with the current power-flow result.

    The column is initialised to NaN for every row, then values are filled from
    ``net`` result tables: branch-style elements (line, trafo, trafo3w) use
    :data:`RESULT_COLUMNS` via :func:`_extract_res_values`, and bus rows use
    :func:`_extract_bus_voltage`. Unmatched ``(condition_element_table,
    condition_type, condition_side)`` combinations stay NaN and fail numeric
    checks in :func:`_evaluate_conditions`.

    Parameters
    ----------
    conditions : pat.DataFrame[SppsConditionsPandapowerSchema]
        Rule rows; must include ``condition_element_table`` and related
        condition columns. Mutated in place.
    net : pandapower.pandapowerNet
        Network with converged or last-good ``res_*`` tables (``res_bus``,
        ``res_line``, etc.) for the current solve. Not modified.

    Returns
    -------
    None
        *conditions* gains a ``condition_element_value`` column (float); *net*
        is unchanged.
    """
    conditions["condition_element_value"] = pd.Series(float("nan"), index=conditions.index, dtype="float64")
    for element_type, col_map in RESULT_COLUMNS.items():
        _extract_res_values(conditions, net, element_type, col_map)
    _extract_bus_voltage(conditions, net)


# --------------------------------------------------------------------------- #
# Internal helpers — condition evaluation & activation
# --------------------------------------------------------------------------- #


def _evaluate_conditions(conditions: pat.DataFrame[SppsConditionsPandapowerSchema]) -> None:
    """Evaluate every rule's check and write the result into ``is_condition``.

    For each row, ``condition_check_type`` (see :class:`SppsConditionCheckType`
    wire values) selects the branch:

    * :attr:`SppsConditionCheckType.GT` / LT / EQ — compare ``condition_element_value`` to
      ``condition_limit_value`` for numeric thresholds.
    * :attr:`SppsConditionCheckType.FAILED` — use the ``failed`` column from
      :func:`_populate_failed`.
    * :attr:`SppsConditionCheckType.DE_ENERGIZED` — use ``energized`` from
      :func:`_populate_energized` (pass when ``energized`` is false).

    NaNs in value/limit/failed/energized comparisons yield ``False`` for that
    row. Unknown or unsupported check types keep ``is_condition == False``
    (consider validating upstream with :class:`SppsConditionsPandapowerSchema`).

    Parameters
    ----------
    conditions : pat.DataFrame[SppsConditionsPandapowerSchema]
        Conditions rows. Must have ``condition_check_type``, ``condition_element_value``,
        ``condition_limit_value``, and for state checks the ``failed`` and
        ``energized`` columns from :func:`_populate_failed` and
        :func:`_populate_energized`. Mutated in place.

    Returns
    -------
    None
        *conditions* gets a boolean column ``is_condition`` (overwritten in full
        on each call).
    """
    conditions["is_condition"] = False

    check = conditions["condition_check_type"]
    value = conditions["condition_element_value"]
    limit = conditions["condition_limit_value"]

    gt_mask = check == SppsConditionCheckType.GT
    conditions.loc[gt_mask, "is_condition"] = (value[gt_mask] > limit[gt_mask]).fillna(False)

    lt_mask = check == SppsConditionCheckType.LT
    conditions.loc[lt_mask, "is_condition"] = (value[lt_mask] < limit[lt_mask]).fillna(False)

    eq_mask = check == SppsConditionCheckType.EQ
    conditions.loc[eq_mask, "is_condition"] = (value[eq_mask] == limit[eq_mask]).fillna(False)

    failed_mask = check == SppsConditionCheckType.FAILED
    conditions.loc[failed_mask, "is_condition"] = conditions.loc[failed_mask, "failed"].fillna(False)

    de_mask = check == SppsConditionCheckType.DE_ENERGIZED
    conditions.loc[de_mask, "is_condition"] = ~conditions.loc[de_mask, "energized"].fillna(False)


def _satisfied_scheme_names(conditions: pat.DataFrame[SppsConditionsPandapowerSchema]) -> set[str]:
    """Return scheme names whose condition rows pass under that scheme's ``condition_logic``.

    For each ``scheme_name``, all condition rows carry the same ``condition_logic``
    (``all`` = logical AND over ``is_condition``, ``any`` = logical OR).

    If ``condition_logic`` is absent (legacy tables), ``all`` is assumed.

    Parameters
    ----------
    conditions
        Conditions SppsConditionsPandapowerSchema
    """
    ic = conditions["is_condition"].fillna(False)
    names = conditions["scheme_name"]
    if "condition_logic" not in conditions.columns:
        satisfied = ic.groupby(names).all()
        return set(satisfied[satisfied].index.tolist())

    out: set[str] = set()
    for scheme_name, grp in conditions.groupby("scheme_name", sort=False):
        mode = grp["condition_logic"].iloc[0]
        g_ic = ic.loc[grp.index]
        if mode == SppsConditionLogic.ANY.value:
            if bool(g_ic.any()):
                out.add(scheme_name)
        elif bool(g_ic.all()):
            out.add(scheme_name)
    return out


# --------------------------------------------------------------------------- #
# Internal helpers — action application
# --------------------------------------------------------------------------- #


def _apply_switch_actions(actions: pd.DataFrame, net: pp.pandapowerNet) -> None:
    """Apply switch rows in *actions* by writing ``net.switch.closed``.

    Only rows with ``measure_element_table == "switch"`` are considered.
    ``measure_value`` is compared to :attr:`SppsSwitchActionTarget.CLOSED`
    (wire value ``"closed"``); a match sets ``closed = True``, any other
    value sets ``closed = False``. If there are no switch rows, this is a
    no-op.

    Parameters
    ----------
    actions : pd.DataFrame
        Must include ``measure_element_table``, ``measure_element_table_id``,
        and ``measure_value``. Not modified.
    net : pandapower.pandapowerNet
        ``net.switch`` rows referenced by id are updated in place.

    Returns
    -------
    None
        *net.switch.closed* is written for the affected indices; *actions* is
        unchanged.
    """
    sw = actions[actions["measure_element_table"] == "switch"]
    if sw.empty:
        return
    ids = sw["measure_element_table_id"]
    closed = (sw["measure_value"] == SppsSwitchActionTarget.CLOSED.value).to_numpy()
    net.switch.loc[ids, "closed"] = closed
    logger.debug("Applied %d switch actions (ids=%s)", len(sw), ids.tolist())


def _apply_actions(actions: pat.DataFrame[SppsActionsPandapowerSchema], net: pp.pandapowerNet) -> None:
    """Apply every action row in *actions* to *net*.

    * Rows with ``measure_element_table == "switch"`` go to
      :func:`_apply_switch_actions` (``net.switch.closed`` from ``measure_value``).
    * All other supported ``(measure_element_table, measure_type)`` pairs are
      resolved via :data:`ACTION_COLUMNS` to a column on ``net.<table>`` and
      written with ``pd.to_numeric`` (``p_mw``, ``q_mvar``, ``vm_pu``, etc.).

    Voltage setpoints must already be in per-unit; see
    :func:`spps.preprocessing.convert_voltage_rules_to_pu`. Combinations not
    present in :data:`ACTION_COLUMNS` are skipped. If *actions* is empty,
    this is a no-op.

    Parameters
    ----------
    actions : pat.DataFrame[SppsActionsPandapowerSchema]
        One row per action to apply (``scheme_name``, ``measure_element_table``,
        ``measure_element_table_id``, ``measure_type``, ``measure_value``).
    net : pandapower.pandapowerNet
        Network whose element and switch tables are updated in place.

    Returns
    -------
    None
        *net* is mutated; *actions* is not modified.
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
    prefix (if any) are ignored.     Only used when
    :attr:`SppsPowerFlowFailurePolicy.KEEP_PREVIOUS` is selected so the normal
    path pays no copy cost.
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
    """Dispatch to the pandapower load-flow solver for *net*.

    * ``method="ac"`` — :func:`pandapower.runpp` (AC Newton power flow).
    * ``method="dc"`` — :func:`pandapower.rundcpp` (linear DC power flow).

    ``runpp_kwargs`` is forwarded unchanged. Used so the initial solve and every
    in-loop solve share one code path. Exceptions (e.g. non-convergence) are
    not caught here.

    Parameters
    ----------
    net : pandapower.pandapowerNet
        Network to solve; ``res_*`` and element tables are updated in place by
        pandapower.
    method : {"ac", "dc"}
        ``"ac"`` for full AC, ``"dc"`` for DC approximation.
    runpp_kwargs : dict[str, Any]
        Extra keyword arguments for :func:`pandapower.runpp` or
        :func:`pandapower.rundcpp` (empty dict is fine).

    Returns
    -------
    None
        *net* holds the solver result on success; on failure, pandapower raises.
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
    on_power_flow_error: SppsPowerFlowFailurePolicy = SppsPowerFlowFailurePolicy.RAISE,
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
        Compound uids of elements considered failed (for ``"failed"`` checks).
        Each uid must follow the format ``f"{elem_id}{SEPARATOR}{elem_type}"``
        (see :func:`spps.schema.make_element_uid`) and can reference
        ``switch``, ``line``, ``bus``, ``trafo``, ``trafo3w``, ``impedance``,
        ``ward``, ``xward``, etc. The set is mutated in-place as switches are
        opened by activated schemes.
    method
        ``"ac"`` (default) runs ``pp.runpp``; ``"dc"`` runs ``pp.rundcpp``.
    max_iterations
        Upper bound on iterations. The loop stops early when no scheme triggers.
        After each batch of actions (including on the final iteration), a
        power flow is run so ``res_*`` matches the updated net.
    runpp_kwargs
        Keyword arguments forwarded to the power-flow solver on every call.
    on_power_flow_error
        Strategy for power-flow failures after applying actions (the initial PF
        always raises, as there is no previous state to fall back on):

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
        If the initial power flow fails, or a post-action power flow fails
        while ``on_power_flow_error="raise"``. Subclasses of :class:`SppsError`
        raised by the solver path are not double-wrapped.
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

    _run_power_flow(net, method, runpp_kwargs)

    for iteration in range(1, max_iterations + 1):
        _populate_failed(conditions, failed_elements, net)
        _extract_condition_values(conditions, net)
        _evaluate_conditions(conditions)
        candidate_schemes = _satisfied_scheme_names(conditions)
        new_schemes = sorted(n for n in candidate_schemes if n not in activated_scheme_names)
        if not new_schemes:
            logger.info("No new schemes triggered — stopping.")
            break

        iterations = iteration
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

        switch_activations = active_actions[active_actions["measure_element_table"] == "switch"]
        if not switch_activations.empty:
            opened_uids = switch_activations["measure_element_table_id"].astype(int).astype(str) + SEPARATOR + "switch"
            failed_elements.update(opened_uids.tolist())

        activated_scheme_names.update(new_schemes)
        _apply_actions(active_actions, net)

        snapshot = _snapshot_res_tables(net) if on_power_flow_error == SppsPowerFlowFailurePolicy.KEEP_PREVIOUS else None
        try:
            _run_power_flow(net, method, runpp_kwargs)
        except Exception as exc:
            if on_power_flow_error == SppsPowerFlowFailurePolicy.RAISE:
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

        if iteration == max_iterations:
            max_iterations_reached = True

    if max_iterations_reached:
        logger.warning(
            "SpPS exhausted max_iterations=%d while still activating schemes.",
            max_iterations,
        )

        if on_power_flow_error == SppsPowerFlowFailurePolicy.RAISE:
            raise SppsPowerFlowError(
                f"SpPS exhausted max_iterations={max_iterations} while new schemes were still activating."
            )

    return SppsResult(
        net=net,
        iterations=iterations,
        activated_schemes_per_iter=schemes_per_iter,
        max_iterations_reached=max_iterations_reached,
        power_flow_failed=power_flow_failed,
    )
