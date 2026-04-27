# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Utilities for translating CGMES and globally unique IDs into pandapower tables, monitored elements."""

import numpy as np
import pandapower
import pandas as pd
import pandera as pa
import pandera.typing as pat
from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
    PandapowerElements,
    PandapowerMonitoredElementSchema,
    SppsActionsPandapowerSchema,
    SppsConditionsPandapowerSchema,
)
from toop_engine_grid_helpers.pandapower.pandapower_helpers import get_element_table
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    parse_globally_unique_id,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.nminus1_definition import (
    Action,
    Condition,
    Contingency,
    GridElement,
    SppsRule,
)


def get_cgmes_id_to_table_df(net: pandapowerNet) -> tuple[pd.DataFrame, list[str]]:
    """Get a DataFrame mapping cgmes ids to their table and table_id.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.

    Returns
    -------
    cgmes_ids: pd.DataFrame
        A DataFrame mapping cgmes ids to their table and table_id.
    duplicated_ids: list[str]
        A list of ids that were not unique in the grid. This is only relevant for cgmes ids.
    """
    cgmes_ids = pd.DataFrame(columns=["origin_id", "table", "table_id"])
    for element in pandapower.toolbox.pp_elements(bus_elements=True, branch_elements=True, other_elements=False):
        element_df = net[element].reset_index(names="table_id").assign(table=element)
        element_df["origin_id"] = (
            element_df["origin_id"].astype(str)
            if "origin_id" in element_df.columns
            else np.full_like(element_df.index, np.nan, dtype=object)
        )
        cgmes_ids = pd.concat([cgmes_ids, element_df[["origin_id", "table", "table_id"]]], ignore_index=True)
    cgmes_ids.dropna(subset=["origin_id"], inplace=True)
    cgmes_ids.index = cgmes_ids.origin_id
    is_duplicate = cgmes_ids.index.duplicated(keep="first")
    duplicated_ids = cgmes_ids.index[is_duplicate].astype(str).unique().tolist()
    cgmes_ids = cgmes_ids[~is_duplicate]
    return cgmes_ids, duplicated_ids


def extract_contingencies_with_unique_pandapower_id(
    net: pandapowerNet, contingencies: list[Contingency]
) -> tuple[list[PandapowerContingency], list[Contingency], list[str]]:
    """Extract contingencies with unique pandapower ids.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    contingencies : list[Contingency]
        The list of contingencies to translate.

    Returns
    -------
    pp_contingencies: list[PandapowerContingency]
        A list translated Contingency to be used in Pandapower.
    missing_contingencies: list[Contingency]
        A list of contingencies that were not found in the network.
    duplicated_ids: list[str]
        A list of ids that were not unique in the grid. This is only relevant for cgmes ids.
    """
    pp_contingencies = []
    missing_contingencies = []
    duplicated_ids = []
    for contingency in contingencies:
        outaged_elements = []
        for element in contingency.elements:
            try:
                pp_id, pp_type = parse_globally_unique_id(element.id)
            except ValueError:
                # If the id is not a globally unique id, we cannot translate it
                missing_contingencies.append(contingency)
                break
            table = get_element_table(pp_type, res_table=False)
            if pp_id not in net[table].index:
                missing_contingencies.append(contingency)
                break
            outaged_elements.append(
                PandapowerElements(unique_id=element.id, table=table, table_id=pp_id, name=element.name or "")
            )
        else:
            pp_contingency = PandapowerContingency(
                unique_id=contingency.id,
                name=contingency.name or "",
                elements=outaged_elements,
            )
            pp_contingencies.append(pp_contingency)
    return pp_contingencies, missing_contingencies, duplicated_ids


def extract_monitored_elements_with_cgmes_id(
    net: pandapowerNet, monitored_elements: list[GridElement]
) -> tuple[pat.DataFrame[PandapowerMonitoredElementSchema], list[GridElement], list[str]]:
    """Extract monitored elements with unique cgmes guids.

    Uses the globally unique ids of the elements to find them in the pandapower network columns "origin_id".

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    monitored_elements : list[GridElement]
        The list of monitored elements to translate.

    Returns
    -------
    pat.DataFrame[PandapowerMonitoredElementSchema]
        A pandas DataFrame containing the monitored elements with their globally unique ids, table, table_id, kind and name.
    list[GridElement]
        A list of monitored elements that were not found in the network.
    list[str]
        A list of ids that were not unique in the grid. This is only relevant for cgmes ids.
    """
    pandapower_monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    cgmes_ids, duplicated_ids = get_cgmes_id_to_table_df(net)

    pandapower_monitored_elements = pandapower_monitored_elements.reindex([element.id for element in monitored_elements])
    pandapower_monitored_elements["name"] = [element.name for element in monitored_elements]
    pandapower_monitored_elements["kind"] = [element.kind for element in monitored_elements]
    pandapower_monitored_elements["table"] = cgmes_ids["table"]
    pandapower_monitored_elements["table_id"] = cgmes_ids["table_id"].astype(int)
    pandapower_monitored_elements.dropna(subset=["table", "table_id"], inplace=True)
    missing_elements = [element for element in monitored_elements if element.id not in pandapower_monitored_elements.index]
    duplicated_ids = [element.id for element in monitored_elements if element.id in duplicated_ids]
    return pandapower_monitored_elements, missing_elements, duplicated_ids


@pa.check_types
def extract_monitored_elements_with_unique_pandapower_id(
    net: pandapowerNet, monitored_elements: list[GridElement]
) -> tuple[pat.DataFrame[PandapowerMonitoredElementSchema], list[GridElement], list[str]]:
    """Extract monitored elements with unique pandapower ids.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    monitored_elements : list[GridElement]
        The list of monitored elements to translate.

    Returns
    -------
    pat.DataFrame[PandapowerMonitoredElementSchema]
        A pandas DataFrame containing the monitored elements with their globally unique ids, table, table_id, kind and name.
    list[GridElement]
        A list of monitored elements that were not found in the network.
    list[str]
        A list of ids that were not unique in the grid. This is only relevant for cgmes ids.
    """
    empty_df = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)
    missing_elements = []
    index = []
    df_rows = []
    for element in monitored_elements:
        try:
            pp_id, pp_type = parse_globally_unique_id(element.id)
        except ValueError:
            # If the id is not a globally unique id, we cannot translate it
            missing_elements.append(element)
            continue
        table = get_element_table(pp_type, res_table=False)
        if pp_id not in net[table].index.values:
            missing_elements.append(element)
            continue
        index.append(element.id)
        df_rows.append(
            {
                "table": table,
                "table_id": pp_id,
                "kind": element.kind,
                "name": element.name or "",
            }
        )

    pandapower_monitored_elements = pd.concat([empty_df, pd.DataFrame(df_rows, index=index)])
    return pandapower_monitored_elements, missing_elements, []


def extract_contingencies_with_cgmes_id(
    net: pandapowerNet, contingencies: list[Contingency]
) -> tuple[list[PandapowerContingency], list[Contingency], list[str]]:
    """Extract contingencies with unique cgmes ids.

    Uses the globally unique ids of the elements to find them in the pandapower network columns "origin_id".

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    contingencies : list[Contingency]
        The list of contingencies to translate.

    Returns
    -------
    pp_contingencies: list[PandapowerContingency]
        A list translated Contingency to be used in Pandapower.
    missing_contingencies: list[Contingency]
        A list of contingencies that were not found in the network.
    duplicated_ids: list[str]
        A list of ids that were not unique in the grid. This is only relevant for cgmes ids.
    """
    pp_contingencies = []
    missing_contingencies = []

    cgmes_ids, duplicated_ids = get_cgmes_id_to_table_df(net)

    for contingency in contingencies:
        outaged_elements = []
        for element in contingency.elements:
            try:
                table_id, table = cgmes_ids.loc[element.id, ["table_id", "table"]].tolist()
            except KeyError:
                # If the id is not found in the cgmes_ids, we cannot translate it
                missing_contingencies.append(contingency)
                break
            outaged_elements.append(
                PandapowerElements(unique_id=element.id, table=table, table_id=table_id, name=element.name or "")
            )
        else:
            pp_contingency = PandapowerContingency(
                unique_id=contingency.id,
                name=contingency.name or "",
                elements=outaged_elements,
            )
            pp_contingencies.append(pp_contingency)
    duplicated_ids = [
        monitored_element.id
        for contingency in contingencies
        for monitored_element in contingency.elements
        if monitored_element.id in duplicated_ids
    ]
    return pp_contingencies, missing_contingencies, duplicated_ids


def _resolve_unique_pandapower_element(net: pandapowerNet, unique_id: str) -> tuple[str, int]:
    """Resolve a pandapower globally unique id into a ``(table, table_id)`` pair.

    Raises ``ValueError`` if the id cannot be parsed or the element is not
    present in ``net``.
    """
    pp_id, pp_type = parse_globally_unique_id(unique_id)
    table = get_element_table(pp_type, res_table=False)
    if pp_id not in net[table].index:
        raise ValueError(f"Pandapower id {unique_id!r} not found in net")
    return table, int(pp_id)


def _build_spps_condition_rows(
    rule: SppsRule,
    resolved_conditions: list[tuple[Condition, str, int]],
) -> list[dict]:
    """Build one DataFrame row per resolved condition, including ``scheme_name``."""
    rows = []
    for condition, cond_table, cond_table_id in resolved_conditions:
        rows.append(
            {
                "scheme_name": rule.scheme_name,
                "condition_type": condition.condition_type,
                "condition_check_type": condition.condition_check_type,
                "condition_side": condition.condition_side,
                "condition_limit_value": condition.condition_limit_value,
                "condition_element_table": cond_table,
                "condition_element_table_id": cond_table_id,
            }
        )
    return rows


def _build_spps_action_rows(
    rule: SppsRule,
    resolved_actions: list[tuple[Action, str, int]],
) -> list[dict]:
    """Build one DataFrame row per resolved action, including ``scheme_name``."""
    rows = []
    for action, meas_table, meas_table_id in resolved_actions:
        rows.append(
            {
                "scheme_name": rule.scheme_name,
                "measure_element_table": meas_table,
                "measure_element_table_id": meas_table_id,
                "measure_type": action.measure_type,
                "measure_value": action.measure_value,
            }
        )
    return rows


@pa.check_types
def extract_spps_rules_with_unique_pandapower_id(
    net: pandapowerNet,
    rules: list[SppsRule],
) -> tuple[
    pat.DataFrame[SppsConditionsPandapowerSchema], pat.DataFrame[SppsActionsPandapowerSchema], list[SppsRule], list[str]
]:
    """Extract SpPS rules into pandapower condition and action tables using globally unique ids.

    Each condition's ``condition_element_unique_id`` and each action's
    ``measure_element_unique_id`` are parsed as pandapower globally unique ids
    (``"<table_id>%%<table>"``) and resolved into ``(table, table_id)`` pairs.
    Resolved pairs are validated against ``net``.

    Rules whose conditions or actions reference ids not present in ``net`` are
    dropped from the result and returned in ``missing_rules``.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network used to validate resolved condition / action ids.
    rules : list[SppsRule]
        The list of SpPS rules to translate.

    Returns
    -------
    pat.DataFrame[SppsConditionsPandapowerSchema]
        One row per condition, keyed by ``scheme_name``.
    pat.DataFrame[SppsActionsPandapowerSchema]
        One row per action, keyed by ``scheme_name``.
    list[SppsRule]
        SpPS rules dropped because a condition or action id could not be
        parsed or the referenced element was not found in ``net``.
    list[str]
        A list of ids that were not unique in the grid. This is only relevant
        for cgmes ids and is always empty in this mode.
    """
    empty_conditions = get_empty_dataframe_from_model(SppsConditionsPandapowerSchema)
    empty_actions = get_empty_dataframe_from_model(SppsActionsPandapowerSchema)
    cond_rows: list[dict] = []
    act_rows: list[dict] = []
    missing_rules: list[SppsRule] = []
    if not rules:
        return empty_conditions, empty_actions, missing_rules, []

    for rule in rules:
        try:
            resolved_conditions = [
                (c, *_resolve_unique_pandapower_element(net, c.condition_element_unique_id)) for c in rule.conditions
            ]
            resolved_actions = [
                (a, *_resolve_unique_pandapower_element(net, a.measure_element_unique_id)) for a in rule.actions
            ]
        except ValueError:
            missing_rules.append(rule)
            continue
        cond_rows.extend(_build_spps_condition_rows(rule, resolved_conditions))
        act_rows.extend(_build_spps_action_rows(rule, resolved_actions))

    spps_conditions = pd.concat([empty_conditions, pd.DataFrame(cond_rows)], ignore_index=True)
    spps_actions = pd.concat([empty_actions, pd.DataFrame(act_rows)], ignore_index=True)
    return spps_conditions, spps_actions, missing_rules, []


def _resolve_cgmes_element(
    cgmes_ids: pd.DataFrame,
    duplicated_ids: list[str],
    seen_duplicated_ids: list[str],
    guid: str,
) -> tuple[str, int]:
    """Resolve a CGMES GUID into a pandapower ``(table, table_id)`` pair.

    Appends ``guid`` to ``seen_duplicated_ids`` when it was flagged as
    non-unique during the ``cgmes_ids`` lookup build.

    Raises ``KeyError`` if the GUID is not present in ``cgmes_ids``.
    """
    if guid in duplicated_ids:
        seen_duplicated_ids.append(guid)
    table_id, table = cgmes_ids.loc[guid, ["table_id", "table"]].tolist()
    return table, int(table_id)


@pa.check_types
def extract_spps_rules_with_cgmes_id(
    net: pandapowerNet,
    rules: list[SppsRule],
) -> tuple[
    pat.DataFrame[SppsConditionsPandapowerSchema], pat.DataFrame[SppsActionsPandapowerSchema], list[SppsRule], list[str]
]:
    """Extract SpPS rules into pandapower condition and action tables, resolving CGMES GUIDs.

    In CGMES mode each condition's ``condition_element_unique_id`` and each
    action's ``measure_element_unique_id`` carry the CGMES GUID of the target
    element. Both are resolved to pandapower ``(table, table_id)`` pairs via
    the network's ``origin_id`` columns.

    Rules for which any condition or action GUID cannot be resolved to an
    element in ``net`` are dropped from the result and returned in
    ``missing_rules``.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network used to resolve CGMES GUIDs into pandapower
        table names and row ids.
    rules : list[SppsRule]
        The list of SpPS rules to translate. Each condition's
        ``condition_element_unique_id`` and each action's
        ``measure_element_unique_id`` must contain the CGMES GUID of the
        target element.

    Returns
    -------
    pat.DataFrame[SppsConditionsPandapowerSchema]
    pat.DataFrame[SppsActionsPandapowerSchema]
    list[SppsRule]
        The SpPS rules that were dropped because at least one of their
        condition/action GUIDs could not be resolved in ``net``.
    list[str]
        A list of GUIDs that were not unique in the grid.
    """
    empty_conditions = get_empty_dataframe_from_model(SppsConditionsPandapowerSchema)
    empty_actions = get_empty_dataframe_from_model(SppsActionsPandapowerSchema)
    cgmes_ids, duplicated_ids = get_cgmes_id_to_table_df(net)

    cond_rows: list[dict] = []
    act_rows: list[dict] = []
    missing_rules: list[SppsRule] = []
    seen_duplicated_ids: list[str] = []

    if not rules:
        return empty_conditions, empty_actions, missing_rules, []

    for rule in rules:
        try:
            resolved_conditions = [
                (c, *_resolve_cgmes_element(cgmes_ids, duplicated_ids, seen_duplicated_ids, c.condition_element_unique_id))
                for c in rule.conditions
            ]
            resolved_actions = [
                (a, *_resolve_cgmes_element(cgmes_ids, duplicated_ids, seen_duplicated_ids, a.measure_element_unique_id))
                for a in rule.actions
            ]
        except KeyError:
            missing_rules.append(rule)
            continue
        cond_rows.extend(_build_spps_condition_rows(rule, resolved_conditions))
        act_rows.extend(_build_spps_action_rows(rule, resolved_actions))

    spps_conditions = pd.concat([empty_conditions, pd.DataFrame(cond_rows)], ignore_index=True)
    spps_actions = pd.concat([empty_actions, pd.DataFrame(act_rows)], ignore_index=True)
    return spps_conditions, spps_actions, missing_rules, list(dict.fromkeys(seen_duplicated_ids))
