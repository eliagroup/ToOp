# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Convert three-winding transformers in an N-1 definition to their two-winding legs.

The importer replaces every three-winding transformer by three two-winding transformers
``<id>-Leg1``/``-Leg2``/``-Leg3`` (``pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers``).
N-1 definitions reference three-winding transformers by their original id until this conversion stage, which is the
last stage of the N-1 definition pipeline before the definition is saved next to the converted grid.
"""

from typing import TypeVar

import structlog
from pypowsybl.network.impl.network import Network
from toop_engine_interfaces.nminus1_definition import (
    Action,
    Condition,
    GridElement,
    Nminus1Definition,
    SppsRule,
)

logger = structlog.get_logger(__name__)

# Regex matching the suffix of the two-winding legs of a converted three-winding transformer
CONVERTED_TRAFO3W_ENDING = "-Leg[123]$"

GridElementT = TypeVar("GridElementT", bound=GridElement)


def get_converted_transformer_legs(network: Network) -> dict[str, list[GridElement]]:
    """Map each converted three-winding transformer id to its two-winding legs.

    Parameters
    ----------
    network : pypowsybl.network.Network
        The grid after the three-winding transformer conversion.

    Returns
    -------
    dict[str, list[GridElement]]
        The legs per original three-winding transformer id, ordered Leg1, Leg2, Leg3.
    """
    trafos = network.get_2_windings_transformers(attributes=["name"])
    legs = trafos[trafos.index.str.contains(CONVERTED_TRAFO3W_ENDING)].sort_index()
    legs_by_trafo3w: dict[str, list[GridElement]] = {}
    for leg_id, leg_name in zip(legs.index, legs["name"], strict=True):
        trafo3w_id = leg_id.rsplit("-Leg", maxsplit=1)[0]
        legs_by_trafo3w.setdefault(trafo3w_id, []).append(
            GridElement(id=leg_id, name=leg_name or "", type="TWO_WINDINGS_TRANSFORMER", kind="branch")
        )
    return legs_by_trafo3w


def _convert_elements(
    elements: list[GridElementT], legs_by_trafo3w: dict[str, list[GridElement]], *, context: str
) -> list[GridElementT]:
    """Replace three-winding transformers by their legs, keeping order and dropping duplicates.

    Parameters
    ----------
    elements : list[GridElement]
        Contingency or monitored elements. Subclasses such as MonitoredElement keep their class and extra fields.
    legs_by_trafo3w : dict[str, list[GridElement]]
        The legs from :func:`get_converted_transformer_legs`.
    context : str
        Where the elements are used, for logging.

    Returns
    -------
    list[GridElement]
        The converted elements.
    """
    converted: dict[str, GridElementT] = {}
    for element in elements:
        legs = legs_by_trafo3w.get(element.id)
        if legs is None:
            if element.type == "THREE_WINDINGS_TRANSFORMER":
                logger.warning(
                    "three_winding_transformer_legs_missing",
                    context=context,
                    element_id=element.id,
                    element_name=element.name,
                )
            converted.setdefault(element.id, element)
            continue
        for leg in legs:
            converted.setdefault(
                leg.id, element.model_copy(update={"id": leg.id, "name": leg.name, "type": leg.type, "kind": leg.kind})
            )
    return list(converted.values())


def _convert_spps_rule(rule: SppsRule, legs_by_trafo3w: dict[str, list[GridElement]]) -> SppsRule:
    """Replace every condition and action on a three-winding transformer by one per leg.

    This applies regardless of ``condition_logic``. Under ``ANY`` logic the rule therefore triggers as soon as a single
    leg meets the condition.

    Parameters
    ----------
    rule : SppsRule
        The SPPS rule to convert.
    legs_by_trafo3w : dict[str, list[GridElement]]
        The legs from :func:`get_converted_transformer_legs`.

    Returns
    -------
    SppsRule
        The converted rule.
    """
    conditions: list[Condition] = []
    for condition in rule.conditions:
        legs = legs_by_trafo3w.get(condition.condition_element_unique_id)
        if legs is None:
            conditions.append(condition)
        else:
            conditions.extend(condition.model_copy(update={"condition_element_unique_id": leg.id}) for leg in legs)
    actions: list[Action] = []
    for action in rule.actions:
        legs = legs_by_trafo3w.get(action.measure_element_unique_id)
        if legs is None:
            actions.append(action)
        else:
            actions.extend(action.model_copy(update={"measure_element_unique_id": leg.id}) for leg in legs)
    return rule.model_copy(update={"conditions": conditions, "actions": actions})


def convert_three_winding_transformers_in_nminus1_definition(
    definition: Nminus1Definition, network: Network
) -> Nminus1Definition:
    """Convert every three-winding transformer reference in an N-1 definition to its two-winding legs.

    This is the conversion stage of the N-1 definition pipeline. Contingency and monitored elements that reference a
    converted three-winding transformer are replaced by its three legs. SPPS conditions and actions on it are copied
    once per leg, regardless of the rule's ``condition_logic``. The conversion is idempotent: legs that are already
    present are left unchanged and not duplicated.

    Parameters
    ----------
    definition : Nminus1Definition
        The N-1 definition, referencing three-winding transformers by their original id.
    network : pypowsybl.network.Network
        The grid **after** ``replace_3_windings_transformers_with_3_2_windings_transformers``.

    Returns
    -------
    Nminus1Definition
        The definition referencing only elements of the converted grid. A three-winding transformer whose legs are not
        in the grid is kept as is and logged as a warning.
    """
    legs_by_trafo3w = get_converted_transformer_legs(network)
    contingencies = [
        contingency.model_copy(
            update={"elements": _convert_elements(contingency.elements, legs_by_trafo3w, context="contingency")}
        )
        for contingency in definition.contingencies
    ]
    monitored_elements = _convert_elements(definition.monitored_elements, legs_by_trafo3w, context="monitored_element")
    spps_rules = (
        [_convert_spps_rule(rule, legs_by_trafo3w) for rule in definition.spps_rules]
        if definition.spps_rules is not None
        else None
    )
    converted_definition = definition.model_copy(
        update={"contingencies": contingencies, "monitored_elements": monitored_elements, "spps_rules": spps_rules}
    )
    return Nminus1Definition.model_validate(converted_definition.model_dump())


def get_nminus1_definition_element_ids(definition: Nminus1Definition) -> set[str]:
    """Collect the ids of all elements an N-1 definition references.

    Parameters
    ----------
    definition : Nminus1Definition
        The N-1 definition.

    Returns
    -------
    set[str]
        The ids of all contingency elements, monitored elements and SPPS condition and action elements.
    """
    element_ids = {element.id for contingency in definition.contingencies for element in contingency.elements}
    element_ids.update(element.id for element in definition.monitored_elements)
    for rule in definition.spps_rules or []:
        element_ids.update(condition.condition_element_unique_id for condition in rule.conditions)
        element_ids.update(action.measure_element_unique_id for action in rule.actions)
    return element_ids
