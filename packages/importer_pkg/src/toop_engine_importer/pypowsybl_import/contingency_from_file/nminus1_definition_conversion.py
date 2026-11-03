# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Convert three-winding transformers in an N-1 definition to their two-winding legs ``<id>-Leg1``/``2``/``3``.

This is the last stage of the N-1 definition pipeline before the definition is saved next to the converted grid.
"""

from collections.abc import Iterable
from typing import TypeVar

import structlog
from pydantic import BaseModel
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.trafo3w_legs import get_trafo3w_id
from toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input import GridElementT
from toop_engine_interfaces.messages.preprocess.preprocess_commands import Trafo3wLegMonitoring
from toop_engine_interfaces.nminus1_definition import GridElement, MonitoredElement, Nminus1Definition

logger = structlog.get_logger(__name__)

SppsItemT = TypeVar("SppsItemT", bound=BaseModel)


def _drop_duplicate_ids(elements: Iterable[GridElementT]) -> list[GridElementT]:
    """Keep the first element of every id, in order."""
    unique: dict[str, GridElementT] = {}
    for element in elements:
        unique.setdefault(element.id, element)
    return list(unique.values())


def _copy_to_all_legs(element: GridElementT, legs: list[GridElement]) -> list[GridElementT]:
    """Copy a three-winding transformer element once per leg, keeping all its other fields."""
    return [element.model_copy(update=dict(leg)) for leg in legs]


def _replace_contingency_elements_by_legs(
    elements: list[GridElement], legs_by_trafo3w: dict[str, list[GridElement]]
) -> list[GridElement]:
    """Replace the three-winding transformers of a contingency by all their legs, keeping order and dropping duplicates."""
    replaced: list[GridElement] = []
    for element in elements:
        legs = legs_by_trafo3w.get(element.id)
        if legs is None:
            if element.type == "THREE_WINDINGS_TRANSFORMER":
                logger.warning(
                    "Contingency for 3W transformer kept because grid has no converted legs of that trafo.",
                    element_id=element.id,
                    element_name=element.name,
                )
            replaced.append(element)
        else:
            replaced.extend(_copy_to_all_legs(element, legs))
    return _drop_duplicate_ids(replaced)


def _copy_to_lead_leg(
    element: MonitoredElement,
    legs: list[GridElement],
    leg_ids_with_limits: set[str],
    leg_monitoring: Trafo3wLegMonitoring,
) -> list[MonitoredElement]:
    """Copy a monitored three-winding transformer to its legs, so that only the lead leg represents it.

    The lead leg is the first leg, in leg order, that has an operational limit, or the first leg if none has one. It
    inherits all flags of the three-winding transformer. With ``SINGLE_LEG``, the other legs are hidden and not
    optimized, and are non-worsening only if the three-winding transformer is optimized and not non-worsening. With
    ``REMOVE_NON_LEAD_LEGS``, they are left out.
    """
    lead_leg = next((leg for leg in legs if leg.id in leg_ids_with_limits), None)
    if lead_leg is None:
        logger.warning(
            "No 3W trafo leg has an operational limit. Setting lead leg to 1st leg.",
            element_id=element.id,
            element_name=element.name,
            lead_leg_id=legs[0].id,
        )
        lead_leg = legs[0]
    if leg_monitoring == Trafo3wLegMonitoring.REMOVE_NON_LEAD_LEGS:
        return [element.model_copy(update=dict(lead_leg))]
    hidden = {"optimized": False, "non_worsening": element.optimized and not element.non_worsening, "displayed": False}
    return [element.model_copy(update=dict(leg) if leg == lead_leg else dict(leg) | hidden) for leg in legs]


def _replace_monitored_elements_by_legs(
    elements: list[MonitoredElement],
    legs_by_trafo3w: dict[str, list[GridElement]],
    network: Network,
    leg_monitoring: Trafo3wLegMonitoring,
) -> list[MonitoredElement]:
    """Replace monitored three-winding transformers by their legs as ``leg_monitoring`` says, dropping duplicates."""
    leg_ids_with_limits: set[str] = set()
    if leg_monitoring != Trafo3wLegMonitoring.ALL_LEGS:
        leg_ids_with_limits = set(network.get_operational_limits(attributes=[]).index.get_level_values("element_id"))
    replaced: list[MonitoredElement] = []
    for element in elements:
        legs = legs_by_trafo3w.get(element.id)
        if legs is None:
            if element.type == "THREE_WINDINGS_TRANSFORMER":
                logger.warning(
                    "Monitored 3W trafo legs kept because grid has no legs",
                    element_id=element.id,
                    element_name=element.name,
                )
            replaced.append(element)
        elif leg_monitoring == Trafo3wLegMonitoring.ALL_LEGS:
            replaced.extend(_copy_to_all_legs(element, legs))
        else:
            replaced.extend(_copy_to_lead_leg(element, legs, leg_ids_with_limits, leg_monitoring))
    return _drop_duplicate_ids(replaced)


def _copy_per_leg(items: list[SppsItemT], id_field: str, legs_by_trafo3w: dict[str, list[GridElement]]) -> list[SppsItemT]:
    """Replace every SPPS condition or action whose ``id_field`` is a three-winding transformer by one copy per leg."""
    copied: list[SppsItemT] = []
    for item in items:
        legs = legs_by_trafo3w.get(getattr(item, id_field))
        copied.extend([item] if legs is None else [item.model_copy(update={id_field: leg.id}) for leg in legs])
    return copied


def convert_three_winding_transformers_in_nminus1_definition(
    definition: Nminus1Definition,
    network: Network,
    *,
    leg_monitoring: Trafo3wLegMonitoring = Trafo3wLegMonitoring.ALL_LEGS,
) -> Nminus1Definition:
    """Convert every three-winding transformer reference in an N-1 definition to its two-winding legs.

    Contingency and monitored elements on a converted three-winding transformer are replaced by its three legs.
    SPPS conditions and actions on it are copied once per leg, regardless of the rule's ``condition_logic``, so an
    ``ANY`` rule triggers as soon as a single leg meets the condition. The conversion is idempotent.

    How the legs of a monitored three-winding transformer inherit its flags depends on ``leg_monitoring``, see
    :class:`Trafo3wLegMonitoring`. Contingencies and SPPS rules always use all legs.

    Parameters
    ----------
    definition : Nminus1Definition
        The N-1 definition, referencing three-winding transformers by their original id.
    network : pypowsybl.network.Network
        The grid **after** ``replace_3_windings_transformers_with_3_2_windings_transformers``.
    leg_monitoring : Trafo3wLegMonitoring
        How the legs of a monitored three-winding transformer are monitored.

    Returns
    -------
    Nminus1Definition
        The converted definition. A three-winding transformer whose legs are not in the grid is kept and warned about.
    """
    trafos = network.get_2_windings_transformers(attributes=["name"]).sort_index()
    legs_by_trafo3w: dict[str, list[GridElement]] = {}
    for trafo_id, trafo_name in trafos["name"].items():
        leg_id = str(trafo_id)
        trafo3w_id = get_trafo3w_id(leg_id)
        if trafo3w_id is None:
            # not a leg of a converted three-winding transformer
            continue
        legs_by_trafo3w.setdefault(trafo3w_id, []).append(
            GridElement(id=leg_id, name=trafo_name or "", type="TWO_WINDINGS_TRANSFORMER", kind="branch")
        )

    contingencies = [
        contingency.model_copy(
            update={"elements": _replace_contingency_elements_by_legs(contingency.elements, legs_by_trafo3w)}
        )
        for contingency in definition.contingencies
    ]
    monitored_elements = _replace_monitored_elements_by_legs(
        definition.monitored_elements, legs_by_trafo3w, network, leg_monitoring
    )
    spps_rules = definition.spps_rules and [
        rule.model_copy(
            update={
                "conditions": _copy_per_leg(rule.conditions, "condition_element_unique_id", legs_by_trafo3w),
                "actions": _copy_per_leg(rule.actions, "measure_element_unique_id", legs_by_trafo3w),
            }
        )
        for rule in definition.spps_rules
    ]
    converted_definition = definition.model_copy(
        update={"contingencies": contingencies, "monitored_elements": monitored_elements, "spps_rules": spps_rules}
    )
    return Nminus1Definition.model_validate(converted_definition.model_dump())


def get_nminus1_definition_element_ids(definition: Nminus1Definition) -> set[str]:
    """Collect the ids of all contingency, monitored, SPPS condition and SPPS action elements of an N-1 definition.

    Parameters
    ----------
    definition : Nminus1Definition
        The N-1 definition.

    Returns
    -------
    set[str]
        The referenced element ids.
    """
    element_ids = {element.id for contingency in definition.contingencies for element in contingency.elements}
    element_ids.update(element.id for element in definition.monitored_elements)
    for rule in definition.spps_rules or []:
        element_ids.update(condition.condition_element_unique_id for condition in rule.conditions)
        element_ids.update(action.measure_element_unique_id for action in rule.actions)
    return element_ids
