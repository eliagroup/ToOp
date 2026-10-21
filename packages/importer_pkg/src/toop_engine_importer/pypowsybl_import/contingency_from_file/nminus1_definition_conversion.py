# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Convert three-winding transformers in an N-1 definition to their two-winding legs ``<id>-Leg1``/``2``/``3``.

This is the last stage of the N-1 definition pipeline before the definition is saved next to the converted grid.
"""

from typing import TypeVar

import structlog
from pydantic import BaseModel
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.trafo3w_legs import TRAFO3W_LEG_PATTERN, get_trafo3w_id
from toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input import GridElementT
from toop_engine_interfaces.nminus1_definition import GridElement, Nminus1Definition

logger = structlog.get_logger(__name__)

SppsItemT = TypeVar("SppsItemT", bound=BaseModel)


def _replace_by_legs(
    elements: list[GridElementT], legs_by_trafo3w: dict[str, list[GridElement]], *, context: str
) -> list[GridElementT]:
    """Replace three-winding transformers by their legs, keeping order and element class and dropping duplicates."""
    converted: dict[str, GridElementT] = {}
    for element in elements:
        legs = legs_by_trafo3w.get(element.id)
        if legs is None and element.type == "THREE_WINDINGS_TRANSFORMER":
            logger.warning(
                "three_winding_transformer_legs_missing", context=context, element_id=element.id, element_name=element.name
            )
        for replacement in [element] if legs is None else [element.model_copy(update=dict(leg)) for leg in legs]:
            converted.setdefault(replacement.id, replacement)
    return list(converted.values())


def _copy_per_leg(items: list[SppsItemT], id_field: str, legs_by_trafo3w: dict[str, list[GridElement]]) -> list[SppsItemT]:
    """Replace every SPPS condition or action whose ``id_field`` is a three-winding transformer by one copy per leg."""
    copied: list[SppsItemT] = []
    for item in items:
        legs = legs_by_trafo3w.get(getattr(item, id_field))
        copied.extend([item] if legs is None else [item.model_copy(update={id_field: leg.id}) for leg in legs])
    return copied


def convert_three_winding_transformers_in_nminus1_definition(
    definition: Nminus1Definition, network: Network
) -> Nminus1Definition:
    """Convert every three-winding transformer reference in an N-1 definition to its two-winding legs.

    Contingency and monitored elements on a converted three-winding transformer are replaced by its three legs.
    SPPS conditions and actions on it are copied once per leg, regardless of the rule's ``condition_logic``, so an
    ``ANY`` rule triggers as soon as a single leg meets the condition. The conversion is idempotent.

    Parameters
    ----------
    definition : Nminus1Definition
        The N-1 definition, referencing three-winding transformers by their original id.
    network : pypowsybl.network.Network
        The grid **after** ``replace_3_windings_transformers_with_3_2_windings_transformers``.

    Returns
    -------
    Nminus1Definition
        The converted definition. A three-winding transformer whose legs are not in the grid is kept and warned about.
    """
    trafos = network.get_2_windings_transformers(attributes=["name"])
    legs = trafos[trafos.index.str.contains(TRAFO3W_LEG_PATTERN)].sort_index()
    legs_by_trafo3w: dict[str, list[GridElement]] = {}
    for leg_id, leg_name in legs["name"].items():
        legs_by_trafo3w.setdefault(get_trafo3w_id(leg_id), []).append(
            GridElement(id=leg_id, name=leg_name or "", type="TWO_WINDINGS_TRANSFORMER", kind="branch")
        )

    contingencies = [
        contingency.model_copy(
            update={"elements": _replace_by_legs(contingency.elements, legs_by_trafo3w, context="contingency")}
        )
        for contingency in definition.contingencies
    ]
    monitored_elements = _replace_by_legs(definition.monitored_elements, legs_by_trafo3w, context="monitored_element")
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
