# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Validate an input N-1 definition against a Powsybl grid.

The input N-1 definition is a Pydantic JSON dump of
:class:`~toop_engine_interfaces.nminus1_definition.Nminus1Definition` with element identifiers that were resolved
by the producer. This module implements the grid-file stage of turning it into a grid-validated definition:
elements that do not exist in the grid are dropped and element types are aligned with the grid. Area settings and
the three-winding transformer conversion are later stages and are not applied here.
"""

from pathlib import Path
from typing import TypeVar

import structlog
from fsspec import AbstractFileSystem
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.contingency_from_file.helper_functions import get_all_element_names
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    Nminus1Definition,
    SppsRule,
    load_nminus1_definition_fs,
)

logger = structlog.get_logger(__name__)

# The N-1 element kind for each Powsybl element type in the grid inventory
KIND_BY_ELEMENT_TYPE: dict[str, str] = {
    "LINE": "branch",
    "TWO_WINDINGS_TRANSFORMER": "branch",
    "THREE_WINDINGS_TRANSFORMER": "branch",
    "HVDC_LINE": "branch",
    "TIE_LINE": "branch",
    "GENERATOR": "injection",
    "LOAD": "injection",
    "BOUNDARY_LINE": "injection",
    "SHUNT_COMPENSATOR": "injection",
    "BUS": "bus",
    "BUSBAR_SECTION": "bus",
    "SWITCH": "switch",
}

GridElementT = TypeVar("GridElementT", bound=GridElement)


def _get_element_types_by_id(network: Network) -> dict[str, list[str]]:
    """Map every element id in the network to the Powsybl element types carrying that id.

    Parameters
    ----------
    network : pypowsybl.network.Network
        The network to inventory.

    Returns
    -------
    dict[str, list[str]]
        The element types per grid model id, in inventory order.
    """
    all_elements = get_all_element_names(network)
    element_types_by_id: dict[str, list[str]] = {}
    for grid_model_id, element_type in zip(all_elements.grid_model_id, all_elements.element_type, strict=True):
        element_types_by_id.setdefault(str(grid_model_id), []).append(str(element_type))
    return element_types_by_id


def _validate_element(
    element: GridElementT,
    element_types_by_id: dict[str, list[str]],
    *,
    context: str,
    contingency_id: str | None = None,
) -> GridElementT | None:
    """Check an element against the grid inventory, correcting its type and kind to the grid.

    Parameters
    ----------
    element : GridElement
        The element from the input N-1 definition. Subclasses such as MonitoredElement keep their extra fields.
    element_types_by_id : dict[str, list[str]]
        The grid inventory from :func:`_get_element_types_by_id`.
    context : str
        Where the element is used, e.g. ``"contingency"`` or ``"monitored_element"``, for logging.
    contingency_id : str, optional
        The contingency the element belongs to, for logging.

    Returns
    -------
    GridElement | None
        The element with type and kind taken from the grid, or None if the element is not in the grid.
    """
    element_types = element_types_by_id.get(element.id)
    if element_types is None:
        logger.warning(
            "unknown_nminus1_element_dropped",
            context=context,
            contingency_id=contingency_id,
            element_id=element.id,
            element_name=element.name,
            declared_type=element.type,
        )
        return None

    actual_type = element.type if element.type in element_types else element_types[0]
    actual_kind = KIND_BY_ELEMENT_TYPE.get(actual_type)
    if actual_kind is None:
        logger.warning(
            "unsupported_nminus1_element_type_dropped",
            context=context,
            contingency_id=contingency_id,
            element_id=element.id,
            element_name=element.name,
            actual_type=actual_type,
        )
        return None

    if element.type == actual_type and element.kind == actual_kind:
        return element
    logger.warning(
        "nminus1_element_type_corrected",
        context=context,
        contingency_id=contingency_id,
        element_id=element.id,
        element_name=element.name,
        declared_type=element.type,
        declared_kind=element.kind,
        actual_type=actual_type,
        actual_kind=actual_kind,
    )
    return element.model_copy(update={"type": actual_type, "kind": actual_kind})


def _filter_contingency(contingency: Contingency, element_types_by_id: dict[str, list[str]]) -> Contingency | None:
    """Filter the outaged elements of a contingency to the grid.

    Parameters
    ----------
    contingency : Contingency
        The contingency from the input N-1 definition.
    element_types_by_id : dict[str, list[str]]
        The grid inventory from :func:`_get_element_types_by_id`.

    Returns
    -------
    Contingency | None
        The contingency with only grid elements, or None if all of its elements were dropped. Contingencies that
        were empty in the input definition (the base case) are returned unchanged.
    """
    if not contingency.elements:
        return contingency
    elements = [
        validated
        for element in contingency.elements
        if (
            validated := _validate_element(
                element, element_types_by_id, context="contingency", contingency_id=contingency.id
            )
        )
        is not None
    ]
    if not elements:
        logger.warning("empty_nminus1_contingency_dropped", contingency_id=contingency.id, contingency_name=contingency.name)
        return None
    return contingency.model_copy(update={"elements": elements})


def _filter_spps_rule(
    rule: SppsRule, contingency_ids: set[str], element_types_by_id: dict[str, list[str]]
) -> SppsRule | None:
    """Keep an SPPS rule only if its contingency and every element it references exist.

    Parameters
    ----------
    rule : SppsRule
        The SPPS rule from the input N-1 definition.
    contingency_ids : set[str]
        The ids of the contingencies that remain after filtering.
    element_types_by_id : dict[str, list[str]]
        The grid inventory from :func:`_get_element_types_by_id`.

    Returns
    -------
    SppsRule | None
        The unchanged rule, or None if it has to be dropped. Rules are never partially pruned, because removing a
        condition would change when the rule fires.
    """
    if rule.scheme_name not in contingency_ids:
        logger.warning("spps_rule_dropped", scheme_name=rule.scheme_name, reason="contingency_dropped")
        return None
    referenced_ids = [condition.condition_element_unique_id for condition in rule.conditions] + [
        action.measure_element_unique_id for action in rule.actions
    ]
    missing_ids = [element_id for element_id in referenced_ids if element_id not in element_types_by_id]
    if missing_ids:
        logger.warning("spps_rule_dropped", scheme_name=rule.scheme_name, reason="unknown_elements", missing_ids=missing_ids)
        return None
    return rule


def filter_nminus1_definition_to_network(definition: Nminus1Definition, network: Network) -> Nminus1Definition:
    """Validate a input N-1 definition against the grid, dropping what the grid does not contain.

    This is the grid-file stage of the input N-1 pipeline. The network must be the grid **before** the
    three-winding transformer conversion
    (``pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers``), because three-winding
    transformers are referenced by their original id and converted in a later stage.

    Parameters
    ----------
    definition : Nminus1Definition
        The input N-1 definition with Powsybl element ids.
    network : pypowsybl.network.Network
        The unconverted grid to validate against.

    Returns
    -------
    Nminus1Definition
        The definition restricted to the grid. Every change is logged as a warning:

        - elements whose id is not in the grid are dropped, from contingencies and monitored elements;
        - elements whose type or kind differ from the grid are corrected to the grid's values;
        - contingencies that lose all their elements are dropped, while contingencies that were empty
          (the base case) are kept;
        - SPPS rules are dropped as a whole if their contingency was dropped or if any condition or action
          element is not in the grid.
    """
    element_types_by_id = _get_element_types_by_id(network)

    contingencies = [
        filtered
        for contingency in definition.contingencies
        if (filtered := _filter_contingency(contingency, element_types_by_id)) is not None
    ]
    monitored_elements = [
        validated
        for element in definition.monitored_elements
        if (validated := _validate_element(element, element_types_by_id, context="monitored_element")) is not None
    ]
    spps_rules = None
    if definition.spps_rules is not None:
        contingency_ids = {contingency.id for contingency in contingencies}
        spps_rules = [
            filtered
            for rule in definition.spps_rules
            if (filtered := _filter_spps_rule(rule, contingency_ids, element_types_by_id)) is not None
        ] or None

    filtered_definition = definition.model_copy(
        update={"contingencies": contingencies, "monitored_elements": monitored_elements, "spps_rules": spps_rules}
    )
    return Nminus1Definition.model_validate(filtered_definition.model_dump())


def load_nminus1_definition_for_network(
    network: Network,
    file_path: str | Path,
    filesystem: AbstractFileSystem,
) -> Nminus1Definition:
    """Load an input N-1 definition JSON and validate it against the grid.

    Parameters
    ----------
    network : pypowsybl.network.Network
        The unconverted grid to validate against, see :func:`filter_nminus1_definition_to_network`.
    file_path : str or pathlib.Path
        Path to the Pydantic JSON of an `Nminus1Definition`.
    filesystem : fsspec.AbstractFileSystem
        Filesystem from which to read ``file_path``.

    Returns
    -------
    Nminus1Definition
        The grid-validated N-1 definition.
    """
    definition = load_nminus1_definition_fs(filesystem=filesystem, file_path=file_path)
    return filter_nminus1_definition_to_network(definition, network)
