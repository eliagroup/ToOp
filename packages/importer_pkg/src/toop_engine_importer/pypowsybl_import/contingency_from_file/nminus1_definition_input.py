# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Grid-file stage of the input N-1 definition: validate a Pydantic dump of ``Nminus1Definition`` against a grid."""

from pathlib import Path
from typing import TypeVar

import structlog
from fsspec import AbstractFileSystem
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.contingency_from_file.helper_functions import get_all_element_names
from toop_engine_interfaces.nminus1_definition import GridElement, Nminus1Definition, SppsRule, load_nminus1_definition_fs

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
    "BUS_BREAKER_BUS": "bus",
    "SWITCH": "switch",
}

GridElementT = TypeVar("GridElementT", bound=GridElement)


def _align_elements_with_grid(
    elements: list[GridElementT], grid_element_types: dict[str, list[str]], contingency_id: str | None = None
) -> list[GridElementT]:
    """Drop the elements that are not in the grid and take type and kind of the others from the grid.

    Parameters
    ----------
    elements : list[GridElement]
        Contingency or monitored elements. Subclasses such as MonitoredElement keep their extra fields.
    grid_element_types : dict[str, list[str]]
        The Powsybl element types per grid element id.
    contingency_id : str, optional
        The contingency the elements belong to, None for monitored elements. Only used for logging.

    Returns
    -------
    list[GridElement]
        The elements that are in the grid, with type and kind corrected to the grid.
    """
    aligned = []
    for element in elements:
        log = logger.bind(
            context="monitored_element" if contingency_id is None else "contingency",
            contingency_id=contingency_id,
            element_id=element.id,
            element_name=element.name,
        )
        element_types = grid_element_types.get(element.id)
        if element_types is None:
            log.warning("unknown_nminus1_element_dropped", declared_type=element.type)
            continue
        actual_type = element.type if element.type in element_types else element_types[0]
        actual_kind = KIND_BY_ELEMENT_TYPE[actual_type]
        if (element.type, element.kind) == (actual_type, actual_kind):
            aligned.append(element)
            continue
        log.warning(
            "nminus1_element_type_corrected",
            declared_type=element.type,
            declared_kind=element.kind,
            actual_type=actual_type,
            actual_kind=actual_kind,
        )
        aligned.append(element.model_copy(update={"type": actual_type, "kind": actual_kind}))
    return aligned


def _spps_rule_is_resolvable(rule: SppsRule, contingency_ids: set[str], grid_element_ids: set[str]) -> bool:
    """Check that the contingency of an SPPS rule and every element it references exist, logging dropped rules.

    Rules are never partially pruned, because removing a condition would change when the rule fires.
    """
    if rule.scheme_name not in contingency_ids:
        logger.warning("spps_rule_dropped", scheme_name=rule.scheme_name, reason="contingency_dropped")
        return False
    referenced_ids = [condition.condition_element_unique_id for condition in rule.conditions] + [
        action.measure_element_unique_id for action in rule.actions
    ]
    if missing_ids := [element_id for element_id in referenced_ids if element_id not in grid_element_ids]:
        logger.warning("spps_rule_dropped", scheme_name=rule.scheme_name, reason="unknown_elements", missing_ids=missing_ids)
        return False
    return True


def filter_nminus1_definition_to_network(definition: Nminus1Definition, network: Network) -> Nminus1Definition:
    """Validate an input N-1 definition against the grid, dropping what the grid does not contain.

    The network must be the grid **before** the three-winding transformer conversion, because three-winding
    transformers are referenced by their original id.

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
    grid_element_types: dict[str, list[str]] = (
        get_all_element_names(network).groupby("grid_model_id", sort=False)["element_type"].agg(list).to_dict()
    )
    # Configured buses of bus-breaker voltage levels are identifiables; computed buses of node-breaker levels are not.
    identifiables = network.get_identifiables()
    for bus_id in identifiables.index[identifiables["type"] == "BUS"]:
        grid_element_types.setdefault(bus_id, []).append("BUS_BREAKER_BUS")

    contingencies = []
    for contingency in definition.contingencies:
        elements = _align_elements_with_grid(contingency.elements, grid_element_types, contingency.id)
        if contingency.elements and not elements:
            logger.warning(
                "empty_nminus1_contingency_dropped", contingency_id=contingency.id, contingency_name=contingency.name
            )
        else:
            contingencies.append(contingency.model_copy(update={"elements": elements}))

    spps_rules = None
    if definition.spps_rules is not None:
        contingency_ids = {contingency.id for contingency in contingencies}
        spps_rules = [
            rule
            for rule in definition.spps_rules
            if _spps_rule_is_resolvable(rule, contingency_ids, set(grid_element_types))
        ] or None

    filtered_definition = definition.model_copy(
        update={
            "contingencies": contingencies,
            "monitored_elements": _align_elements_with_grid(definition.monitored_elements, grid_element_types),
            "spps_rules": spps_rules,
        }
    )
    # Re-validate, which model_copy skips, so the SPPS integrity check runs on the result
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
