# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Grid-file stage of the input N-1 definition: validate a Pydantic dump of ``Nminus1Definition`` against a grid."""

from pathlib import Path
from typing import TypeVar, get_args

import pandas as pd
import structlog
from fsspec import AbstractFileSystem
from pypowsybl.network import ElementType
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.branch_names import NAMED_SOURCE_FORMATS, get_branch_names
from toop_engine_importer.pypowsybl_import.contingency_from_file.helper_functions import get_all_element_names
from toop_engine_interfaces.nminus1_definition import (
    GridElement,
    GridElementType,
    Nminus1Definition,
    SppsRule,
    load_nminus1_definition_fs,
)

logger = structlog.get_logger(__name__)

# The N-1 element kind for each Powsybl element type in the grid inventory
KIND_BY_ELEMENT_TYPE: dict[GridElementType, str] = {
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


# Branch types the DC solver can outage, so a contingency with one of them is a DC single outage
DC_BRANCH_TYPES: tuple[GridElementType, ...] = ("LINE", "TWO_WINDINGS_TRANSFORMER", "TIE_LINE")


def _get_grid_element_names(network: Network) -> dict[str, str]:
    """Get the name per grid element id. Unnamed elements are left out.

    In UCTE, CGMES and hybrid grids, lines, transformers and tie lines are named as by the DC solver preprocessing,
    see :func:`~toop_engine_grid_helpers.powsybl.branch_names.get_branch_names`. All other elements are named by their
    UCTE ``elementName`` if set, else their Powsybl ``name``.
    """
    names = []
    for element_type in get_args(GridElementType):
        if element_type in DC_BRANCH_TYPES and network._source_format in NAMED_SOURCE_FORMATS:
            names.append(get_branch_names(network, element_type))
            continue
        # BUS elements are voltage levels, as in get_all_element_names
        powsybl_type = ElementType.VOLTAGE_LEVEL if element_type == "BUS" else ElementType.__members__[element_type]
        elements = network.get_elements(powsybl_type, all_attributes=True)
        element_names = elements.get("elementName", elements["name"]).fillna("")
        names.append(element_names.where(element_names != "", elements["name"]))
    names = pd.concat(names).fillna("")
    return names[names != ""].to_dict()


def _get_dc_single_outage_name(elements: list[GridElement], grid_element_names: dict[str, str]) -> str | None:
    """Get the grid name of the element a contingency outages alone in the DC solver, None for other contingencies.

    As in the DC solver's projection of contingencies, a contingency is a single outage if it outages exactly one DC
    branch, or no DC branch and exactly one injection. Isolating switches do not count.
    """
    outaged = [element for element in elements if element.type in DC_BRANCH_TYPES] or [
        element for element in elements if element.kind == "injection"
    ]
    return grid_element_names.get(outaged[0].id) if len(outaged) == 1 else None


def _align_elements_with_grid(
    elements: list[GridElementT],
    grid_element_types: dict[str, list[GridElementType]],
    grid_element_names: dict[str, str],
    contingency_id: str | None = None,
) -> list[GridElementT]:
    """Drop the elements that are not in the grid and take type, kind and name of the others from the grid.

    Parameters
    ----------
    elements : list[GridElement]
        Contingency or monitored elements. Subclasses such as MonitoredElement keep their extra fields.
    grid_element_types : dict[str, list[str]]
        The Powsybl element types per grid element id.
    grid_element_names : dict[str, str]
        The grid name per grid element id, see :func:`_get_grid_element_names`. Elements without a grid name keep
        their declared name.
    contingency_id : str, optional
        The contingency the elements belong to, None for monitored elements. Only used for logging.

    Returns
    -------
    list[GridElement]
        The elements that are in the grid, with type, kind and name corrected to the grid.
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
        name_update = {}
        if element.name != (actual_name := grid_element_names.get(element.id, element.name)):
            log.warning("nminus1_element_name_corrected", declared_name=element.name, actual_name=actual_name)
            name_update = {"name": actual_name}
        if (element.type, element.kind) == (actual_type, actual_kind):
            aligned.append(element.model_copy(update=name_update))
            continue
        log.warning(
            "nminus1_element_type_corrected",
            declared_type=element.type,
            declared_kind=element.kind,
            actual_type=actual_type,
            actual_kind=actual_kind,
        )
        aligned.append(element.model_copy(update={**name_update, "type": actual_type, "kind": actual_kind}))
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
        - element names are overwritten by the grid name where the grid has one, see :func:`_get_grid_element_names`;
        - contingencies that the DC solver outages as a single element take the grid name of that element;
        - contingencies that lose all their elements are dropped, while contingencies that were empty
          (the base case) are kept;
        - SPPS rules are dropped as a whole if their contingency was dropped or if any condition or action
          element is not in the grid.
    """
    grid_element_types: dict[str, list[GridElementType]] = (
        get_all_element_names(network).groupby("grid_model_id", sort=False)["element_type"].agg(list).to_dict()
    )
    grid_element_names = _get_grid_element_names(network)

    contingencies = []
    for contingency in definition.contingencies:
        elements = _align_elements_with_grid(contingency.elements, grid_element_types, grid_element_names, contingency.id)
        if contingency.elements and not elements:
            logger.warning(
                "empty_nminus1_contingency_dropped", contingency_id=contingency.id, contingency_name=contingency.name
            )
            continue
        update = {"elements": elements}
        if (name := _get_dc_single_outage_name(elements, grid_element_names)) and name != contingency.name:
            logger.warning(
                "nminus1_contingency_name_corrected",
                context="contingency",
                contingency_id=contingency.id,
                declared_name=contingency.name,
                actual_name=name,
            )
            update["name"] = name
        contingencies.append(contingency.model_copy(update=update))

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
            "monitored_elements": _align_elements_with_grid(
                definition.monitored_elements, grid_element_types, grid_element_names
            ),
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
