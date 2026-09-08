# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Import grouped/multi-outage contingency cases from an internal JSON format."""

import json
from pathlib import Path

import pandas as pd
import structlog
from fsspec import AbstractFileSystem
from pydantic import BaseModel, Field
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.contingency_from_file.helper_functions import get_all_element_names
from toop_engine_interfaces.nminus1_definition import (
    Action,
    Condition,
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
    SppsRule,
)
from toop_engine_interfaces.spps_parameters import (
    SppsConditionCheckType,
    SppsConditionLogic,
    SppsConditionType,
    SppsMeasureType,
    SppsSwitchActionTarget,
)

logger = structlog.get_logger(__name__)


class ContingencyFileElement(BaseModel):
    """An element referenced by a contingency file."""

    name: str = Field(alias="Name")
    rdf_id: str = Field(alias="RdfId")


class ContingencyFileCase(BaseModel):
    """A grouped contingency case from the JSON file."""

    name: str = Field(alias="Name")
    fault_case: str = Field(alias="FaultCase")
    interrupted_components: list[ContingencyFileElement] = Field(alias="InterruptedComponents")
    opened_switches: list[ContingencyFileElement] = Field(alias="OpenedSwitches")
    closed_switches: list[ContingencyFileElement] = Field(alias="ClosedSwitches")
    out_of_service: int = Field(alias="OutOfService")


def _normalise_rdf_id(rdf_id: str) -> str:
    """Remove the optional leading underscore used by CIM RDF identifiers.

    Parameters
    ----------
    rdf_id : str
        RDF identifier from the contingency file.

    Returns
    -------
    str
        The identifier without a leading underscore.
    """
    return rdf_id.removeprefix("_")


def _resolve_element(
    element: ContingencyFileElement,
    all_elements: pd.DataFrame,
    *,
    expected_type: str | None = None,
    contingency_id: str | None = None,
    contingency_name: str | None = None,
) -> GridElement:
    """Resolve a file element against the Powsybl network inventory.

    Parameters
    ----------
    element : ContingencyFileElement
        Element reference read from the contingency file.
    all_elements : pandas.DataFrame
        Network element inventory containing ``grid_model_id``,
        ``grid_model_name``, and ``element_type`` columns.
    expected_type : str, optional
        Restrict the match to a specific Powsybl element type.
    contingency_id : str, optional
        Contingency identifier used to provide context in errors.
    contingency_name : str, optional
        Contingency name used to provide context in errors.

    Returns
    -------
    GridElement
        The resolved shared grid element.

    Raises
    ------
    ValueError
        If the element cannot be resolved to exactly one network element.
    """
    attempts: list[tuple[str, str]] = [(element.rdf_id, "grid_model_id")]
    normalised_rdf_id = _normalise_rdf_id(element.rdf_id)
    if normalised_rdf_id != element.rdf_id:
        attempts.append((normalised_rdf_id, "grid_model_id"))
    attempts.append((element.name, "grid_model_name"))
    for attempt, column in attempts:
        candidates = all_elements[all_elements[column] == attempt]
        if expected_type is not None:
            candidates = candidates[candidates.element_type == expected_type]
        if len(candidates) == 1:
            break
        if len(candidates) > 1:
            candidate_ids = candidates.grid_model_id.astype(str).tolist()
            logger.error(
                "ambiguous_contingency_element",
                contingency_id=contingency_id,
                contingency_name=contingency_name,
                source_reference=element.rdf_id,
                source_name=element.name,
                expected_type=expected_type,
                resolution_attempts=[value for value, _ in attempts],
                matching_candidates=candidate_ids,
            )
            raise ValueError(
                f"Ambiguous contingency element {element.rdf_id!r} ({element.name!r}) in "
                f"{contingency_id!r} ({contingency_name!r}); matches: {candidate_ids}"
            )
    else:
        logger.error(
            "unknown_contingency_element",
            contingency_id=contingency_id,
            contingency_name=contingency_name,
            source_reference=element.rdf_id,
            source_name=element.name,
            expected_type=expected_type,
            resolution_attempts=[value for value, _ in attempts],
        )
        raise ValueError(
            f"Could not resolve contingency element {element.rdf_id!r} ({element.name!r}) in "
            f"{contingency_id!r} ({contingency_name!r}); attempts: {[value for value, _ in attempts]}; "
            f"expected_type={expected_type!r}"
        )

    row = candidates.iloc[0]
    element_type = row.element_type
    kind = "branch"
    if element_type in {"GENERATOR", "LOAD", "BOUNDARY_LINE", "SHUNT_COMPENSATOR"}:
        kind = "injection"
    elif element_type in {"BUS", "BUSBAR_SECTION"}:
        kind = "bus"
    return GridElement(
        id=str(row.grid_model_id), name=str(row.grid_model_name or element.name), type=element_type, kind=kind
    )


def _resolve_converted_transformer_legs(
    transformer_id: str,
    all_elements: pd.DataFrame,
    *,
    contingency_id: str,
    contingency_name: str,
) -> list[GridElement]:
    """Resolve the three two-winding legs created from a transformer.

    Parameters
    ----------
    transformer_id : str
        Original three-winding transformer identifier.
    all_elements : pandas.DataFrame
        Network element inventory.
    contingency_id : str
        Contingency identifier used to provide context in errors.
    contingency_name : str
        Contingency name used to provide context in errors.

    Returns
    -------
    list[GridElement]
        The three resolved transformer legs.

    Raises
    ------
    ValueError
        If one or more converted transformer legs cannot be resolved.
    """
    leg_ids = [f"{transformer_id}-Leg{leg_number}" for leg_number in range(1, 4)]
    return [
        _resolve_element(
            ContingencyFileElement(Name="", RdfId=leg_id),
            all_elements,
            expected_type="TWO_WINDINGS_TRANSFORMER",
            contingency_id=contingency_id,
            contingency_name=contingency_name,
        )
        for leg_id in leg_ids
    ]


def _resolve_interrupted_elements(
    element: ContingencyFileElement,
    all_elements: pd.DataFrame,
    *,
    contingency_id: str,
    contingency_name: str,
) -> list[GridElement]:
    """Resolve an interrupted component, expanding a converted three-winding transformer.

    Parameters
    ----------
    element : ContingencyFileElement
        Element reference read from the contingency file.
    all_elements : pandas.DataFrame
        Network element inventory.
    contingency_id : str
        Contingency identifier used to provide context in errors.
    contingency_name : str
        Contingency name used to provide context in errors.

    Returns
    -------
    list[GridElement]
        Resolved element, or the three converted transformer legs.

    Raises
    ------
    ValueError
        If the element or all three converted transformer legs cannot be resolved.
    """
    transformer_id = _normalise_rdf_id(element.rdf_id)
    leg_ids = [f"{transformer_id}-Leg{leg_number}" for leg_number in range(1, 4)]
    has_all_legs = all(
        len(
            all_elements[
                (all_elements["grid_model_id"] == leg_id) & (all_elements.element_type == "TWO_WINDINGS_TRANSFORMER")
            ]
        )
        == 1
        for leg_id in leg_ids
    )
    has_original = bool(
        all_elements[all_elements["grid_model_id"].isin({element.rdf_id, transformer_id})].shape[0]
        or all_elements[all_elements["grid_model_name"] == element.name].shape[0]
    )
    if has_all_legs and not has_original:
        return _resolve_converted_transformer_legs(
            transformer_id,
            all_elements,
            contingency_id=contingency_id,
            contingency_name=contingency_name,
        )
    try:
        resolved = _resolve_element(
            element,
            all_elements,
            contingency_id=contingency_id,
            contingency_name=contingency_name,
        )
    except ValueError as original_error:
        if not has_all_legs:
            raise original_error
        return _resolve_converted_transformer_legs(
            transformer_id,
            all_elements,
            contingency_id=contingency_id,
            contingency_name=contingency_name,
        )
    if resolved.type != "THREE_WINDINGS_TRANSFORMER":
        return [resolved]
    return _resolve_converted_transformer_legs(
        transformer_id,
        all_elements,
        contingency_id=contingency_id,
        contingency_name=contingency_name,
    )


def load_nminus1_definition_from_file(
    network: Network,
    file_path: str | Path,
    filesystem: AbstractFileSystem,
    monitored_elements: list[MonitoredElement],
    base_case: Contingency | None = None,
) -> Nminus1Definition:
    """Load grouped contingencies and SPPS rules from a Powsybl JSON file.

    Parameters
    ----------
    network : pypowsybl.network.Network
        Network used to resolve RDF identifiers and element types.
    file_path : str or pathlib.Path
        Path to the JSON contingency file.
    filesystem : fsspec.AbstractFileSystem
        Filesystem from which to read ``file_path``.
    monitored_elements : list[MonitoredElement]
        Monitored elements to retain in the resulting definition.
    base_case : Contingency, optional
        Base-case contingency to prepend. If omitted, a default ``BASECASE``
        contingency is created.

    Returns
    -------
    Nminus1Definition
        Shared definition containing grouped outage elements and SPPS rules.

    Notes
    -----
    ``InterruptedComponents`` and ``OpenedSwitches`` are combined in each
    contingency. ``ClosedSwitches`` become SPPS actions guarded by the
    interrupted components being de-energized. A three-winding transformer
    reference in ``InterruptedComponents`` is expanded to its three converted
    ``-Leg1``/``-Leg2``/``-Leg3`` two-winding transformers. ``OutOfService`` is
    currently read for schema compatibility and intentionally ignored.

    Raises
    ------
    ValueError
        If an element reference cannot be resolved uniquely in ``network``.
    json.JSONDecodeError
        If ``file_path`` does not contain valid JSON.
    """
    with filesystem.open(str(file_path), "r") as file:
        raw_cases = json.load(file)
    cases = [ContingencyFileCase.model_validate(case) for case in raw_cases.values()]
    all_elements = get_all_element_names(network)

    contingencies = [base_case or Contingency(id="BASECASE", name="BASECASE", elements=[])]
    spps_rules: list[SppsRule] = []
    for case in cases:
        interrupted = [
            resolved
            for element in case.interrupted_components
            for resolved in _resolve_interrupted_elements(
                element, all_elements, contingency_id=case.name, contingency_name=case.fault_case
            )
        ]
        opened_switches = [
            _resolve_element(
                element,
                all_elements,
                expected_type="SWITCH",
                contingency_id=case.name,
                contingency_name=case.fault_case,
            )
            for element in case.opened_switches
        ]
        closed_switches = [
            _resolve_element(
                element,
                all_elements,
                expected_type="SWITCH",
                contingency_id=case.name,
                contingency_name=case.fault_case,
            )
            for element in case.closed_switches
        ]
        if case.name == "BASECASE":
            logger.error(
                "reserved_complex_contingency_id",
                contingency_id=case.name,
                contingency_name=case.fault_case,
                source_reference=case.name,
                resolution_attempts=[],
            )
            raise ValueError("BASECASE is reserved and cannot be supplied as a complex contingency")
        if not interrupted and not opened_switches:
            logger.error(
                "empty_complex_contingency",
                contingency_id=case.name,
                contingency_name=case.fault_case,
                source_reference=case.name,
                resolution_attempts=[],
            )
            raise ValueError(f"Contingency {case.name!r} ({case.fault_case!r}) has no outage elements")
        if case.name in {contingency.id for contingency in contingencies}:
            logger.warning("duplicate_contingency_id", contingency_id=case.name, contingency_name=case.fault_case)
            continue
        contingencies.append(Contingency(id=case.name, name=case.fault_case, elements=interrupted + opened_switches))

        if closed_switches:
            spps_rules.append(
                SppsRule(
                    scheme_name=case.name,
                    condition_logic=SppsConditionLogic.ALL,
                    conditions=[
                        Condition(
                            condition_type=SppsConditionType.STATE,
                            condition_check_type=SppsConditionCheckType.DE_ENERGIZED,
                            condition_element_unique_id=element.id,
                        )
                        for element in interrupted
                    ]
                    + [
                        Condition(
                            condition_type=SppsConditionType.SWITCHING_STATE,
                            condition_check_type=SppsConditionCheckType.EQ,
                            condition_limit_value=SppsSwitchActionTarget.OPEN,
                            condition_element_unique_id=element.id,
                        )
                        for element in opened_switches
                    ],
                    actions=[
                        Action(
                            measure_element_unique_id=element.id,
                            measure_type=SppsMeasureType.SWITCHING_STATE,
                            measure_value=SppsSwitchActionTarget.CLOSED,
                        )
                        for element in closed_switches
                    ],
                )
            )

    definition = Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=contingencies,
        spps_rules=spps_rules or None,
        id_type="powsybl",
        source_schema="complex",
    )
    return definition
