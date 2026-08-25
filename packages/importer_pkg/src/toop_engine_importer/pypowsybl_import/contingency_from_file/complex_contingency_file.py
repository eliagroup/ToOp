"""Import grouped contingency cases from the Powsybl JSON format."""

import json
from pathlib import Path

import pandas as pd
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

    Returns
    -------
    GridElement
        The resolved shared grid element.

    Raises
    ------
    ValueError
        If the element cannot be resolved to exactly one network element.
    """
    candidates = all_elements[all_elements.grid_model_id.isin([element.rdf_id, _normalise_rdf_id(element.rdf_id)])]
    if candidates.empty:
        candidates = all_elements[all_elements.grid_model_name == element.name]
    if expected_type is not None:
        candidates = candidates[candidates.element_type == expected_type]
    if len(candidates) != 1:
        raise ValueError(f"Could not uniquely resolve contingency element {element.rdf_id!r} ({element.name!r})")

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


def _resolve_interrupted_elements(element: ContingencyFileElement, all_elements: pd.DataFrame) -> list[GridElement]:
    """Resolve an interrupted component, expanding a converted three-winding transformer.

    Parameters
    ----------
    element : ContingencyFileElement
        Element reference read from the contingency file.
    all_elements : pandas.DataFrame
        Network element inventory.

    Returns
    -------
    list[GridElement]
        Resolved element, or the three converted transformer legs.

    Raises
    ------
    ValueError
        If the element or all three converted transformer legs cannot be resolved.
    """
    try:
        return [_resolve_element(element, all_elements)]
    except ValueError as error:
        transformer_id = _normalise_rdf_id(element.rdf_id)
        leg_ids = [f"{transformer_id}-Leg{leg_number}" for leg_number in range(1, 4)]
        leg_rows = all_elements[all_elements.grid_model_id.isin(leg_ids)]
        if len(leg_rows) != len(leg_ids):
            raise error
        return [
            _resolve_element(
                ContingencyFileElement(Name=str(row.grid_model_name), RdfId=str(row.grid_model_id)),
                all_elements,
                expected_type="TWO_WINDINGS_TRANSFORMER",
            )
            for _, row in leg_rows.set_index("grid_model_id").loc[leg_ids].reset_index().iterrows()
        ]


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
            for resolved in _resolve_interrupted_elements(element, all_elements)
        ]
        opened_switches = [
            _resolve_element(element, all_elements, expected_type="SWITCH") for element in case.opened_switches
        ]
        closed_switches = [
            _resolve_element(element, all_elements, expected_type="SWITCH") for element in case.closed_switches
        ]
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

    return Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=contingencies,
        spps_rules=spps_rules or None,
        id_type="powsybl",
    )
