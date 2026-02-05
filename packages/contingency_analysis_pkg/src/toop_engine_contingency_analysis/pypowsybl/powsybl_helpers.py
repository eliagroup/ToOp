# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions to translate the N-1 definition into a usable format for Powsybl.

This includes translating contingencies, monitored elements and collecting
the necessary data from the network, so this only has to happen once.
"""

from copy import deepcopy
from typing import get_args

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
import pypowsybl
from beartype.typing import Literal, Optional
from pydantic import BaseModel
from pypowsybl._pypowsybl import PostContingencyResult, PreContingencyResult
from pypowsybl.network import Network
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_result_helpers import get_failed_branch_results, get_failed_node_results
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    BranchSide,
    ConvergedSchema,
    ConvergenceStatus,
    LoadflowResultTable,
    NodeResultSchema,
    RegulatingElementResultSchema,
    RegulatingElementType,
    VADiffResultSchema,
)
from toop_engine_interfaces.nminus1_definition import (
    POWSYBL_SUPPORTED_ID_TYPES,
    Contingency,
    GridElement,
    LoadflowParameters,
    Nminus1Definition,
)
from typing_extensions import TypedDict

POWSYBL_CONVERGENCE_MAP = {
    pypowsybl.loadflow.ComponentStatus.CONVERGED.value: ConvergenceStatus.CONVERGED.value,
    pypowsybl.loadflow.ComponentStatus.FAILED.value: ConvergenceStatus.FAILED.value,
    pypowsybl.loadflow.ComponentStatus.NO_CALCULATION.value: ConvergenceStatus.NO_CALCULATION.value,
    pypowsybl.loadflow.ComponentStatus.MAX_ITERATION_REACHED.value: ConvergenceStatus.MAX_ITERATION_REACHED.value,
}


class PowsyblContingency(BaseModel):
    """A Powsybl contingency.

    This is a simplified version of the PandapowerContingency that is used in Powsybl.
    It contains only the necessary information to run an N-1 analysis in Powsybl.
    """

    id: str
    """The unique id of the contingency."""

    name: str = ""
    """The name of the contingency."""

    elements: list[str]
    """The list of outaged element ids."""

    def is_basecase(self) -> bool:
        """Check if the contingency is a basecase.

        A basecase contingency has no outaged elements.
        """
        return len(self.elements) == 0


class PowsyblMonitoredElements(TypedDict):
    """A dictionary to hold the monitored element ids for the N-1 analysis.

    This is used to store the monitored elements in a format that can be used in Powsybl.
    """

    branches: list[str]
    trafo3w: list[str]
    switches: list[str]
    voltage_levels: list[str]
    buses: list[str]


class PowsyblNMinus1Definition(BaseModel):
    """A Powsybl N-1 definition.

    This is a simplified version of the NMinus1Definition that is used in Powsybl.
    It contains only the necessary information to run an N-1 analysis in Powsybl.
    """

    model_config = {"arbitrary_types_allowed": True}

    contingencies: list[PowsyblContingency]
    """The outages to be considered. Maps contingency id to outaged element ids."""

    monitored_elements: PowsyblMonitoredElements
    """The list of branches with two sides, to be monitored during the N-1 analysis."""

    missing_elements: list[GridElement] = []
    """A list of monitored elements that are not present in the network."""

    missing_contingencies: list[Contingency] = []
    """A list of contingencies whose elements are (partially) not present in the network."""

    branch_limits: pd.DataFrame
    """The branch limits to be used during the N-1 analysis. If None, the default limits will be used."""

    blank_va_diff: pd.DataFrame
    """The buses to be used during the N-1 analysis. This is used to determine the voltage levels of the monitored buses.
    Could be a busbar section or a bus_breaker_buse depending on the model type."""

    bus_map: pd.DataFrame
    """A mapping from busbar sections and bus breaker buses to bus breaker buses, electrical buses and voltage_levels.
      This help to always get the correct buses, even if the model type changes."""

    element_name_mapping: dict[str, str]
    """A mapping from element ids to their names. This is used to convert the element ids to their names in the results."""

    contingency_name_mapping: dict[str, str]
    """A mapping from contingency ids to their names.
    This is used to convert the contingency ids to their names in the results."""

    voltage_levels: pd.DataFrame
    """The voltage levels of the buses. This is used to determine voltage limits."""

    distributed_slack: bool = True
    """Whether to distribute the slack across the generators in the grid. Only relevant for powsybl grids."""

    contingency_propagation: bool = False
    """Whether to enable powsybl's contingency propagation in the N-1 analysis.

    https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/en/latest/security/parameters.html
    Security Analysis will determine by topological search the switches with type circuit breakers
    (i.e. capable of opening fault currents) that must be opened to isolate the fault. Depending on the network structure,
    this could lead to more equipments to be simulated as tripped, because disconnectors and load break switches
    (i.e., not capable of opening fault currents) are not considered.
    """

    def __getitem__(self, key: str | int | slice) -> "PowsyblNMinus1Definition":
        """Get a subset of the nminus1definition based on the contingencies.

        If a string is given, the contingency id must be in the contingencies list.
        If an integer or slice is given, the case id will be indexed by the integer or slice.
        """
        if isinstance(key, str):
            contingency_ids = [contingency.id for contingency in self.contingencies]
            if key not in contingency_ids:
                raise KeyError(f"Contingency id {key} not in contingencies.")
            index = contingency_ids.index(key)
            index = slice(index, index + 1)
        elif isinstance(key, int):
            index = slice(key, key + 1)
        elif isinstance(key, slice):
            index = key
        else:
            raise TypeError("Key must be a string, int or slice.")

        updated_definition = self.model_copy(
            update={
                "contingencies": self.contingencies[index],
            }
        )
        # pylint: disable=unsubscriptable-object
        return PowsyblNMinus1Definition.model_validate(updated_definition)


def translate_contingency_to_powsybl(
    contingencies: list[Contingency], identifiables: pd.Index
) -> tuple[list[PowsyblContingency], list[Contingency]]:
    """Translate the contingencies to a format that can be used in Powsybl.

    Parameters
    ----------
    contingencies : list[Contingency]
        The list of contingencies to translate.
    identifiables : pd.DataFrame
        A dataframe containing the identifiables of the network.
        This is used to check if the elements are present in the network.

    Returns
    -------
    pow_contingency: list[PowsyblContingency]
        A list of PowsyblContingency objects, each containing the id, name and elements.
    missing_contingency: list[Contingency]
        A list of all contingencies that are not fully present in the network.
    """
    pow_contingencies = []
    missing_contingencies = []
    for contingency in contingencies:
        outaged_elements = []
        for element in contingency.elements:
            if element.id not in identifiables:
                missing_contingencies.append(contingency)
                break
            outaged_elements.append(element.id)
        else:
            pp_contingency = PowsyblContingency(
                id=contingency.id,
                name=contingency.name or "",
                elements=outaged_elements,
            )
            pow_contingencies.append(pp_contingency)

    return pow_contingencies, missing_contingencies


def translate_monitored_elements_to_powsybl(
    nminus1_definition: Nminus1Definition, branches: pd.DataFrame, buses: pd.DataFrame, switches: pd.DataFrame
) -> tuple[PowsyblMonitoredElements, dict[str, str], list[GridElement]]:
    """Translate the monitored elements to a format that can be used in Powsybl.

    Also adds busses that are not monitored per se, but are needed for the voltage angle difference calculation.

    Parameters
    ----------
    nminus1_definition: Nminus1Definition
        The original Nminus1Definition containing the monitored elements and outages.
    branches : pd.DataFrame
        The dataframe containing the branches of the network and their voltage_id including 3w-trafos.
    buses : pd.DataFrame
        The dataframe containing the buses of the network and their voltage_id.
        These include busbar sections and bus_breaker buses.
    switches : pd.DataFrame
        The dataframe containing the switches of the network and their voltage_id.


    Returns
    -------
    monitored_elements: PowsyblMonitoredElements
        A dictionary containing the monitored elements in a format that can be used in Powsybl.
    element_name_mapping: dict[str, str]
        A mapping from element ids to their names. This is used to convert the element ids to their names in the results.
    missing_elements: list[GridElement]
        A list of monitored elements that are not present in the network.
    """
    monitored_elements = nminus1_definition.monitored_elements
    all_monitored_branches = [element.id for element in monitored_elements if element.kind == "branch"]
    missing_branches = set(all_monitored_branches) - set(branches.index)
    monitored_branch_df = branches.loc[
        [branch_id for branch_id in all_monitored_branches if branch_id not in missing_branches]
    ]
    monitored_branches = monitored_branch_df.loc[
        monitored_branch_df.type.isin(["LINE", "TWO_WINDINGS_TRANSFORMER", "TIE_LINE"])
    ].index.tolist()

    monitored_trafo3w = monitored_branch_df.loc[monitored_branch_df.type == "THREE_WINDINGS_TRANSFORMER"].index.tolist()

    all_monitored_buses = [element.id for element in monitored_elements if element.kind == "bus"]
    missing_buses = set(all_monitored_buses) - set(buses.index)
    monitored_buses = [bus_id for bus_id in all_monitored_buses if bus_id not in missing_buses]

    all_monitored_switches = [element.id for element in monitored_elements if element.kind == "switch"]
    missing_switches = set(all_monitored_switches) - set(switches.index)
    monitored_switches = [switch_id for switch_id in all_monitored_switches if switch_id not in missing_switches]

    # The voltagelevels of outaged branches are relevant for the voltage angle difference calculation.
    all_outaged_branch_ids = [
        elem.id for contingency in nminus1_definition.contingencies for elem in contingency.elements if elem.kind == "branch"
    ]
    missing_outage_branches = set(all_outaged_branch_ids) - set(branches.index)
    outaged_branch_df = branches.loc[
        [branch_id for branch_id in all_outaged_branch_ids if branch_id not in missing_outage_branches]
    ]
    outaged_branch_ids = outaged_branch_df.loc[
        outaged_branch_df.type.isin(["LINE", "TWO_WINDINGS_TRANSFORMER", "TIE_LINE"])
    ].index.tolist()
    monitored_voltage_levels = set(
        buses.loc[monitored_buses, "voltage_level_id"].unique().tolist()
        + switches.loc[monitored_switches, "voltage_level_id"].unique().tolist()
        + branches.loc[monitored_branches + outaged_branch_ids, "voltage_level1_id"].unique().tolist()
        + branches.loc[monitored_branches + outaged_branch_ids, "voltage_level2_id"].unique().tolist()
    )
    powsybl_monitored_elements = PowsyblMonitoredElements(
        branches=monitored_branches,
        trafo3w=monitored_trafo3w,
        switches=monitored_switches,
        buses=monitored_buses,
        voltage_levels=list(monitored_voltage_levels),
    )

    element_name_mapping = {element.id: element.name or "" for element in monitored_elements}
    missing_element_ids = missing_branches | missing_buses | missing_switches
    missing_elements = [element for element in monitored_elements if element.id in missing_element_ids]
    return powsybl_monitored_elements, element_name_mapping, missing_elements


def prepare_branch_limits(branch_limits: pd.DataFrame, chosen_limit: str, monitored_branches: list[str]) -> pd.DataFrame:
    """Prepare the branch limits for the N-1 analysis.

    This is done here, to avoid having to do this in every process.

    Parameters
    ----------
    branch_limits : pd.DataFrame
        The dataframe containing the branch limits of the network.
    chosen_limit : str
        The name of the limit to be used for the N-1 analysis. This is usually "permanent_limit".
        #TODO Decide if and how this could be extended to other limits.
    monitored_branches : list[str]
        The list of branches to be monitored during the N-1 analysis.

    Returns
    -------
    branch_limits : pd.DataFrame
        The dataframe containing the branch limits for the N-1 analysis in the right format.
    """
    translated_limits = branch_limits.reset_index()
    chosen_limit_type = translated_limits["name"] == chosen_limit
    limit_monitored = translated_limits["element_id"].isin(monitored_branches)
    translated_limits = translated_limits[chosen_limit_type & limit_monitored]
    translated_limits["side"] = translated_limits["side"].map({"ONE": 1, "TWO": 2, "THREE": 3})
    return translated_limits.groupby(by=["element_id", "side"]).min()[["value"]]


def get_blank_va_diff(
    all_outages: list[str], single_branch_outages: dict[str, str], monitored_switches: list[str]
) -> pd.DataFrame:
    """Get a blank dataframe for the voltage angle difference results.

    This already includes all possible contingencies and monitored switches.
    The buses of the switches and the outaged branches are added later.

    Parameters
    ----------
    all_outages : list[str]
        The list of all outages to be considered. For all of these cases, all switches need to be checked
    single_branch_outages : dict[str, str]
        A dictionary mapping contingency ids to single outaged element ids.
        For all of these cases, the specific outaged branch need to be checked.
    monitored_switches : list[str]
        The list of monitored switches to be considered. These are only the switches that are open and retained.

    Returns
    -------
    pd.DataFrame
        A blank dataframe with the correct index for the voltage angle difference results.
        The index is a MultiIndex with the following levels:
        - timestep: The timestep of the results
        - contingency: The contingency id (including an empty string for the base case)
        - element: The element id (the switch or outaged branch)
    """
    basecase_in_result = ""
    switch_va_diff_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [
                [basecase_in_result, *all_outages],  # Add the empty string for the basecase
                monitored_switches,
            ],
            names=["contingency", "element"],
        )
    )
    outage_va_diff_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [
                single_branch_outages.keys(),
            ],
            names=["contingency"],
        )
    )
    outage_va_diff_df["element"] = single_branch_outages.values()
    outage_va_diff_df.set_index(["element"], append=True, inplace=True)
    blank_va_diff_df = pd.concat([switch_va_diff_df, outage_va_diff_df], axis=0)
    return blank_va_diff_df


def get_blank_va_diff_with_buses(
    branches: pd.DataFrame,
    switches: pd.DataFrame,
    pow_contingencies: list[PowsyblContingency],
    monitored_switches: list[str],
) -> pd.DataFrame:
    """Get a blank dataframe for the voltage angle difference results with the buspairs that need checking.

    The buspairs are bus_breaker_buses (net.get_bus_breaker_view_buses)

    Parameters
    ----------
    branches : pd.DataFrame
        The dataframe containing the branches of the network and their bus_breaker_buses.
    switches : pd.DataFrame
        The dataframe containing the switches of the network and their bus_breaker_buses.
    pow_contingencies: list[PowsyblContingency]
        The list of all contingencies to be considered. For all of these cases, all switches need to be checked.
        For single outages we also consider the outaged branches.
    monitored_switches : list[str]
        The list of monitored switches to be considered. These are only the switches that are open and retained.

    Returns
    -------
    pd.DataFrame
        A blank dataframe with the correct index for the voltage angle difference results.
        The index is a MultiIndex with the following levels:
        - contingency: The contingency id (including an empty string for the base case)
        - element: The element id (the switch or outaged branch)
        - bus_breaker_bus1_id: The first bus_breaker_bus_id of the element
        - bus_breaker_bus2_id: The second bus_breaker_bus_id of the element

    """
    branch_indizes = branches.query("type in ['LINE', 'TWO_WINDINGS_TRANSFORMER', 'TIE_LINE']").index
    single_contingencies = [contingency for contingency in pow_contingencies if len(contingency.elements) == 1]
    single_branch_outages = {
        contingency.id: contingency.elements[0]
        for contingency in single_contingencies
        if contingency.elements[0] in branch_indizes
    }
    branches = branches.loc[single_branch_outages.values()][["bus_breaker_bus1_id", "bus_breaker_bus2_id"]]
    switches = switches.loc[monitored_switches]
    switches = switches[switches.open & switches.retained][["bus_breaker_bus1_id", "bus_breaker_bus2_id"]]
    element_df = pd.concat([branches, switches], axis=0)
    all_outage_ids = [contingency.id for contingency in pow_contingencies if len(contingency.elements) > 0]
    blank_va_diff = get_blank_va_diff(all_outage_ids, single_branch_outages, switches.index.tolist())

    va_diff_with_buses = blank_va_diff.merge(
        element_df, left_on=blank_va_diff.index.get_level_values("element"), right_index=True, how="left"
    ).drop(columns="key_0")
    return va_diff_with_buses


@pa.check_types
def get_va_diff_results(
    bus_results: pd.DataFrame, outages: list[str], va_diff_with_buses: pd.DataFrame, bus_map: pd.DataFrame, timestep: int
) -> pat.DataFrame[VADiffResultSchema]:
    """Get the voltage angle difference results for the given outages and bus results.

    Parameters
    ----------
    bus_results : pd.DataFrame
        The dataframe containing the bus results of powsybl contingency analysis.
    outages : list[str]
        The list of outages to be considered. These are the contingency ids that are outaged.
    va_diff_with_buses : pd.DataFrame
        The dataframe containing the voltage angle difference results with the bus pairs that need checking.
    bus_map: pd.DataFrame
        A mapping from busbar sections to bus breaker buses. This is used to convert the busbar sections to bus breaker buses
        in the Node Breaker model.
    timestep : int
        The timestep of the results.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the voltage angle difference results for the given outages.
    """
    if len(outages) == 0 or len(va_diff_with_buses) == 0:
        return get_empty_dataframe_from_model(VADiffResultSchema)
    basecase_in_result = ""
    iteration_va_diff = va_diff_with_buses.loc[
        va_diff_with_buses.index.get_level_values("contingency").isin([basecase_in_result, *outages])
    ]
    iteration_va_diff["timestep"] = timestep
    # Map busbar sections where there are any. For the rest use the bus_breaker_bus_id from the results (here the bus id)
    bus_results = bus_results.merge(
        bus_map.bus_breaker_bus_id, left_on=bus_results.index.get_level_values("bus_id"), right_index=True, how="left"
    )

    iteration_va_diff = iteration_va_diff.reset_index()
    # Map the values from the results to the buses of the switches and the outaged branches
    iteration_va_diff = iteration_va_diff.merge(
        bus_results[["v_angle"]].add_suffix("_1"),
        left_on=["contingency", "bus_breaker_bus1_id"],
        right_on=[bus_results.index.get_level_values("contingency_id"), bus_results.bus_breaker_bus_id],
        how="left",
    )
    iteration_va_diff = iteration_va_diff.merge(
        bus_results[["v_angle"]].add_suffix("_2"),
        left_on=["contingency", "bus_breaker_bus2_id"],
        right_on=[bus_results.index.get_level_values("contingency_id"), bus_results.bus_breaker_bus_id],
        how="left",
    )
    iteration_va_diff.drop_duplicates(inplace=True)
    iteration_va_diff.set_index(["timestep", "contingency", "element"], inplace=True)
    iteration_va_diff["va_diff"] = iteration_va_diff["v_angle_1"] - iteration_va_diff["v_angle_2"]

    iteration_va_diff = iteration_va_diff.drop(
        columns=["bus_breaker_bus1_id", "bus_breaker_bus2_id", "v_angle_1", "v_angle_2"]
    )

    # set empty columns to NaN
    iteration_va_diff["element_name"] = ""
    iteration_va_diff["contingency_name"] = ""

    return iteration_va_diff


def get_busbar_mapping(net: Network) -> pd.DataFrame:
    """Get a map between the different kind of buses in the network.

    Maps busbar sections and bus breaker buses to bus breaker buses and electrical buses.
    Maps the electrical buses to monitored buses (either bus breaker or busbar sections).

    Parameters
    ----------
    net : Network
        The Powsybl network to use for the translation. This is used to get the busbar sections and bus breaker buses.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the busbar mapping from busbar sections to bus breaker buses.
    """
    busbar_sections = net.get_injections(attributes=["type", "bus_breaker_bus_id", "bus_id", "voltage_level_id"]).query(
        "type == 'BUSBAR_SECTION'"
    )
    mapping_cols = ["bus_breaker_bus_id", "bus_id", "voltage_level_id"]
    bus_breaker_buses = net.get_bus_breaker_view_buses(attributes=["voltage_level_id", "bus_id"])
    bus_breaker_buses["bus_breaker_bus_id"] = bus_breaker_buses.index
    bus_map = pd.concat([busbar_sections[mapping_cols], bus_breaker_buses[mapping_cols]], axis=0)
    return bus_map


def translate_nminus1_for_powsybl(n_minus_1_definition: Nminus1Definition, net: Network) -> PowsyblNMinus1Definition:
    """Translate the N-1 definition to a format that can be used in Powsybl.

    Parameters
    ----------
    n_minus_1_definition : Nminus1Definition
        The N-1 definition to translate.
    net : Network
        The Powsybl network to use for the translation. This is used to get the busbarsections, buses, branches and switches.

    Returns
    -------
    PowsyblNMinus1Definition
        The translated N-1 definition that can be used in Powsybl.
    """
    id_type = n_minus_1_definition.id_type or "powsybl"
    # By default we assume the id_type is powsybl. This works for all powsybl identifiables
    if id_type not in (supported_ids := get_args(POWSYBL_SUPPORTED_ID_TYPES)):
        raise ValueError(
            f"Unsupported id_type {n_minus_1_definition.id_type}. Only {supported_ids} are supported for Powsybl."
        )

    # Load data once from the network
    busmap = get_busbar_mapping(net)
    voltage_levels = net.get_voltage_levels(attributes=["nominal_v", "high_voltage_limit", "low_voltage_limit"])
    voltage_levels["high_voltage_limit"] = voltage_levels["high_voltage_limit"].fillna(voltage_levels["nominal_v"] * 1.2)
    voltage_levels["low_voltage_limit"] = voltage_levels["low_voltage_limit"].fillna(voltage_levels["nominal_v"] * 0.8)

    branch_limits = net.get_operational_limits().query("type=='CURRENT'")
    branches = net.get_branches(
        attributes=["type", "voltage_level1_id", "voltage_level2_id", "bus_breaker_bus1_id", "bus_breaker_bus2_id"]
    )
    trafo3ws = net.get_3_windings_transformers(
        attributes=[
            "voltage_level1_id",
            "voltage_level2_id",
            "voltage_level3_id",
            "bus_breaker_bus1_id",
            "bus_breaker_bus2_id",
            "bus_breaker_bus3_id",
        ]
    ).assign(type="THREE_WINDINGS_TRANSFORMER")
    all_branches = pd.concat([branches, trafo3ws], axis=0)
    switches = net.get_switches(
        attributes=["open", "retained", "voltage_level_id", "bus_breaker_bus1_id", "bus_breaker_bus2_id"]
    )
    trafo3ws = net.get_3_windings_transformers(
        attributes=[
            "voltage_level1_id",
            "voltage_level2_id",
            "voltage_level3_id",
            "bus_breaker_bus1_id",
            "bus_breaker_bus2_id",
            "bus_breaker_bus3_id",
        ]
    )
    identifiables = net.get_identifiables(attributes=[]).index
    pow_contingencies, missing_contingencies = translate_contingency_to_powsybl(
        n_minus_1_definition.contingencies, identifiables
    )
    contingency_name_map = {contingency.id: contingency.name or "" for contingency in n_minus_1_definition.contingencies}
    (monitored_elements, element_name_map, missing_elements) = translate_monitored_elements_to_powsybl(
        n_minus_1_definition, all_branches, busmap, switches
    )

    # create an empty dataframe with the correct index
    va_diff_with_buses = get_blank_va_diff_with_buses(branches, switches, pow_contingencies, monitored_elements["switches"])
    branch_limits = prepare_branch_limits(branch_limits, "permanent_limit", monitored_elements["branches"])
    return PowsyblNMinus1Definition(
        contingencies=pow_contingencies,
        blank_va_diff=va_diff_with_buses,
        monitored_elements=monitored_elements,
        branch_limits=branch_limits,
        bus_map=busmap,
        element_name_mapping=element_name_map,
        contingency_name_mapping=contingency_name_map,
        voltage_levels=voltage_levels,
        distributed_slack=n_minus_1_definition.loadflow_parameters.distributed_slack,
        missing_elements=missing_elements,
        missing_contingencies=missing_contingencies,
        contingency_propagation=n_minus_1_definition.loadflow_parameters.contingency_propagation,
    )


@pa.check_types
def get_regulating_element_results(
    monitored_buses: list[str], timestep: int, basecase_name: str | None = None
) -> pat.DataFrame[RegulatingElementResultSchema]:
    """Get the regulating element results for the given outages and timestep.

    TODO: This is a fake implementation, we need to get the real results from the powsybl security analysis

    Parameters
    ----------
    monitored_buses : list[str]
        The list of monitored buses to get the regulating element results for
    timestep : int
        The timestep to get the regulating element results for
    basecase_name : str | None, optional
        The name of the basecase contingency, if it is included in the run. Otherwise None, by default None

    Returns
    -------
    pat.DataFrame[RegulatingElementResultSchema]
        The regulating element results for the given outages and timestep
    """
    regulating_elements = get_empty_dataframe_from_model(RegulatingElementResultSchema)
    # TODO dont fake this
    if basecase_name and len(monitored_buses) > 0:
        regulating_elements.loc[(timestep, basecase_name, monitored_buses[0]), "value"] = -9999.0
        regulating_elements.loc[(timestep, basecase_name, monitored_buses[0]), "regulating_element_type"] = (
            RegulatingElementType.GENERATOR_Q.value
        )
        regulating_elements.loc[(timestep, basecase_name, monitored_buses[0]), "value"] = 9999.0
        regulating_elements.loc[(timestep, basecase_name, monitored_buses[0]), "regulating_element_type"] = (
            RegulatingElementType.SLACK_P.value
        )
    return regulating_elements


@pa.check_types
def get_node_results(
    bus_results: pd.DataFrame,
    monitored_buses: list[str],
    bus_map: pd.DataFrame,
    voltage_levels: pd.DataFrame,
    failed_outages: list[str],
    timestep: int,
    method: Literal["ac", "dc"],
) -> pat.DataFrame[NodeResultSchema]:
    """Get the node results for the given outages and timestep.

    TODO: This is currently faking the sum of p and q at the node

    Parameters
    ----------
    bus_results : pd.DataFrame
        The bus results from the powsybl security analysis
    monitored_buses : list[str]
        The list of monitored buses to get the node results for
    bus_map: pd.DataFrame,
        A mapping from busbar sections or bus_breaker_buses to the electrical buses.
        This is used to map the buses from bus_results to electrical buses and back to the monitored buses.
    voltage_levels: pd.DataFrame,
        The voltage levels of the buses. This is used to determine
        voltage limits and nominal v in DC.
    failed_outages : list[str]
        The list of failed outages to get nan-node results for
    timestep : int
        The timestep to get the node results for
    method : Literal["ac", "dc"]
        The method to use for the node results. Either "ac" or "dc"

    Returns
    -------
    pat.DataFrame[NodeResultSchema]
        The node results for the given outages and timestep
    """
    if bus_results.empty:
        return get_failed_node_results(timestep, failed_outages, monitored_buses)
    # Translate bus_ids that could be busbar sections or bus_breaker_buses to the monitored buses
    # Should work for both busbar and bus_breaker models
    node_results = deepcopy(bus_results)
    node_results["bus_breaker_bus_id"] = node_results.index.get_level_values("bus_id").map(bus_map.bus_breaker_bus_id)
    node_results = node_results.dropna(subset=["bus_breaker_bus_id"])
    monitored_bus_map = bus_map.loc[monitored_buses]
    bus_to_element_map = pd.DataFrame(
        data={"element": monitored_bus_map.index.values}, index=monitored_bus_map.bus_breaker_bus_id.values
    )
    node_results = node_results.merge(bus_to_element_map, right_index=True, left_on="bus_breaker_bus_id")

    # Merge the actual voltage level in kV
    voltage_columns = voltage_levels.columns.to_list()
    node_results[voltage_columns] = voltage_levels.loc[node_results.index.get_level_values("voltage_level_id")].values
    node_results = node_results.assign(timestep=0)
    node_results.index = pd.MultiIndex.from_arrays(
        [
            node_results.timestep.values,
            node_results.index.get_level_values("contingency_id").values,
            node_results.element.values,
        ],
        names=["timestep", "contingency", "element"],
    )

    node_results.rename(columns={"v_mag": "vm", "v_angle": "va"}, inplace=True)

    # Calculate the values
    if method == "dc":
        has_va = node_results["va"].notna().values
        node_results.loc[has_va, "vm"] = node_results.loc[has_va, "nominal_v"]
    vm_deviation = node_results["vm"].values - node_results["nominal_v"].values
    deviation_to_max = vm_deviation / (node_results["high_voltage_limit"].values - node_results["nominal_v"].values)
    deviation_to_min = vm_deviation / (node_results["nominal_v"].values - node_results["low_voltage_limit"].values)
    higher_voltage = vm_deviation > 0
    node_results.loc[higher_voltage, "vm_loading"] = deviation_to_max[higher_voltage]
    node_results.loc[~higher_voltage, "vm_loading"] = deviation_to_min[~higher_voltage]
    # TODO Add sum of p and q at the node
    failed_node_results = get_failed_node_results(timestep, failed_outages, monitored_buses)

    all_node_results = pd.concat([node_results, failed_node_results], axis=0)[["vm", "va", "vm_loading"]]

    # set empty dataframe columns to NaN
    all_node_results["p"] = np.nan
    all_node_results["q"] = np.nan
    all_node_results["element_name"] = ""
    all_node_results["contingency_name"] = ""

    return all_node_results


@pa.check_types
def get_branch_results(
    branch_results: pd.DataFrame,
    three_winding_results: pd.DataFrame,
    monitored_branches: list[str],
    monitored_trafo3w: list[str],
    failed_outages: list[str],
    timestep: int,
    branch_limits: pd.DataFrame,
) -> pat.DataFrame[BranchResultSchema]:
    """Get the branch results for the given outages and timestep.

    Parameters
    ----------
    branch_results : pd.DataFrame
        The branch results from the powsybl security analysis
    three_winding_results : pd.DataFrame
        The three winding transformer results from the powsybl security analysis
    monitored_branches : list[str]
        The list of monitored branches with 2 sides to get the branch results for
    monitored_trafo3w : list[str]
        The list of monitored three winding transformers to get the branch results for
    failed_outages : list[str]
        The list of failed outages to get nan-branch results for
    timestep : int
        The timestep to get the branch results for
    branch_limits : pd.DataFrame
        The branch limits from the powsybl network

    Returns
    -------
    pat.DataFrame[BranchResultSchema]
        The branch results for the given outages and timestep
    """
    # Align all indizes
    branch_results = branch_results.droplevel("operator_strategy_id")
    branch_results.index.rename({"contingency_id": "contingency", "branch_id": "element"}, inplace=True)
    three_winding_results.index.rename({"contingency_id": "contingency", "transformer_id": "element"}, inplace=True)

    side_one_results = (
        pd.concat([branch_results[["p1", "q1", "i1"]], three_winding_results[["p1", "q1", "i1"]]], axis=0)
        .assign(side=BranchSide.ONE.value)
        .rename(columns={"p1": "p", "q1": "q", "i1": "i"})
    )
    side_two_results = (
        pd.concat([branch_results[["p2", "q2", "i2"]], three_winding_results[["p2", "q2", "i2"]]], axis=0)
        .assign(side=BranchSide.TWO.value)
        .rename(columns={"p2": "p", "q2": "q", "i2": "i"})
    )
    side_three_results = (
        three_winding_results[["p3", "q3", "i3"]]
        .assign(side=BranchSide.THREE.value)
        .rename(columns={"p3": "p", "q3": "q", "i3": "i"})
    )
    # Combine and Add timestep column
    converted_branch_results = pd.concat([side_one_results, side_two_results, side_three_results], axis=0).assign(
        timestep=timestep
    )
    converted_branch_results = converted_branch_results.set_index(["side", "timestep"], append=True)
    converted_branch_results = converted_branch_results.reorder_levels(["timestep", "contingency", "element", "side"])

    # Add missing MultiIndex levels
    # divide current flow by current limits, but only keep the rows, that were there before
    indexer = pd.MultiIndex.from_arrays(
        [converted_branch_results.index.get_level_values("element"), converted_branch_results.index.get_level_values("side")]
    )
    converted_branch_results["loading"] = converted_branch_results["i"].values / branch_limits.reindex(indexer).value.values

    # Add results for non convergent contingencies
    failed_branch_results = get_failed_branch_results(timestep, failed_outages, monitored_branches, monitored_trafo3w)

    converted_branch_results = pd.concat([converted_branch_results, failed_branch_results], axis=0)
    return converted_branch_results


@pa.check_types
def get_convergence_result_df(
    post_contingency_results: dict[str, PostContingencyResult],
    pre_contingency_result: PreContingencyResult,
    outages: list[str],
    timestep: int,
    basecase_name: Optional[str] = None,
) -> tuple[pat.DataFrame[ConvergedSchema], list[str]]:
    """Get the convergence dataframe for the given outages and timestep.

    Parameters
    ----------
    post_contingency_results: dict[str, PostContingencyResult],
        The post contingency results from the powsybl security analysis.
        Maps contingency id to PostContingencyResult.
    pre_contingency_result : PreContingencyResult
        The pre contingency result from the powsybl security analysis. Holds the Basecase.
    outages : list[str]
        The list of outages to get the convergence results for
    timestep : int
        The timestep to get the convergence results for
    basecase_name : Optional[str], optional
        The name of the basecase contingency, if it is included in the run. Otherwise None, by default None

    Returns
    -------
    pat.DataFrame[ConvergedSchema]
        The convergence dataframe for the given outages and timestep
    list[str]
        The list of failed outages
    """
    converge_converted_df = pd.DataFrame(index=outages)
    converge_converted_df.index.name = "contingency"
    converge_converted_df["timestep"] = timestep
    converge_converted_df.set_index(["timestep"], inplace=True, append=True)
    converge_converted_df = converge_converted_df.reorder_levels(["timestep", "contingency"], axis=0)
    converge_converted_df["status"] = [
        post_contingency_results[contingency].status.value
        if contingency in post_contingency_results
        else pypowsybl.loadflow.ComponentStatus.NO_CALCULATION.value
        for contingency in outages
    ]
    converge_converted_df.status = converge_converted_df["status"].map(POWSYBL_CONVERGENCE_MAP)
    failed_outages = [
        outage
        for outage, success in zip(outages, converge_converted_df.status.values == "CONVERGED", strict=True)
        if not success
    ]

    if basecase_name is not None:
        # Add the basecase to the convergence dataframe
        converge_converted_df.loc[(timestep, basecase_name), "status"] = POWSYBL_CONVERGENCE_MAP[
            pre_contingency_result.status.value
        ]

    converge_converted_df["iteration_count"] = np.nan
    converge_converted_df["warnings"] = ""
    converge_converted_df["contingency_name"] = ""

    return converge_converted_df, failed_outages


@pa.check_types(inplace=True)
def update_basename(
    result_df: LoadflowResultTable,
    basecase_name: Optional[str] = None,
) -> LoadflowResultTable:
    """Update the basecase name in the results dataframes.

    This function updates the contingency index level of the results dataframes to
    reflect the basecase name. If the basecase is not included in the run, it will
    remove it from the results. Powsybl includes the basecase as an empty string by default.

    The Dataframes are expected to have a multi-index with a "contingency" level.
    The Dataframes are updated inplace.

    Parameters
    ----------
    result_df: LOADFLOW_RESULT_TABLE
        The dataframe containing the branch / node / VADiff results
    basecase_name: Optional[str], optional
        The name of the basecase contingency, if it is included in the run. Otherwise None, by default None

    Returns
    -------
    LOADFLOW_RESULT_TABLE
        The updated dataframes with the basecase name set or removed.
    """
    contingency_index_level = result_df.index.names.index("contingency")
    if basecase_name is not None:
        result_df.index = result_df.index.set_levels(
            result_df.index.levels[contingency_index_level].map(lambda x: basecase_name if x == "" else x),
            level=contingency_index_level,
        )
        return result_df
    result_df.drop("", level=contingency_index_level, axis=0, inplace=True, errors="ignore")
    return result_df


@pa.check_types(inplace=True)
def add_name_column(
    result_df: LoadflowResultTable,
    name_map: dict[str, str],
    index_level: str = "element",
) -> LoadflowResultTable:
    """Translate the element ids in the results dataframes to the original names.

    This function translates the element names in the results dataframes to the original names
    from the Powsybl network. This is useful for debugging and for displaying the results.

    Parameters
    ----------
    result_df: LoadflowResultTable
        The dataframe containing the node / branch / VADiff results
    name_map: dict[str | str]
        A mapping from the element ids to the original names. This is used to translate the element names in the results.
    index_level: str, optional
        The index level storing the ids that should be mapped to the names. by default "element" for the monitored elements.

    Returns
    -------
    LoadflowResultTable
        The updated dataframe with the ids translated to the original names.
    """
    result_df[f"{index_level}_name"] = result_df.index.get_level_values(index_level).map(name_map).fillna("")
    return result_df


def set_target_values_to_lf_values_incl_distributed_slack(net: Network, method: Literal["ac", "dc"]) -> Network:
    """Update the target values of generators to include the distributed slack.

    This is necessary if you want to run the security analysis for generators without distributed their
    outaged power across the whole network, but still want to mantain the original n0-flows.

    Parameters
    ----------
    net : Network
        The powsybl network to update
    method : Literal["ac", "dc"]
        The method to use for the loadflow, either "ac" or "dc"

    Returns
    -------
    Network
        The updated network
    """
    if method == "ac":
        pypowsybl.loadflow.run_ac(net, DISTRIBUTED_SLACK)
    else:
        pypowsybl.loadflow.run_dc(net, DISTRIBUTED_SLACK)
    gens = net.get_generators()
    gens["target_p"] = (-gens["p"]).fillna(gens["target_p"])
    if method == "ac":
        gens["target_q"] = (-gens["q"]).fillna(gens["target_q"])
    net.update_generators(gens[["target_p", "target_q"]])
    batteries = net.get_batteries()
    batteries["target_p"] = (-batteries["p"]).fillna(batteries["target_p"])
    if method == "ac":
        batteries["target_q"] = (-batteries["q"]).fillna(batteries["target_q"])
    net.update_batteries(batteries[["target_p", "target_q"]])
    return net


def get_full_nminus1_definition_powsybl(net: pypowsybl.network.Network) -> Nminus1Definition:
    """Get the full N-1 definition from a Powsybl network.

    This function retrieves the N-1 definition from a Powsybl network, including:
        Monitored Elements
            all lines, trafos, buses and switches
        Contingencies
            all lines, trafos, generators and loads
            Basecase contingency with name "BASECASE"

    Parameters
    ----------
    net : pypowsybl.network.Network
        The Powsybl network to retrieve the N-1 definition from.

    Returns
    -------
    Nminus1Definition
        The complete N-1 definition for the given Powsybl network.
    """
    lines = [
        GridElement(id=id, name=getattr(row, "name", ""), type="LINE", kind="branch")
        for id, row in net.get_lines(attributes=["name"]).iterrows()
    ]
    trafo2w = [
        GridElement(id=id, name=getattr(row, "name", ""), type="TWO_WINDINGS_TRANSFORMER", kind="branch")
        for id, row in net.get_2_windings_transformers(attributes=["name"]).iterrows()
    ]
    trafos3w = [
        GridElement(id=id, name=getattr(row, "name", ""), type="THREE_WINDINGS_TRANSFORMER", kind="branch")
        for id, row in net.get_3_windings_transformers(attributes=["name"]).iterrows()
    ]

    branch_elements = [*lines, *trafo2w, *trafos3w]

    switches = [
        GridElement(id=id, name=getattr(row, "name", ""), type="SWITCH", kind="switch")
        for id, row in net.get_switches(attributes=["name"]).iterrows()
    ]
    buses = net.get_busbar_sections(attributes=[])
    if buses.empty:
        buses = net.get_bus_breaker_view_buses(attributes=[])
    buses = [GridElement(id=id, name=getattr(row, "name", ""), type="BUS", kind="bus") for id, row in buses.iterrows()]
    monitored_elements = [*branch_elements, *switches, *buses]

    generators = [
        GridElement(id=id, name=getattr(row, "name", ""), type="GENERATOR", kind="injection")
        for id, row in net.get_generators(attributes=["name"]).iterrows()
    ]
    loads = [
        GridElement(id=id, name=getattr(row, "name", ""), type="LOAD", kind="injection")
        for id, row in net.get_loads(attributes=["name"]).iterrows()
    ]
    outaged_elements = [*branch_elements, *generators, *loads]

    basecase_contingency = [Contingency(id="BASECASE", name="BASECASE", elements=[])]
    single_contingencies = [
        Contingency(id=element.id, name=element.name or "", elements=[element]) for element in outaged_elements
    ]

    nminus1_definition = Nminus1Definition(
        contingencies=[*basecase_contingency, *single_contingencies],
        monitored_elements=monitored_elements,
        id_type="powsybl",
        loadflow_parameters=LoadflowParameters(
            distributed_slack=True,  # This is the default for Powsybl
        ),
    )
    return nminus1_definition
