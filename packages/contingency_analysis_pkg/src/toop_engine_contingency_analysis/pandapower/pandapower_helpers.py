"""Module containing helper functions to translate Pandapower N-1 definitions and results."""

import dataclasses
from typing import Any, get_args

import numpy as np
import pandapower
import pandas as pd
import pandera as pa
import pandera.typing as pat
from beartype.typing import Literal
from networkx.classes import MultiGraph
from pandapower import pandapowerNet
from pandapower.toolbox import res_power_columns
from pandera.typing import Index, Series
from pydantic import BaseModel, Field
from toop_engine_grid_helpers.pandapower.pandapower_helpers import get_element_table
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    get_globally_unique_id,
    get_globally_unique_id_from_index,
    parse_globally_unique_id,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    ConvergedSchema,
    NodeResultSchema,
    RegulatingElementResultSchema,
    RegulatingElementType,
    VADiffResultSchema,
)
from toop_engine_interfaces.nminus1_definition import (
    ELEMENT_ID_TYPES,
    PANDAPOWER_SUPPORTED_ID_TYPES,
    Contingency,
    GridElement,
    LoadflowParameters,
    Nminus1Definition,
)

BUS_COLUMN_MAP = {
    "line": ["from_bus", "to_bus"],
    "trafo": ["hv_bus", "lv_bus"],
    "trafo3w": ["hv_bus", "lv_bus"],
}


@dataclasses.dataclass
class SlackAllocationConfig:
    """Carry configuration required for slack allocation per island."""

    net_graph: MultiGraph
    bus_lookup: list[int]
    min_island_size: int = 11


class VADiffInfo(BaseModel):
    """Class to hold information about which switches to monitor for voltage angle difference.

    For each contingency, we need to know the voltage angle difference between the from and to bus of each affected branch.
    This is necessary to determine if we could easily reconnect the outaged branch after the contingency
    using the existing power switches.
    """

    from_bus: int
    """The from side of the branch. The voltage angle difference is calculated as va(from_bus) - va(to_bus)."""
    to_bus: int
    """The to side of the branch. The voltage angle difference is calculated as va(from_bus) - va(to_bus)."""

    power_switches_from: dict[str, str]
    """A mapping from switch unique ids to their names for the from side of the branch."""

    power_switches_to: dict[str, str]
    """A mapping from switch unique ids to their names for the to side of the branch."""


class PandapowerMonitoredElementSchema(pa.DataFrameModel):
    """Schema for a monitored element in the N-1 definition."""

    unique_id: Index[str] = pa.Field(description="The globally unique id of the monitored element.")
    table: Series[str] = pa.Field(description="The type of the monitored element, e.g. 'line', 'bus', 'load', etc.")
    table_id: Series[int] = pa.Field(description="The id of the monitored element in the corresponding table.")
    kind: Series[str] = pa.Field(
        isin=["branch", "bus", "injection", "switch"],
        description="The kind of the monitored element, e.g. 'branch', 'bus' etc.",
    )
    name: Series[str] = pa.Field(description="The name of the monitored element, if available.")


class PandapowerElements(BaseModel):
    """A Pandapower element with its globally unique id, table and table_id."""

    unique_id: str
    """The globally unique id of the element."""
    table: str
    """The type of the element, e.g. 'line', 'bus', 'load', etc."""
    table_id: int
    """The id of the element in the corresponding table."""
    name: str = ""
    """The name of the element, if available."""


class PandapowerContingency(BaseModel):
    """A contingency for Pandapower.

    Adds info about the table and table_id of the outaged elements.
    """

    unique_id: str
    """The globally unique id of the contingency."""
    name: str = ""
    """The name of the contingency, if available."""

    elements: list[PandapowerElements]
    """The elements that are outaged in this contingency."""

    va_diff_info: list[VADiffInfo] = Field(default_factory=list)
    """A mapping from nodes at branches and their closest Circuit breaker switches."""


class PandapowerNMinus1Definition(BaseModel):
    """A Pandapower N-1 definition.

    This is a simplified version of the NMinus1Definition that is used in Pandapower.
    It contains only the necessary information to run an N-1 analysis in Pandapower.
    """

    model_config = {"arbitrary_types_allowed": True}

    contingencies: list[PandapowerContingency]
    """The outages to be considered. Maps contingency id to outaged element ids."""

    missing_contingencies: list[Contingency]
    """A list of contingencies that were not found in the network."""

    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
    """A dictionary mapping the element kind to a list of element ids that are monitored."""

    missing_elements: list[GridElement]
    """A list of monitored elements that were not found in the network."""

    duplicated_grid_elements: list[str] = Field(
        default_factory=list,
        description="A list of ids that were not unique in the grid. This is only relevant for cgmes ids.",
    )

    def __getitem__(self, key: str | int | slice) -> "PandapowerNMinus1Definition":
        """Get a subset of the nminus1definition based on the contingencies.

        If a string is given, the contingency id must be in the contingencies list.
        If an integer or slice is given, the case id will be indexed by the integer or slice.
        """
        if isinstance(key, str):
            contingency_ids = [contingency.unique_id for contingency in self.contingencies]
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
        return PandapowerNMinus1Definition.model_validate(updated_definition)


def translate_monitored_elements(
    net: pandapowerNet, monitored_elements: list[GridElement], id_type: PANDAPOWER_SUPPORTED_ID_TYPES = "unique_pandapower"
) -> tuple[pat.DataFrame[PandapowerMonitoredElementSchema], list[GridElement], list[str]]:
    """Translate the monitored elements to a format that can be used in Pandapower.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    monitored_elements : list[GridElement]
        The list of monitored elements to translate.
    id_type: ELEMENT_ID_TYPES = "unique_pandapower"
        The type of ids to use for the monitored elements. Currently only "unique_pandapower" and "cgmes" is supported.
        TODO: Add support for other id types.

    Returns
    -------
    pat.DataFrame[PandapowerMonitoredElementSchema]
        A pandas DataFrame containing the monitored elements with their globally unique ids, table, table_id, kind and name.
    list[GridElement]
        A list of monitored elements that were not found in the network.
    list[str]
        A list of ids that were not unique in the grid. This is only relevant for cgmes ids.
    """
    if id_type == "unique_pandapower":
        pandapower_monitored_elements, missing_elements, duplicated_ids = (
            extract_monitored_elements_with_unique_pandapower_id(net, monitored_elements)
        )
    elif id_type == "cgmes":
        pandapower_monitored_elements, missing_elements, duplicated_ids = extract_monitored_elements_with_cgmes_id(
            net, monitored_elements
        )
    else:
        raise ValueError(f"Unsupported id_type: {id_type}")
    return pandapower_monitored_elements, missing_elements, duplicated_ids


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


def translate_contingencies(
    net: pandapowerNet, contingencies: list[Contingency], id_type: ELEMENT_ID_TYPES = "unique_pandapower"
) -> tuple[list[PandapowerContingency], list[Contingency], list[str]]:
    """Translate the contingencies to a format that can be used in Pandapower.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    contingencies : list[Contingency]
        The list of contingencies to translate.
    id_type: ELEMENT_ID_TYPES = "unique_pandapower"
        The type of ids to use for the contingencies. Currently only "unique_pandapower" and "cgmes" is supported.

    Returns
    -------
    pp_contingencies: list[PandapowerContingency]
        A list translated Contingency to be used in Pandapower.
    missing_contingencies: list[Contingency]
        A list of contingencies that were not found in the network.
    duplicated_ids: list[str]
        A list of ids that were not unique in the grid. This is only relevant for cgmes ids.
    """
    if id_type == "unique_pandapower":
        pp_contingencies, missing_contingencies, duplicated_ids = extract_contingencies_with_unique_pandapower_id(
            net, contingencies
        )
    elif id_type == "cgmes":
        pp_contingencies, missing_contingencies, duplicated_ids = extract_contingencies_with_cgmes_id(net, contingencies)
    else:
        raise ValueError(f"Unsupported id_type: {id_type}. Supported id_types are: ['unique_pandapower', 'cgmes']")
    node_to_switch_map = get_node_to_switch_map(net, id_type=id_type)
    for contingency in pp_contingencies:
        contingency.va_diff_info = get_va_diff_info(contingency, net, node_to_switch_map)
    return pp_contingencies, missing_contingencies, duplicated_ids


def get_va_diff_info(
    contingency: PandapowerContingency,
    net: pandapowerNet,
    node_to_switch_map: dict[int, dict[str, str]],
) -> list[VADiffInfo]:
    """Add information about which switches to monitor for voltage angle difference to the contingency.

    This function modifies the contingency in place.

    Parameters
    ----------
    contingency : PandapowerContingency
        The contingency to add the information to.
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    node_to_switch_map : dict[int, list[int]]
        A mapping from nodes at branches and their closest Circuit breaker switches.
    """
    va_diff_info: list[VADiffInfo] = []
    for element in contingency.elements:
        if element.table not in BUS_COLUMN_MAP:
            continue
        from_bus_id, to_bus_id = net[element.table].loc[element.table_id, BUS_COLUMN_MAP[element.table]]
        switches_from = node_to_switch_map.get(from_bus_id, {})
        switches_to = node_to_switch_map.get(to_bus_id, {})
        if not {**switches_from, **switches_to}:
            continue
        va_diff_info.append(
            VADiffInfo(
                from_bus=from_bus_id, to_bus=to_bus_id, power_switches_from=switches_from, power_switches_to=switches_to
            )
        )
    return va_diff_info


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


def match_node_to_next_switch_type(
    node_ids: np.ndarray[tuple[Any, ...], np.dtype[np.integer]],
    switches_df: pd.DataFrame,
    actual_buses: np.ndarray[tuple[Any, ...], np.dtype[np.integer]],
    switch_type: str,
    id_type: PANDAPOWER_SUPPORTED_ID_TYPES,
    max_jumps: int = 4,
) -> pd.DataFrame:
    """Find the next switch of a given type for each node.

    Stops at nodes that are actual busbars.
    Finds switches that are connected by other swiches of another type.
    Only considers switches that are closed at the point of calling.

    Parameters
    ----------
    node_ids : np.ndarray
        The node ids to find the next switch for.
    switches_df : pd.DataFrame
        The switches dataframe from pandapower.
    actual_buses : np.ndarray
        The node ids of the actual busbars in the network.
        All others are helper nodes for modelling purposes.
    switch_type : str
        The type of switch to find. E.g. "CB" for circuit breaker.
    id_type: PANDAPOWER_SUPPORTED_ID_TYPES
        The type of ids to use for the contingencies. Currently only "unique_pandapower" and "cgmes" is supported.
    max_jumps : int
        The maximum number of jumps to make to find the next switch.

    Returns
    -------
    pd.DataFrame
        A dataframe with the original node id, the switch id, switch name and unique id.
    """
    switches_to_check = switches_df.reset_index(names="original_index").query("closed")
    if id_type == "unique_pandapower":
        switches_to_check["unique_id"] = get_globally_unique_id_from_index(switches_to_check.original_index, "switch")
    elif id_type == "cgmes":
        switches_to_check["unique_id"] = switches_to_check["origin_id"]
    else:
        raise ValueError(f"Unsupported ID Type: {id_type}")
    bidirectional_switches = np.concatenate(
        [
            switches_to_check[["original_index", "bus", "element", "type", "name", "unique_id"]].values,
            switches_to_check[["original_index", "element", "bus", "type", "name", "unique_id"]].values,
        ],
        axis=0,
    )
    all_switches_df = pd.DataFrame(
        bidirectional_switches, columns=["switch_idx", "bus", "element", "type", "name", "unique_id"]
    ).drop_duplicates()
    selected_switches = all_switches_df[all_switches_df.type == switch_type]
    other_switches = all_switches_df[all_switches_df.type != switch_type]

    node_df = pd.DataFrame.from_dict({"original_node": node_ids, "merge_node": node_ids})
    switch_found = []
    nodes_to_match = node_df[node_df.original_node.isin(all_switches_df.bus)]

    all_switches_df = all_switches_df[~all_switches_df.bus.isin(actual_buses)]
    for _ in range(max_jumps):
        merged_cbs = nodes_to_match.merge(selected_switches, left_on="merge_node", how="left", right_on="bus")
        switch_found.append(merged_cbs.dropna())
        merged_non_cbs = nodes_to_match.merge(other_switches, left_on="merge_node", how="left", right_on="bus")
        other_switches = other_switches[~other_switches.switch_idx.isin(merged_non_cbs.switch_idx)]
        nodes_to_match = merged_non_cbs.dropna()[["original_node", "element"]].rename(columns={"element": "merge_node"})
        nodes_to_match = nodes_to_match[~nodes_to_match.merge_node.isin(actual_buses)]
        if nodes_to_match.empty:
            break
    matched = pd.concat(switch_found)[["original_node", "switch_idx", "name", "unique_id"]]
    return matched


def get_node_to_switch_map(net: pandapowerNet, id_type: PANDAPOWER_SUPPORTED_ID_TYPES) -> dict[int, dict[str, str]]:
    """Get a mapping from nodes at branches and their closest Circuit breaker switches.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    id_type: PANDAPOWER_SUPPORTED_ID_TYPES
        The type of ids to use for the contingencies. Currently only "unique_pandapower" and "cgmes" is supported.

    Returns
    -------
    node_to_switch_map: dict[int, list[int]]
        A mapping from nodes at branches and their closest Circuit breaker switches.
    """
    considered_nodes = np.concatenate(
        [
            net.line.from_bus.values,
            net.line.to_bus.values,
            net.trafo.hv_bus.values,
            net.trafo.lv_bus.values,
            net.trafo3w.hv_bus.values,
            net.trafo3w.mv_bus.values,
            net.trafo3w.lv_bus.values,
        ]
    )
    actual_busbars = net.bus[net.bus.type == "b"].index.values
    matched = match_node_to_next_switch_type(
        considered_nodes, net.switch, actual_busbars, switch_type="CB", id_type=id_type, max_jumps=4
    )
    grouped_by_bus = matched.groupby("original_node").agg(list)[["unique_id", "name"]].to_dict(orient="index")
    node_to_switch_map = {outage: dict(zip(info["unique_id"], info["name"])) for outage, info in grouped_by_bus.items()}
    return node_to_switch_map


def translate_nminus1_for_pandapower(
    n_minus_1_definition: Nminus1Definition, net: pandapowerNet
) -> PandapowerNMinus1Definition:
    """Translate the N-1 definition to a format that can be used in Powsybl.

    Parameters
    ----------
    n_minus_1_definition : Nminus1Definition
        The N-1 definition to translate.
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc

    Returns
    -------
    PowsyblNMinus1Definition
        The translated N-1 definition that can be used in Powsybl.
    """
    id_type = n_minus_1_definition.id_type or "unique_pandapower"
    # If no id_type is specified, we assume pandapower's unique ids
    if id_type not in (supported_id_types := get_args(PANDAPOWER_SUPPORTED_ID_TYPES)):
        # If the id_type is not supported, we raise an error
        raise ValueError(f"Unsupported id_type: {id_type}. Supported id_types are: {supported_id_types}")
    pandapower_monitored_elements, missing_elements, duplicated_monitored_ids = translate_monitored_elements(
        net, n_minus_1_definition.monitored_elements, id_type=id_type
    )
    contingencies, missing_contingencies, duplicated_outaged_element_ids = translate_contingencies(
        net, n_minus_1_definition.contingencies, id_type=id_type
    )

    return PandapowerNMinus1Definition(
        monitored_elements=pandapower_monitored_elements,
        missing_elements=missing_elements,
        contingencies=contingencies,
        missing_contingencies=missing_contingencies,
        duplicated_grid_elements=duplicated_monitored_ids + duplicated_outaged_element_ids,
    )


@pa.check_types
def get_branch_results(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    timestep: int,
) -> pat.DataFrame[BranchResultSchema]:
    """Get the branch results for the given network and contingency

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the branch results for
    contingency: PandapowerContingency
        The contingency to compute the branch results for
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
        The list of monitored elements including branches
    timestep : int
        The timestep of the results

    Returns
    -------
    pat.DataFrame[BranchResultSchema]
        The branch results for the given network and contingency
    """
    monitored_branches = monitored_elements.query("kind == 'branch'")
    if monitored_branches.empty:
        # If no elements are monitored, return an empty dataframe
        return get_empty_dataframe_from_model(BranchResultSchema)
    max_amount_of_sides = 3
    branch_element_list = []

    for branch_type in monitored_branches.table.unique():
        branch_type_df = monitored_elements.loc[monitored_elements.table == branch_type]
        table_ids = branch_type_df.table_id.values
        unique_ids = branch_type_df.index.values
        for side in range(max_amount_of_sides):
            try:
                columns = res_power_columns(branch_type, side=side)
                columns.append(
                    columns[0].replace("p_", "i_").replace("_mw", "_ka")
                )  # hacky way to include the current aswell
                columns.append("loading_percent")
            except KeyError:
                # This means all sides were considered
                break
            branch_df = net[f"res_{branch_type}"].loc[table_ids, columns]
            branch_df = branch_df.assign(
                timestep=timestep, contingency=contingency.unique_id, side=side + 1, element=unique_ids
            )
            branch_df.set_index(["timestep", "contingency", "element", "side"], inplace=True)
            branch_df.rename(columns=dict(zip(columns, ["p", "q", "i", "loading"])), inplace=True)
            # Fix kA -> A and % -> 1
            branch_df["i"] *= 1000
            branch_df["loading"] /= 100
            branch_df.loc[branch_df.i.isna(), "p"] = np.nan
            branch_df.loc[branch_df.i.isna(), "q"] = np.nan
            branch_element_list.append(branch_df)
    branch_element_df = pd.concat(branch_element_list)
    # fill missing columns with NaN
    branch_element_df["element_name"] = ""
    branch_element_df["contingency_name"] = ""
    return branch_element_df


@pa.check_types
def get_node_result_df(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    timestep: int,
) -> pat.DataFrame[NodeResultSchema]:
    """Get the node results for the given network and contingency

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the node results for
    contingency: PandapowerContingency
        The contingency to compute the node results for
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
        The list of monitored elements including buses
    timestep : int
        The timestep of the results

    Returns
    -------
    pat.DataFrame[NodeResultSchema]
        The node results for the given network and contingency
    """
    monitored_buses = monitored_elements.query("kind == 'bus'")
    if monitored_buses.empty:
        # If no buses are monitored, return an empty dataframe
        return get_empty_dataframe_from_model(NodeResultSchema)
    table_ids = monitored_buses.table_id.to_list()
    unique_ids = monitored_buses.index.to_list()
    node_results_df = net.res_bus.reindex(table_ids)
    node_results_df = node_results_df.assign(timestep=timestep, contingency=contingency.unique_id, element=unique_ids)
    node_results_df.set_index(["timestep", "contingency", "element"], inplace=True)
    max_allowed_deviation = 0.2  # 20% voltage deviation is considered acceptable
    node_results_df["vm_loading"] = (node_results_df["vm_pu"] - 1) / max_allowed_deviation
    node_results_df.rename(columns={"vm_pu": "vm", "va_degree": "va", "p_mw": "p", "q_mvar": "q"}, inplace=True)
    voltage_levels = net.bus.reindex(table_ids)["vn_kv"].values
    node_results_df["vm"] *= voltage_levels
    # fill missing columns with NaN
    node_results_df["element_name"] = ""
    node_results_df["contingency_name"] = ""
    return node_results_df


@pa.check_types
def get_convergence_df(
    timestep: int, contingency: PandapowerContingency, status: Literal["NO_CALCULATION", "CONVERGED", "FAILED"]
) -> pat.DataFrame[ConvergedSchema]:
    """Get the convergence dataframe for the given network and contingency

    Parameters
    ----------
    timestep : int
        The timestep of the results
    contingency: PandapowerContingency
        The contingency to compute the node results for
    status : Literal["NO_CALCULATION", "CONVERGED", "FAILED"]
        The status of the loadflow calculation. Can be one of:
        - "NO_CALCULATION": No loadflow was run, e.g. because no elements were outaged
        - "CONVERGED": The loadflow converged successfully
        - "FAILED": The loadflow failed to converge

    Returns
    -------
    pat.DataFrame[ConvergedSchema]
        The convergence dataframe for the given network and contingency
    """
    convergence_df = pd.DataFrame()
    convergence_df["timestep"] = [timestep]
    convergence_df["contingency"] = contingency.unique_id
    convergence_df["contingency_name"] = contingency.name or ""
    convergence_df["status"] = status
    convergence_df.set_index(["timestep", "contingency"], inplace=True)
    # fill missing columns with NaN
    convergence_df["warnings"] = ""
    convergence_df["iteration_count"] = np.nan

    return convergence_df


@pa.check_types
def get_regulating_element_results(
    timestep: int, monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema], contingency: PandapowerContingency
) -> pat.DataFrame[RegulatingElementResultSchema]:
    """Get the regulating element results for the given network and contingency.

    This currently only returns fake slack bus and generator results for the basecase

    Parameters
    ----------
    timestep : int
        The timestep of the results
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
        The dataframe containing the monitored elements
    contingency: PandapowerContingency
        The contingency to compute the regulating element results for

    Returns
    -------
    pat.DataFrame[RegulatingElementResultSchema]
        The regulating element results for the given network and contingency
    """
    if monitored_elements.empty:
        # If no elements are monitored, return an empty dataframe
        return get_empty_dataframe_from_model(RegulatingElementResultSchema)
    regulating_elements = get_empty_dataframe_from_model(RegulatingElementResultSchema)
    # TODO dont fake this
    if len(contingency.elements) == 0:
        regulating_elements.loc[(timestep, contingency.unique_id, monitored_elements.index[0]), "value"] = -9999.0
        regulating_elements.loc[
            (timestep, contingency.unique_id, monitored_elements.index[0]), "regulating_element_type"
        ] = RegulatingElementType.GENERATOR_Q.value
        regulating_elements.loc[(timestep, contingency.unique_id, monitored_elements.index[1]), "value"] = 9999.0
        regulating_elements.loc[
            (timestep, contingency.unique_id, monitored_elements.index[1]), "regulating_element_type"
        ] = RegulatingElementType.SLACK_P.value

    return regulating_elements


@pa.check_types
def get_va_diff_results(
    net: pandapowerNet,
    timestep: int,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    contingency: PandapowerContingency,
) -> pat.DataFrame[VADiffResultSchema]:
    """Get the voltage angle difference results for the given network and contingency.

    Currently does not return the va_diff of outaged trafo3w

    Parameters
    ----------
    net : pandapowerNet
        The network to compute the voltage angle difference results for
    timestep : int
        The timestep of the results
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
        The dataframe containing the monitored elements
    contingency : Contingency
        The contingency to compute the voltage angle difference results for.
        Will also calculate the va_diff of the outaged elements if they are lines or transformers
    node_to_switch_map: dict[int, list[int]]
        A mapping from nodes at branches and their closest Circuit breaker switches

    Returns
    -------
    pat.DataFrame[VADiffResultSchema]
        The voltage angle difference results for the given network and contingency
    """
    va_diff_df = get_empty_dataframe_from_model(VADiffResultSchema)
    monitored_switches = monitored_elements.query("kind == 'switch'")
    for unique_id, monitored_switch in monitored_switches.iterrows():
        switch_element = net.switch.loc[monitored_switch.table_id]
        if switch_element.closed or switch_element.et != "b":
            # Only consider open bus switches
            continue
        va_diff = net.res_bus.loc[switch_element.bus, "va_degree"] - net.res_bus.loc[switch_element.element, "va_degree"]
        va_diff_df.loc[(timestep, contingency.unique_id, unique_id), "va_diff"] = va_diff

    for va_diff_info in contingency.va_diff_info:
        va_diff = net.res_bus.loc[va_diff_info.from_bus, "va_degree"] - net.res_bus.loc[va_diff_info.to_bus, "va_degree"]
        for switch_id, switch_name in va_diff_info.power_switches_from.items():
            va_diff_df.loc[(timestep, contingency.unique_id, switch_id), ["va_diff", "element_name"]] = va_diff, switch_name
        for switch_id, switch_name in va_diff_info.power_switches_to.items():
            va_diff_df.loc[(timestep, contingency.unique_id, switch_id), ["va_diff", "element_name"]] = -va_diff, switch_name
    return va_diff_df


@pa.check_types
def get_failed_va_diff_results(
    timestep: int, monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema], contingency: PandapowerContingency
) -> pat.DataFrame[VADiffResultSchema]:
    """Get the voltage angle difference results for the given network and contingency when the loadflow failed.

    This will return NaN for all elements that were monitored and the contingency.

    Parameters
    ----------
    timestep : int
        The timestep of the results
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
        The list of monitored elements including switches
    contingency : Contingency
        The contingency to compute the voltage angle difference results for.
    contingency: PandapowerContingency

    Returns
    -------
    pat.DataFrame[VADiffResultSchema]
        The voltage angle difference results for the given network and contingency when the loadflow failed.
        This will return NaN for all elements that were monitored and the contingency.
    """
    monitored_switches = monitored_elements.query("kind == 'switch'").index.to_list()
    all_power_switches = {}
    for va_diff_info in contingency.va_diff_info:
        all_power_switches.update(va_diff_info.power_switches_from)
        all_power_switches.update(va_diff_info.power_switches_to)
    outaged_elements = list(all_power_switches.keys())
    elements = [*monitored_switches, *outaged_elements]
    failed_va_diff_results = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [[timestep], [contingency.unique_id], elements],
            names=["timestep", "contingency", "element"],
        )
    ).assign(va_diff=np.nan)
    # fill missing columns with NaN
    failed_va_diff_results["element_name"] = failed_va_diff_results.index.get_level_values("element").map(all_power_switches)
    failed_va_diff_results["contingency_name"] = ""
    return failed_va_diff_results


def get_full_nminus1_definition_pandapower(net: pandapower.pandapowerNet) -> Nminus1Definition:
    """Get the full N-1 definition from a pandapower network.

    This function retrieves the N-1 definition from a pandapower network, including:
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
        The complete N-1 definition for the given pandapower network.
    """
    lines = [
        GridElement(id=get_globally_unique_id(id, "line"), name=row["name"] or "", type="lines", kind="branch")
        for id, row in net.line.iterrows()
    ]
    trafo2w = [
        GridElement(id=get_globally_unique_id(id, "trafo"), name=row["name"] or "", type="trafo", kind="branch")
        for id, row in net.trafo.iterrows()
    ]
    trafos3w = [
        GridElement(id=get_globally_unique_id(id, "trafo3w"), name=row["name"] or "", type="trafo3w", kind="branch")
        for id, row in net.trafo3w.iterrows()
    ]

    branch_elements = [*lines, *trafo2w, *trafos3w]

    switches = [
        GridElement(id=get_globally_unique_id(id, "switch"), name=row["name"] or "", type="switch", kind="switch")
        for id, row in net.switch.iterrows()
    ]
    buses = [
        GridElement(id=get_globally_unique_id(id, "bus"), name=row["name"] or "", type="bus", kind="bus")
        for id, row in net.bus.iterrows()
    ]
    monitored_elements = [*branch_elements, *switches, *buses]

    generators = [
        GridElement(id=get_globally_unique_id(id, "gen"), name=row["name"] or "", type="gen", kind="injection")
        for id, row in net.gen.iterrows()
    ]
    loads = [
        GridElement(id=get_globally_unique_id(id, "load"), name=row["name"] or "", type="load", kind="injection")
        for id, row in net.load.iterrows()
    ]
    outaged_elements = [*branch_elements, *generators, *loads]

    basecase_contingency = [Contingency(id="BASECASE", name="BASECASE", elements=[])]
    single_contingencies = [
        Contingency(id=element.id, name=element.name or "", elements=[element]) for element in outaged_elements
    ]

    nminus1_definition = Nminus1Definition(
        contingencies=[*basecase_contingency, *single_contingencies],
        monitored_elements=monitored_elements,
        id_type="unique_pandapower",  # Default id type for Pandapower
        loadflow_parameters=LoadflowParameters(
            distributed_slack=False,  # This is the default for Powsybl
        ),
    )
    return nminus1_definition
