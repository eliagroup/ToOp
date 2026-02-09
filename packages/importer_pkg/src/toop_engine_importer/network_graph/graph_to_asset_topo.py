# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Creates a Asset Topology from a Network Graph."""

import logbook
import networkx as nx
import numpy as np
import pandas as pd
from beartype.typing import Literal, Optional, Union
from jaxtyping import Array, Bool
from toop_engine_importer.network_graph.data_classes import (
    BranchSchema,
    BusbarConnectionInfo,
    EdgeConnectionInfo,
    NodeAssetSchema,
    NodeSchema,
    SwitchableAssetSchema,
    SwitchSchema,
)
from toop_engine_interfaces.asset_topology import (
    AssetBay,
)

logger = logbook.Logger(__name__)


def get_busbar_df(nodes_df: NodeSchema, substation_id: str) -> pd.DataFrame:
    """Get the busbar from the NetworkGraphData nodes dataframe.

    Parameters
    ----------
    nodes_df: NodeSchema
        Dataframe with all nodes of the substation or whole network.
        expects NetworkGraphData.nodes
    substation_id: str
        Substation id for which the busbar should be retrieved.

    Returns
    -------
    busbar_df: pd.DataFrame
        busbar_df of the specified substation.
    """
    busbar_df = nodes_df[(nodes_df["substation_id"] == substation_id) & (nodes_df["node_type"] == "busbar")].copy()

    busbar_df = (
        busbar_df.sort_values(by="grid_model_id")
        .reset_index()
        .rename(columns={"foreign_id": "name", "node_type": "type", "bus_id": "bus_branch_bus_id"})
    )
    busbar_df["int_id"] = busbar_df.index

    busbar_df = busbar_df[["grid_model_id", "type", "name", "int_id", "in_service", "bus_branch_bus_id"]]

    return busbar_df


def get_coupler_df(switches_df: SwitchSchema, busbar_df: pd.DataFrame, substation_id: str, graph: nx.Graph) -> pd.DataFrame:
    """Get the busbar couplers from the NetworkGraphData edges dataframe.

    Parameters
    ----------
    switches_df: SwitchSchema
        Dataframe with all switches of the substation.
        expects NetworkGraphData.switches
    busbar_df: pd.DataFrame
        Dataframe with all busbars of the substation.
        expects get_busbar_df
    substation_id: str
        Substation id for which the busbar couplers should be retrieved.
        looks for the "substation_id", "coupler_type" and "asset_type" column in the edges dataframe.
    graph: nx.Graph
        The graph from the NetworkGraphData of the substation or whole network.
        Note: the default strategy functions must been executed before calling this function.

    Returns
    -------
    coupler_df: pd.DataFrame
        coupler_df of the specified substation.
    """
    edge_connection_dict_list = {
        graph.get_edge_data(row["from_node"], row["to_node"])["grid_model_id"]: graph.get_edge_data(
            row["from_node"], row["to_node"]
        )["edge_connection_info"].model_dump()
        for _, row in switches_df.iterrows()
    }
    edge_connection_df = pd.DataFrame.from_dict(edge_connection_dict_list, orient="index")

    switches_connected_to_busbars = switches_df.merge(
        edge_connection_df,
        left_on="grid_model_id",
        right_index=True,
        how="left",
    )
    coupler_df = switches_connected_to_busbars[switches_connected_to_busbars["coupler_type"] != ""]
    coupler_df["from_busbar_grid_model_id"] = ""
    coupler_df["to_busbar_grid_model_id"] = ""
    coupler_df["type"] = ""
    if coupler_df.empty:
        logger.warning(f"No couplers found in the substation {substation_id}. Please check Station.")
        return coupler_df
    busbar_out_of_service = busbar_df[~busbar_df["in_service"]]["grid_model_id"].to_list()
    for index, row in coupler_df.iterrows():
        grid_model_id = row["grid_model_id"]
        bay_df = switches_connected_to_busbars[switches_connected_to_busbars["bay_id"] == grid_model_id]
        coupler_df.loc[index, "from_busbar_grid_model_id"] = select_one_busbar_for_coupler_side(
            coupler_index=index,
            bay_df=bay_df,
            side="from",
            out_of_service_busbar_ids=busbar_out_of_service,
        )
        coupler_df.loc[index, "to_busbar_grid_model_id"] = select_one_busbar_for_coupler_side(
            coupler_index=index,
            bay_df=bay_df,
            side="to",
            out_of_service_busbar_ids=busbar_out_of_service,
            ignore_busbar_id=coupler_df.loc[index, "from_busbar_grid_model_id"],
        )
        coupler_df.loc[index, "type"] = row["asset_type"]
        bay_state = get_state_of_coupler_based_on_bay(
            coupler_index=index,
            bay_df=bay_df,
        )
        if bay_state:
            # coupler is open, if one side has all switches open
            coupler_df.loc[index, "open"] = True
            # coupler state is independent of the bay state, if bay sides have a closed switch
            # -> no else block

    coupler_df = coupler_df.sort_values(by="grid_model_id").reset_index().rename(columns={"foreign_id": "name"})
    # get the busbar id
    coupler_df = coupler_df.merge(
        busbar_df[["grid_model_id", "int_id"]],
        left_on="from_busbar_grid_model_id",
        right_on="grid_model_id",
        how="left",
        suffixes=("", "_from"),
    )
    coupler_df.rename(columns={"int_id": "busbar_from_id"}, inplace=True)
    coupler_df = coupler_df.merge(
        busbar_df[["grid_model_id", "int_id"]],
        left_on="to_busbar_grid_model_id",
        right_on="grid_model_id",
        how="left",
        suffixes=("", "_to"),
    )
    coupler_df.rename(columns={"int_id": "busbar_to_id"}, inplace=True)
    coupler_df = coupler_df[["grid_model_id", "type", "name", "in_service", "open", "busbar_from_id", "busbar_to_id"]]

    return coupler_df


def select_one_busbar_for_coupler_side(
    coupler_index: pd.Index,
    bay_df: pd.DataFrame,
    side: Literal["from", "to"],
    out_of_service_busbar_ids: list[str],
    ignore_busbar_id: str = "",
) -> str:
    """Select one busbar for the coupler side.

    This selects the a busbar from the list of busbar grid model ids, by looking up the
    state of the disconnector. The first disconnector found for an id  in busbar_grid_model_ids,
    with an closed state, is selected.
    If no closed state disconnector is found, the first busbar in the list is selected.

    Parameters
    ----------
    coupler_index: Index
        Index of the coupler in the dataframe.
    bay_df: pd.DataFrame
        Dataframe with the asset bay switches.
        expects the coupler_index to be part of the dataframe.
    side: Literal["from", "to"]
        Side of the coupler to select the busbar for.
        "from" or "to"
    out_of_service_busbar_ids: list[str]
        List of busbar grid model ids to avoid, but are allowed.
    ignore_busbar_id: str
        Busbar grid model ids to ignore.
        This can be used to ignore a busbar that is connected to the other side of the coupler.

    Returns
    -------
    busbar_grid_model_id: str
        Busbar grid model id.
        or None if no busbar is found.
    """
    busbar_grid_model_ids = bay_df.loc[coupler_index, f"{side}_busbar_grid_model_ids"]
    coupler_grid_model_ids = bay_df.loc[coupler_index, f"{side}_coupler_ids"]
    if len(busbar_grid_model_ids) == 1:
        # no selectable busbar -> return the only one
        return busbar_grid_model_ids[0]
    if len(busbar_grid_model_ids) > 1:
        default_busbar_grid_model_id = busbar_grid_model_ids[0]
        if default_busbar_grid_model_id == ignore_busbar_id:
            default_busbar_grid_model_id = busbar_grid_model_ids[1]
        # try to find a busbar with a closed state disconnector
        for busbar_grid_model_id in busbar_grid_model_ids:
            if busbar_grid_model_id == ignore_busbar_id:
                # ignore the busbar that is connected to the other side of the coupler
                continue
            cond_direct_busbar = bay_df["direct_busbar_grid_model_id"] == busbar_grid_model_id
            cond_closed = ~bay_df["open"]
            cond_valid_side = bay_df["grid_model_id"].isin(coupler_grid_model_ids)
            cond_in_of_service = ~bay_df["direct_busbar_grid_model_id"].isin(out_of_service_busbar_ids)
            switches_connected_to_busbar = bay_df[cond_direct_busbar & cond_closed & cond_valid_side & cond_in_of_service]
            if len(switches_connected_to_busbar) > 0:
                return busbar_grid_model_id
            if default_busbar_grid_model_id in out_of_service_busbar_ids:
                switches_connected_to_busbar = bay_df[cond_direct_busbar & cond_valid_side & cond_in_of_service]
                if len(switches_connected_to_busbar) > 0:
                    # gets a busbar that is not out of service and not the ignore busbar
                    default_busbar_grid_model_id = busbar_grid_model_id
    else:
        raise ValueError(f"Coupler has no busbar grid model id. bay_df: {bay_df.to_dict()}")
    # if no closed state disconnector is found, return the first busbar in the list
    return default_busbar_grid_model_id


def get_state_of_coupler_based_on_bay(coupler_index: pd.Index, bay_df: pd.DataFrame) -> bool:
    """Get the state of the coupler, based on the state of the bay.

    This function checks if the coupler is open or closed, based on the state of the switches in the bay_df.
    If there is on either side all switches are open, the coupler is considered open.

    Parameters
    ----------
    coupler_index: int
        Index of the coupler in the dataframe.
    bay_df: pd.DataFrame
        Dataframe with the asset bay switches.
        expects the columns "open", "from_coupler_ids" and "to_coupler_ids" to be present.

    Returns
    -------
    bool
        True if one side has all switches open, False otherwise.
    """
    from_switches = bay_df.loc[coupler_index, "from_coupler_ids"]
    to_switches = bay_df.loc[coupler_index, "to_coupler_ids"]
    open_from_switches_sr = bay_df[
        bay_df["grid_model_id"].isin(from_switches) & (bay_df["direct_busbar_grid_model_id"] != "")
    ]["open"]
    open_to_switches_sr = bay_df[bay_df["grid_model_id"].isin(to_switches) & (bay_df["direct_busbar_grid_model_id"] != "")][
        "open"
    ]
    open_from_switches_sl = bay_df[
        bay_df["grid_model_id"].isin(from_switches) & (bay_df["direct_busbar_grid_model_id"] == "")
    ]["open"]
    open_to_switches_sl = bay_df[bay_df["grid_model_id"].isin(to_switches) & (bay_df["direct_busbar_grid_model_id"] == "")][
        "open"
    ]
    if len(open_from_switches_sr) == 0:
        cond_from = False
    else:
        cond_from = all(open_from_switches_sr)

    if len(open_to_switches_sr) == 0:
        cond_to = False
    else:
        cond_to = all(open_to_switches_sr)

    if len(open_from_switches_sl) != 0:
        cond_from = cond_from or any(open_from_switches_sl)
    if len(open_to_switches_sl) != 0:
        cond_to = cond_to or any(open_to_switches_sl)
    if cond_from or cond_to:
        return True
    return False


def get_switchable_asset(
    busbar_connection_info: dict[str, BusbarConnectionInfo], node_assets_df: NodeAssetSchema, branches_df: BranchSchema
) -> SwitchableAssetSchema:
    """Get the switchable assets from the NetworkGraphData busbar_connection_info dict.

    Parameters
    ----------
    busbar_connection_info: dict[str, BusbarConnectionInfo]
        Dictionary with all busbar connections of the substation.
        expects network_graph.get_busbar_connection_info()
    node_assets_df: NodeAssetSchema
        Dataframe with all nodes of the substation.
        expects get_node_assets_df
    branches_df: BranchSchema
        Dataframe with all branches of the substation.
        expects get_branches_df

    Returns
    -------
    connected_asset_df: pd.DataFrame
        connected_asset_df of the specified substation.
    """
    connected_assets_list = [asset for assets in busbar_connection_info.values() for asset in assets.connectable_assets]
    connected_asset_df = pd.DataFrame(connected_assets_list, columns=["grid_model_id"]).drop_duplicates()
    # merge node_assets_df
    # -> node_assets_df.columns get added ['grid_model_id', 'foreign_id', 'node', 'asset_type', 'in_service']
    connected_asset_df = connected_asset_df.merge(
        node_assets_df[["grid_model_id", "foreign_id", "asset_type", "in_service"]],
        left_on="grid_model_id",
        right_on="grid_model_id",
        how="left",
    )
    # merge branches_df -> branches_df.columns get added with suffix "_1"
    connected_asset_df = connected_asset_df.merge(
        branches_df[["grid_model_id", "foreign_id", "asset_type", "in_service"]],
        left_on="grid_model_id",
        right_on="grid_model_id",
        how="left",
        suffixes=("", "_1"),
    )
    # merge the values of the added columns
    connected_asset_df["asset_type"] = np.where(
        connected_asset_df["asset_type"].notna(), connected_asset_df["asset_type"], connected_asset_df["asset_type_1"]
    )
    connected_asset_df["foreign_id"] = np.where(
        connected_asset_df["foreign_id"].notna(), connected_asset_df["foreign_id"], connected_asset_df["foreign_id_1"]
    )
    connected_asset_df["in_service"] = np.where(
        connected_asset_df["in_service"].notna(), connected_asset_df["in_service"], connected_asset_df["in_service_1"]
    )
    # rename columns to match the AssetTopology
    connected_asset_df.rename(columns={"asset_type": "type", "foreign_id": "name"}, inplace=True)
    connected_asset_df = connected_asset_df[["grid_model_id", "name", "type", "in_service"]]
    # ensure the order of the assets
    connected_asset_df.sort_values(by="grid_model_id", inplace=True)
    connected_asset_df.reset_index(drop=True, inplace=True)
    connected_asset_df["in_service"] = connected_asset_df["in_service"].astype(bool)
    SwitchableAssetSchema.validate(connected_asset_df)
    return connected_asset_df


def get_asset_bay_df(
    switches_df: SwitchSchema,
    asset_grid_model_id: str,
    busbar_df: pd.DataFrame,
    edge_connection_info: dict[str, EdgeConnectionInfo],
) -> pd.DataFrame:
    """Get the asset bay df as preparation for the get_asset_bay().

    Parameters
    ----------
    switches_df: SwitchSchema
        Dataframe with all switches of the substation.
        expects NetworkGraphData.switches
    asset_grid_model_id: str
        Asset grid model id for which the asset bays should be retrieved.
    busbar_df: pd.DataFrame
        Dataframe with all busbars of the substation.
        expects get_busbar_df
    edge_connection_info: dict[str, EdgeConnectionInfo]
        Dictionary with all edge connections of the substation.
        expects network_graph.get_edge_connection_info()

    Returns
    -------
    asset_bay: pd.DataFrame
        AssetBay of the specified asset.
    """
    bay_edge_connection_info = {
        edge_id: edge_info for edge_id, edge_info in edge_connection_info.items() if asset_grid_model_id == edge_info.bay_id
    }
    asset_bays_df = switches_df[(switches_df["grid_model_id"].isin(bay_edge_connection_info.keys()))]
    asset_bays_df["direct_busbar_grid_model_id"] = asset_bays_df["grid_model_id"].map(
        lambda grid_model_id: bay_edge_connection_info[grid_model_id].direct_busbar_grid_model_id
    )
    asset_bays_df = asset_bays_df.merge(
        busbar_df[["grid_model_id", "int_id"]],
        left_on="direct_busbar_grid_model_id",
        right_on="grid_model_id",
        how="left",
        suffixes=("", "_from"),
    )
    return asset_bays_df


def get_sl_switch(asset_bays_df: pd.DataFrame) -> tuple[Optional[str], list[str], int]:
    """Get the sl_switch from the asset bay.

    Parameters
    ----------
    asset_bays_df: pd.DataFrame
        Dataframe with the asset bay switches.
        expects get_get_asset_bay_df

    Returns
    -------
    sl_switch: Optional[str]
        sl_switch of the asset bay.
        or None if no sl_switch is found.
    logs: list[str]
        List of logs that are created during the process.
    n_sl_sw_found: int
        Number of sl switches found in the asset bay.


    Raises
    ------
    ValueError
        If there are multiple sl switches in the asset bay.
    """
    logs = []
    sl_switch = asset_bays_df[
        (asset_bays_df["asset_type"] == "DISCONNECTOR") & (asset_bays_df["direct_busbar_grid_model_id"] == "")
    ]
    n_sl_sw_found = len(sl_switch)
    sl_switch_id = None
    if len(sl_switch) > 1:
        logs.append(
            f"There should be maximum one sl_switch but got {len(sl_switch)} "
            f"with grid_model_id {sl_switch['grid_model_id'].to_list()}"
            f" choosing an open one if available, otherwise the first one."
        )
        sl_switch_open = asset_bays_df[
            (asset_bays_df["asset_type"] == "DISCONNECTOR")
            & (asset_bays_df["direct_busbar_grid_model_id"] == "")
            & asset_bays_df["open"]
        ]
        if len(sl_switch_open) != 0:
            sl_switch_id = sl_switch_open["grid_model_id"].values[0]
        else:
            sl_switch_id = sl_switch["grid_model_id"].values[0]
    if len(sl_switch) == 1:
        sl_switch_id = sl_switch["grid_model_id"].values[0]
    # if len(sl_sw) == 0 -> no sl switch in the asset bay -> is allowed -> no error
    return sl_switch_id, logs, n_sl_sw_found


def get_dv_switch(asset_bays_df: pd.DataFrame, asset_grid_model_id: str) -> str:
    """Get the dv_switch from the asset bay.

    Parameters
    ----------
    asset_bays_df: pd.DataFrame
        Dataframe with the asset bay switches.
        expects get_get_asset_bay_df
    asset_grid_model_id: str
        Asset grid model id for which the asset bays should be retrieved.

    Returns
    -------
    dv_sw_name: str
        dv_switch of the asset bay.
        or "" if no dv_switch is found.
    logs: list[str]
        List of logs that are created during the process.
    n_dv_sw_found: int
        Number of dv switches found in the asset bay.

    Raises
    ------
    ValueError
        If there are multiple dv switches in the asset bay.
    """
    logs = []
    dv_sw = asset_bays_df[(asset_bays_df["asset_type"] == "BREAKER") & (asset_bays_df["direct_busbar_grid_model_id"] == "")]
    n_dv_sw_found = 1
    if len(dv_sw) > 1:
        n_dv_sw_found = len(dv_sw)
        logs.append(
            f"Warning: There should be exactly one dv switch but got '{len(dv_sw)}' "
            f"with grid_model_id {dv_sw['grid_model_id'].to_list()}"
        )
        dv_sw_open = asset_bays_df[
            (asset_bays_df["asset_type"] == "BREAKER")
            & (asset_bays_df["direct_busbar_grid_model_id"] == "")
            & asset_bays_df["open"]
        ]
        if len(dv_sw_open) >= 1:
            dv_sw_grid_model_id = dv_sw_open["grid_model_id"].values[0]
            logs.append(f"Selecting the first open Switch. grid_model_id: {dv_sw_grid_model_id}")
        else:
            dv_sw_grid_model_id = dv_sw["grid_model_id"].values[0]
            logs.append(f"Selecting the first Switch. grid_model_id: {dv_sw_grid_model_id}")
    if len(dv_sw) == 0:
        n_dv_sw_found = 0
        if len(asset_bays_df) > 0:
            grid_model_id = asset_bays_df["grid_model_id"].values[0]
        else:
            grid_model_id = ""
        logs.append(
            "Warning:There should be exactly one dv switch but got '0', dv switch_id is left empty"
            f" for grid_model_id: {asset_grid_model_id},"
            f" grid_model_id of first bay switch: {grid_model_id}"
        )
        dv_sw_grid_model_id = ""
    else:
        dv_sw_grid_model_id = dv_sw["grid_model_id"].values[0]
    return dv_sw_grid_model_id, logs, n_dv_sw_found


def get_sr_switch(asset_bays_df: pd.DataFrame) -> dict[str, str]:
    """Get the sr_switch from the asset bay.

    Parameters
    ----------
    asset_bays_df: pd.DataFrame
        Dataframe with the asset bay switches.
        expects get_get_asset_bay_df

    Returns
    -------
    sr_switch: dict[str, str]
        sr_switch of the asset bay.
        or {} if no sr_switch is found.
    """
    sr_sw = asset_bays_df[
        (asset_bays_df["asset_type"] == "DISCONNECTOR") & (asset_bays_df["direct_busbar_grid_model_id"] != "")
    ]
    return {
        g_id: f_id
        for g_id, f_id in zip(sr_sw["direct_busbar_grid_model_id"].to_list(), sr_sw["grid_model_id"].to_list(), strict=True)
    }


def get_dv_sr_switch(asset_bays_df: pd.DataFrame) -> dict[str, str]:
    """Get the dv_sr_switch from the asset bay.

    This function exists due to data quality issues.
    There are cases where a BREAKER is directly connected to a busbar.
    Sometimes there are BREAKERs instead of sr DISCONNECTORs in the asset bay.
    Note: this function could have been integrated into get_sr_switch,
        but is separated to inform the user about the quality issue.

    Parameters
    ----------
    asset_bays_df: pd.DataFrame
        Dataframe with the asset bay switches.
        expects get_get_asset_bay_df

    Returns
    -------
    dv_sr_dict: dict[str, str]
        dv_sr_switch of the asset bay.
        or {} if no dv_sr_switch is found.
    """
    logs = []
    dv_sr_sw = asset_bays_df[
        (asset_bays_df["asset_type"] == "BREAKER") & (asset_bays_df["direct_busbar_grid_model_id"] != "")
    ]
    if len(dv_sr_sw) >= 1:
        logs.append(
            f"Warning: There is a BREAKER directly connected to a busbar {dv_sr_sw['grid_model_id'].to_list()} "
            "Will be modelled as sr switch."
            f" grid_model_id: {dv_sr_sw['grid_model_id'].values[0]}"
        )
        dv_sr_dict = {
            f_id: g_id
            for f_id, g_id in zip(
                dv_sr_sw["direct_busbar_grid_model_id"].to_list(), dv_sr_sw["grid_model_id"].to_list(), strict=True
            )
        }
    else:
        dv_sr_dict = {}
    return dv_sr_dict, logs


def get_asset_bay(
    switches_df: SwitchSchema,
    asset_grid_model_id: str,
    busbar_df: pd.DataFrame,
    edge_connection_info: dict[str, EdgeConnectionInfo],
) -> Union[AssetBay, None]:
    """Get the asset bay for a asset_grid_model_id.

    Parameters
    ----------
    switches_df: SwitchSchema
        Dataframe with all switches of the substation.
        expects NetworkGraphData.switches
    asset_grid_model_id: str
        Asset grid model id for which the asset bays should be retrieved.
    busbar_df: pd.DataFrame
        Dataframe with all busbars of the substation.
        expects get_busbar_df
    edge_connection_info: dict[str, EdgeConnectionInfo]
        Dictionary with all edge connections of the substation.
        expects network_graph.get_edge_connection_info()

    Returns
    -------
    asset_bay: Union[AssetBay, None]
        AssetBay of the specified asset.
        or None if no AssetBay is found.
    logs: list[str]
        List of logs that are created during the process.

    Raises
    ------
    ValueError
        If the switches found not match the number of switches in the asset bay.

    """
    # get the edge connection info for the asset bay
    asset_bays_df = get_asset_bay_df(
        switches_df=switches_df,
        asset_grid_model_id=asset_grid_model_id,
        busbar_df=busbar_df,
        edge_connection_info=edge_connection_info,
    )
    asset_bay_dict = {}

    sl_switch, sl_log, n_sl_sw_found = get_sl_switch(asset_bays_df)
    if sl_switch is not None:
        # no sl switch in the asset bay -> is allowed -> no error
        asset_bay_dict["sl_switch_grid_model_id"] = sl_switch

    asset_bay_dict["dv_switch_grid_model_id"], dv_logs, n_dv_sw_found = get_dv_switch(
        asset_bays_df=asset_bays_df, asset_grid_model_id=asset_grid_model_id
    )
    asset_bay_dict["sr_switch_grid_model_id"] = get_sr_switch(asset_bays_df=asset_bays_df)
    dv_sr_dict, dv_sr_logs = get_dv_sr_switch(asset_bays_df=asset_bays_df)
    asset_bay_dict["sr_switch_grid_model_id"] = {**asset_bay_dict["sr_switch_grid_model_id"], **dv_sr_dict}

    # check if the number of switches found match the number of switches in the asset bay
    switches_found = n_sl_sw_found + n_dv_sw_found + len(asset_bay_dict["sr_switch_grid_model_id"])
    if switches_found != len(asset_bays_df):
        raise ValueError(f"Expected {len(asset_bays_df)} switches, but got: {asset_bay_dict}")

    logs = []
    logs.extend(sl_log)
    logs.extend(dv_logs)
    logs.extend(dv_sr_logs)
    if len(asset_bay_dict["sr_switch_grid_model_id"]) == 0:
        logs.append(
            "Warning: There should be at least one sr switch but got 0,"
            f" AssetBay ignored for grid_model_id: {asset_grid_model_id}"
        )
        return None, logs
    return AssetBay(**asset_bay_dict), logs


def get_station_connection_tables(
    busbar_connection_info: dict[str, BusbarConnectionInfo],
    busbar_df: pd.DataFrame,
    switchable_assets_df: SwitchableAssetSchema,
) -> tuple[
    Bool[Array, " n_bus n_asset"], Bool[Array, " n_bus n_asset"], Bool[Array, " n_bus n_bus"], Bool[Array, " n_bus n_bus"]
]:
    """Get the switching table physically from the NetworkGraphData busbar_connection_info dict.

    Parameters
    ----------
    busbar_connection_info: dict[str, BusbarConnectionInfo]
        Dictionary with all busbar connections of the substation.
        expects NetworkGraphData.busbar_connection_info
    busbar_df: pd.DataFrame
        Dataframe with all busbars of the substation.
        expects get_busbar_df
    switchable_assets_df: pd.DataFrame
        Dataframe with all switchable assets of the substation.
        expects get_switchable_asset

    Returns
    -------
    asset_connectivity: Bool[Array, " n_bus n_asset"]
        asset_connectivity of the specified substation.
        Holds the all possible layouts of the switching table, shape (n_bus n_asset).
        An entry is true if it is possible to connected an asset to the busbar.
    asset_switching_table: Bool[Array, " n_bus n_asset"]
        Holds the switching of each asset to each busbar, shape (n_bus n_asset).
        An entry is true if the asset is connected to the busbar.
    busbar_connectivity: Bool[Array, " n_bus n_bus"]]
        Holds the all possible layouts of the switching table, shape (n_bus n_bus).
        An entry is true if it is possible to connected an busbar to the busbar.
    busbar_switching_table: Bool[Array, " n_bus n_bus"]]
        Holds the switching of each busbar to each busbar, shape (n_bus n_bus).
        An entry is true if a busbar is connected to the busbar.
    """
    n_bus = busbar_df.shape[0]
    n_asset = switchable_assets_df.shape[0]
    asset_connectivity = np.zeros((n_bus, n_asset), dtype=bool)
    busbar_connectivity = np.zeros((n_bus, n_bus), dtype=bool)
    asset_switching_table = np.zeros((n_bus, n_asset), dtype=bool)
    busbar_switching_table = np.zeros((n_bus, n_bus), dtype=bool)

    for _, row in busbar_df.iterrows():
        # get the asset ids that are connectable to the busbar, not respecting open switches
        asset_grid_model_ids = busbar_connection_info[row["grid_model_id"]].connectable_assets
        asset_ids = switchable_assets_df[switchable_assets_df["grid_model_id"].isin(asset_grid_model_ids)].index.to_list()
        asset_connectivity[row["int_id"], asset_ids] = True

        # get the asset ids that are connectable to the busbar, respecting open switches
        asset_grid_model_ids = busbar_connection_info[row["grid_model_id"]].zero_impedance_connected_assets
        asset_ids = switchable_assets_df[switchable_assets_df["grid_model_id"].isin(asset_grid_model_ids)].index.to_list()
        asset_switching_table[row["int_id"], asset_ids] = True

        # get the busbar ids that are connectable to the busbar, not respecting open switches
        connectable_busbar_grid_model_ids = busbar_connection_info[row["grid_model_id"]].connectable_busbars
        connectable_busbar_ids = busbar_df[
            busbar_df["grid_model_id"].isin(connectable_busbar_grid_model_ids)
        ].index.to_list()
        busbar_connectivity[row["int_id"], connectable_busbar_ids] = True

        # get the busbar ids that are connectable to the busbar, respecting open switches
        connectable_busbar_grid_model_ids = busbar_connection_info[row["grid_model_id"]].zero_impedance_connected_busbars
        connectable_busbar_ids = busbar_df[
            busbar_df["grid_model_id"].isin(connectable_busbar_grid_model_ids)
        ].index.to_list()
        busbar_switching_table[row["int_id"], connectable_busbar_ids] = True

    return asset_connectivity, asset_switching_table, busbar_connectivity, busbar_switching_table


def remove_double_connections(
    switching_table: Bool[Array, " n_bus n_asset"],
    substation_id: Optional[str] = None,
) -> Bool[Array, " n_bus n_asset"]:
    """Remove double connections from the switching table.

    An Asset can be connected to multiple busbars.
    This function removes the double connections, by keeping the first connection and removing the others.

    Parameters
    ----------
    switching_table: Bool[Array, " n_bus n_asset"]
        The switching table with the connections.
    substation_id: Optional[str]
        The substation id for which the switching table is created.
        If given, a warning is logged if double connections are detected.

    Returns
    -------
    switching_table: Bool[Array, " n_bus n_asset"]
        The switching table with the double connections removed.
    """
    results = np.zeros_like(switching_table).T
    for i, col in enumerate(switching_table.T):
        if np.any(col):
            results[i, np.argmax(col)] = True
    switching_table_mod = results.T
    if not np.array_equal(switching_table, switching_table_mod):
        if substation_id is None:
            substation_id = "not_given"
        logger.warning(f"Double connections in the switching table detected and removed. Station: {substation_id}")
    return switching_table_mod
