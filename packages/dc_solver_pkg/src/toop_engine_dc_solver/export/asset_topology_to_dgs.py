"""Module containing functions to translate asset topology model to the DGS format.

File: asset_topology_to_dgs.py
Author:  Benjamin Petrick
Created: 2024-11-12

Note: this module currently ignores the asset_setpoints.
"""

import io
from copy import deepcopy

import logbook
import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
from beartype.typing import Optional
from pypowsybl.network import Network
from toop_engine_dc_solver.export.dgs_v7_definitions import (
    DGS_GENERAL_SHEET_CONTENT_FID,
    DGS_GENERAL_SHEET_CONTENT_FID_CIM,
    DGS_SHEETS,
    DgsElmCoupSchema,
    DgsGeneralSchema,
)
from toop_engine_interfaces.asset_topology import (
    BusbarCoupler,
    PowsyblSwitchValues,
    Station,
    Topology,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model

logger = logbook.Logger(__name__)


class SwitchUpdateSchema(pa.DataFrameModel):
    """Schema for the switch update DataFrame."""

    grid_model_id: pat.Series[str] = pa.Field(coerce=True, nullable=True)
    """ The grid_model_id to be updated."""

    open: pat.Series[bool] = pa.Field(coerce=True, nullable=True)
    """ The value to be set for the switch, True for open, False for closed."""


class ForeignIdSchema(pa.DataFrameModel):
    """Schema for the switch update DataFrame."""

    grid_model_id: pat.Series[str] = pa.Field(coerce=True)
    """ The grid_model_id to be updated."""

    foreign_id: pat.Series[str] = pa.Field(coerce=True)
    """ The foreign id e.g. from PowerFactory."""


@pa.check_types
def switch_update_schema_to_dgs(
    switch_update_schema: pat.DataFrame[SwitchUpdateSchema],
    foreign_ids: Optional[pat.DataFrame[ForeignIdSchema]] = None,
    cim: bool = True,
) -> pat.DataFrame[DgsElmCoupSchema]:
    """Translate the switch update schema to the DGS format.

    Provide a ForeignIdSchema to update translate the grid_model_id to a foreign id.

    Parameters
    ----------
    switch_update_schema : pat.DataFrame[SwitchUpdateSchema]
        The switch update schema to be translated
    foreign_ids: Optional[pat.DataFrame[ForeignIdSchema]]
        The foreign ids to be used for the translation
    cim: bool, default True
        If True, the ids are expected to be in the CIM format -> a missing underscore at the beginning of the id.
        and RDF_ID is used as the foreign key.
        If False, the ids are expected to be in the DGS FID format.

    Returns
    -------
    switch_update_schema: pat.DataFrame[DgsElmCoupSchema]
        The switch update schema in the DGS format
    """
    dgs_df = deepcopy(switch_update_schema)
    if foreign_ids is not None:
        dgs_df = dgs_df.merge(
            foreign_ids,
            left_on="grid_model_id",
            right_on="grid_model_id",
            how="left",
        )
        assert dgs_df["foreign_id"].notna().all(), "Not all grid_model_ids have a foreign id"
        dgs_df.rename(columns={"foreign_id": "FID(a:40)", "open": "on_off"}, inplace=True)
        dgs_df.drop(columns=["grid_model_id"], inplace=True)
    else:
        # if no foreign ids are given, we use the grid_model_id as the FID(a:40)
        dgs_df.rename(columns={"grid_model_id": "FID(a:40)", "open": "on_off"}, inplace=True)

    dgs_df["OP"] = "U"
    # dgs on_off is inverted to the powsybl model
    # dgs: 0 for open, 1 for closed
    dgs_df["on_off"] = ~dgs_df["on_off"]
    dgs_df = dgs_df.astype({"FID(a:40)": str, "OP": str, "on_off": int})
    if cim:
        dgs_df["FID(a:40)"] = "_" + dgs_df["FID(a:40)"]
    return dgs_df


@pa.check_types
def get_dgs_general_schema(
    cim: bool = True,
    general_info: Optional[list[dict[str, str]]] = None,
) -> pat.DataFrame[DgsGeneralSchema]:
    """Get the DGS General schema."""
    if general_info is None and cim:
        general_info = DGS_GENERAL_SHEET_CONTENT_FID_CIM
    elif general_info is None and not cim:
        general_info = DGS_GENERAL_SHEET_CONTENT_FID

    df_general = pd.DataFrame(general_info)
    return df_general


@pa.check_types
def switch_dgs_schema_to_xlsx(
    switch_dgs_schema: pat.DataFrame[DgsElmCoupSchema],
    df_general: pat.DataFrame[DgsGeneralSchema],
    file_name: str,
    sheet_name: DGS_SHEETS = "ElmCoup",
) -> None:
    """Write the DGS information to an xlsx file.

    This is the DGS dump function for the switch update schema.
    Consider only dumping changes to the network, not the whole network.
    use get_changing_switches_from_topology() to get the changes.

    Parameters
    ----------
    switch_dgs_schema : pat.DataFrame[DgsElmCoupSchema]
        The switch update schema in the DGS format
    df_general : pat.DataFrame[DgsGeneralSchema]
        The general information for the DGS format
    file_name : str
        Name of the xlsx file
    sheet_name : DGS_SHEETS
        Name of the sheet in the xlsx file
        The DGS format uses predefined sheet names defined in DGS_SHEETS

    Returns
    -------
    None
    """
    with pd.ExcelWriter(file_name) as writer:
        df_general.to_excel(writer, index=False, sheet_name="General")
        switch_dgs_schema.to_excel(writer, index=False, sheet_name=sheet_name)


@pa.check_types
def switch_dgs_schema_to_bytes_io(
    switch_dgs_schema: pat.DataFrame[DgsElmCoupSchema],
    df_general: pat.DataFrame[DgsGeneralSchema],
    sheet_name: DGS_SHEETS = "ElmCoup",
) -> io.BytesIO:
    """Write the DGS information to a BytesIO object.

    This is the DGS dump function for the switch update schema.
    Consider only dumping changes to the network, not the whole network.
    use get_changing_switches_from_topology() to get the changes.

    Parameters
    ----------
    switch_dgs_schema : pat.DataFrame[DgsElmCoupSchema]
        The switch update schema in the DGS format
    df_general : pat.DataFrame[DgsGeneralSchema]
        The general information for the DGS format
    sheet_name : DGS_SHEETS
        Name of the sheet in the xlsx file
        The DGS format uses predefined sheet names defined in DGS_SHEETS

    Returns
    -------
    bytes_io: io.BytesIO
        BytesIO object containing the xlsx file with the DGS information
    """
    bytes_io = io.BytesIO()
    with pd.ExcelWriter(bytes_io, engine="openpyxl") as writer:
        df_general.to_excel(writer, index=False, sheet_name="General")
        switch_dgs_schema.to_excel(writer, index=False, sheet_name=sheet_name)
    bytes_io.seek(0)
    return bytes_io


@pa.check_types
def get_coupler_states_from_busbar_couplers(station_couplers: list[BusbarCoupler]) -> pat.DataFrame[SwitchUpdateSchema]:
    """Translate the coupler states to the SwitchUpdateSchema format.

    Parameters
    ----------
    station_couplers : list[BusbarCoupler]
        List of BusbarCoupler objects containing the coupler states

    Returns
    -------
    switch_df: pat.DataFrame[SwitchUpdateSchema]
        SwitchUpdateSchema object containing the switch state of every coupler.
    """
    switch_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    for coupler in station_couplers:
        if not coupler.in_service:
            raise ValueError(f"Coupler {coupler.grid_model_id} is not in service, undefined behavior")
        switch_df.loc[switch_df.shape[0]] = {
            "grid_model_id": coupler.grid_model_id,
            "open": coupler.open,
        }
    switch_df = switch_df.astype({"grid_model_id": str, "open": bool})
    return switch_df


@pa.check_types
def get_asset_switch_states_from_station(
    station: Station,
) -> tuple[pat.DataFrame[SwitchUpdateSchema], pat.DataFrame[SwitchUpdateSchema]]:
    """Translate the asset switch states to the SwitchUpdateSchema format.

    Parameters
    ----------
    station : Station
        Station object containing the asset switch states

    Returns
    -------
    switch_reassignment_df: pat.DataFrame[SwitchUpdateSchema]
        SwitchUpdateSchema object containing the switch state of every asset that is reassigned.
    switch_disconnection_df: pat.DataFrame[SwitchUpdateSchema]
        SwitchUpdateSchema object containing the switch state of every asset that is disconnected.
    """
    switch_reassignment_list = []
    switch_disconnection_list = []
    busbar_id_dict = {index: busbar.grid_model_id for index, busbar in enumerate(station.busbars)}
    # busbar_grid_model_id_dict = {busbar.grid_model_id: index for index, busbar in enumerate(station.busbars)}
    asset_reassignment_list = get_asset_bay_grid_model_id_list(station)
    assert station.asset_switching_table.shape[1] == len(asset_reassignment_list), (
        "The asset switching table has a different number of columns than the asset reassignment list. "
        f"Columns: {station.asset_switching_table.shape[1]}, Reassignment list: {len(asset_reassignment_list)}"
    )

    for column in range(station.asset_switching_table.shape[1]):
        # a column translates to the index of the station.assets list
        if asset_reassignment_list[column] is None:
            # no asset bay for this column, hence no switch
            continue
        asset_switch_states = station.asset_switching_table[:, column]
        if asset_switch_states.sum() == 1:
            # reassign
            assigned_busbar = np.where(asset_switch_states)[0][0]
            for busbar, switch_id in asset_reassignment_list[column].items():
                if busbar_id_dict[assigned_busbar] == busbar:
                    set_value_switch = PowsyblSwitchValues.CLOSED.value
                else:
                    set_value_switch = PowsyblSwitchValues.OPEN.value
                switch_reassignment_list.append(
                    {
                        "grid_model_id": switch_id,
                        "open": set_value_switch,
                    }
                )
        elif asset_switch_states.sum() == 0:
            # disconnect
            switch_disconnection_list.append(
                {
                    "grid_model_id": station.assets[column].asset_bay.dv_switch_grid_model_id,
                    "open": PowsyblSwitchValues.OPEN.value,
                }
            )
        else:
            raise ValueError(
                f"Switching table column {column} has more than one True value: {station.asset_switching_table[:, column]}"
            )
    switch_reassignment_df = pd.DataFrame.from_records(switch_reassignment_list, columns=["grid_model_id", "open"])
    switch_disconnection_df = pd.DataFrame.from_records(switch_disconnection_list, columns=["grid_model_id", "open"])
    switch_reassignment_df = switch_reassignment_df.astype({"grid_model_id": str, "open": bool})
    switch_disconnection_df = switch_disconnection_df.astype({"grid_model_id": str, "open": bool})
    return switch_reassignment_df, switch_disconnection_df


def get_asset_bay_grid_model_id_list(station: Station) -> list[dict[str, str]]:
    """Get the list of asset bay grid_model_ids for the given station.

    Note: This list only contains busbars where there is a switch.
          Hence the dict might have different lengths for different assets within one station.

    Parameters
    ----------
    station : Station
        The station object containing the asset bays.

    Returns
    -------
    list[dict[str, str]]
        List of dictionaries containing the grid_model_ids for the asset bays
        The keys are the busbar ids and the values are the grid_model_ids.
        The list is empty if no asset bays are found.
    """
    asset_bays = [asset.asset_bay for asset in station.assets]
    sr_switch_grid_model_id_list = []
    for asset_bay in asset_bays:
        if asset_bay is None or asset_bay.sr_switch_grid_model_id is None:
            sr_switch_grid_model_id_list.append(None)
        else:
            sr_switch_grid_model_id_list.append(asset_bay.sr_switch_grid_model_id)
    return sr_switch_grid_model_id_list


def get_busbar_lookup(station: Station) -> dict[int, str]:
    """Get the busbar lookup for the given station.

    Parameters
    ----------
    station : Station
        The station object containing the busbars.

    Returns
    -------
    dict[int, str]
        Dictionary containing the busbar int ids and their corresponding grid_model_id.
        The keys are the busbar int ids and the values are the grid_model_id.
        The dictionary is empty if no busbars are found.
    """
    return {index: busbar.grid_model_id for index, busbar in enumerate(station.busbars)}


@pa.check_types
def get_switch_update_schema_from_topology(topology: Topology) -> pat.DataFrame[SwitchUpdateSchema]:
    """Translate a topology to the SwitchUpdateSchema format.

    Parameters
    ----------
    topology : Topology
        Topology object containing the topology information

    Returns
    -------
    switch_update_df: pat.DataFrame[SwitchUpdateSchema]
        SwitchUpdateSchema object containing the Switch states information for the topology
        No filtering is done, all switches are included.
    """
    switch_df = get_empty_dataframe_from_model(SwitchUpdateSchema)

    for station in topology.stations:
        coupler_df = get_coupler_states_from_busbar_couplers(station.couplers)
        switch_reassignment_df, switch_disconnection_df = get_asset_switch_states_from_station(station)
        # concatenate the two dataframes
        # this order of concatenation is preferred for readability of the dgs format later on
        # -> all coupler of a station are in one block
        switch_df_update = pd.concat([coupler_df, switch_reassignment_df, switch_disconnection_df], ignore_index=True)
        switch_df = pd.concat([switch_df, switch_df_update], ignore_index=True)

    # check for duplicates
    if switch_df.duplicated(subset=["grid_model_id"]).any():
        logger.warning(
            "Duplicate switch ids found in the switch update schema"
            f" {switch_df[switch_df.duplicated(subset=['grid_model_id'])]['grid_model_id'].to_list()}"
        )
        switch_df = switch_df.drop_duplicates(subset=["grid_model_id"])
    switch_df = switch_df.astype({"grid_model_id": str, "open": bool})
    return switch_df


@pa.check_types
def get_diff_switch_states(
    network: Network, switch_df: pat.DataFrame[SwitchUpdateSchema]
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get the diff switch states for the given network and switch update schema.

    Filters the SwitchUpdateSchema to only include switches that have a different state to the current network state.

    Parameters
    ----------
    network : Network
        The network object containing the network information.
    switch_df : pat.DataFrame[SwitchUpdateSchema]
        The switch update schema containing the switch states.

    Returns
    -------
    diff_switch_df: pat.DataFrame[SwitchUpdateSchema]
        SwitchUpdateSchema object containing the diff switch states.

    Raises
    ------
    ValueError
        If the switch id is not found in the network.
    """
    # get the diff between the current switch states and the switch update schema
    diff_switch_df = switch_df.merge(
        network.get_switches(attributes=["open"]),
        left_on="grid_model_id",
        right_index=True,
        how="left",
        suffixes=("", "_network"),
    )
    if diff_switch_df["open_network"].isna().any():
        raise ValueError(
            "Switch id not found in the networkSwitch id: "
            f"{diff_switch_df.loc[diff_switch_df['open_network'].isna(), 'grid_model_id']}"
        )
    # filter out the rows where the switch states are equal
    diff_switch_df = diff_switch_df[diff_switch_df["open"] != diff_switch_df["open_network"]]
    diff_switch_df = diff_switch_df[["grid_model_id", "open"]]
    diff_switch_df = diff_switch_df.astype({"grid_model_id": str, "open": bool})
    return diff_switch_df


@pa.check_types
def get_changing_switches_from_topology(network: Network, target_topology: Topology) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get the changing switches from the topology and network.

    Gets all switches from a Topology and compares them to the current network state.
    Returns a filtered SwitchUpdateSchema containing only the switches that have a different state in the network.

    Parameters
    ----------
    network : Network
        The network object containing the current network information.
    target_topology : Topology
        The target topology object containing the target topology information.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        SwitchUpdateSchema object containing the changing switches.
    """
    switch_update_df = get_switch_update_schema_from_topology(topology=target_topology)
    switch_update_df = get_diff_switch_states(network, switch_update_df)
    return switch_update_df
