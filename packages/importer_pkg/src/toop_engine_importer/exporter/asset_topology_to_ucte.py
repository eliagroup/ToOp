# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module containing functions to translate asset topology model to UCT model.

File: asset_topology_to_uct.py
Author:  Benjamin Petrick
Created: 2024-10-22

Note: this module currently ignores the asset_setpoints.
Note: this module currently ignores generator and load reassignments.
"""

from pathlib import Path

import logbook
import pandas as pd
from beartype.typing import Optional, Union
from toop_engine_importer.ucte_toolset.ucte_io import make_ucte, parse_ucte
from toop_engine_interfaces.asset_topology import BusbarCoupler, Station, Topology

logger = logbook.Logger(__name__)
# For parsing the UCTE format we need those colspecs, which are taken from
# https://eepublicdownloads.entsoe.eu/clean-documents/pre2015/publications/ce/otherreports/UCTE-format.pdf
# TODO user specs from ucte_io.py
UCTE_STATUS_CODES = {
    0: {"name": "in_service", "opposite": 8},
    1: {"name": "in_service", "opposite": 9},
    2: {"name": "in_service", "opposite": 7},
    7: {"name": "out_of_service", "opposite": 2},
    8: {"name": "out_of_service", "opposite": 0},
    9: {"name": "out_of_service", "opposite": 1},
}
UCTE_STATUS_CODE_SWITCH = {True: 7, False: 2}


def load_ucte(
    input_uct: Path,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load UCTE file and return its contents as separate dataframes.

    Parameters
    ----------
    input_uct : Path
        Path to the UCTE file.

    Returns
    -------
    preamble : str
        Preamble of the UCTE file.
    nodes : pd.DataFrame
        Nodes of the UCTE file.
    lines : pd.DataFrame
        Lines of the UCTE file.
    trafos : pd.DataFrame
        Transformers of the UCTE file.
    trafo_reg : pd.DataFrame
        Transformer regulation of the UCTE file.
    postamble : str
        Postamble of the UCTE file.
    """
    with open(input_uct, "r") as f:
        ucte_contents = f.read()
    preamble, nodes, lines, trafos, trafo_reg, postamble = parse_ucte(ucte_contents)

    return preamble, nodes, lines, trafos, trafo_reg, postamble


def asset_topo_to_uct(
    asset_topology: Topology,
    grid_model_file_output: Path,
    grid_model_file_input: Optional[Path] = None,
    station_list: Optional[str] = None,
) -> None:
    """Translate asset topology model to UCT and saves the model.

    Parameters
    ----------
    asset_topology : Topology
        Asset topology model to be translated.
        Based on the pydantic model from interfaces.asset_topology
    grid_model_file_output : Path
        Path to save the UCTE file.
    grid_model_file_input : Optional[Path]
        Path to the grid model file. If not provided, the Topology.grid_model_file will be used.
    station_list : Optional[str]
        List of station grid_model_ids to be translated.
        If not provided, all stations in the asset_topology will be translated.

    Raises
    ------
    NotImplementedError
        If asset_topology.asset_setpoints is not None.

    """
    if asset_topology.asset_setpoints is not None:
        raise NotImplementedError("Asset setpoints are not supported yet.")
    if grid_model_file_input is None:
        grid_model_file_input = asset_topology.grid_model_file
    preamble, nodes, lines, trafos, trafo_reg, postamble = load_ucte(grid_model_file_input)
    for station in asset_topology.stations:
        if station_list is not None and station.grid_model_id not in station_list:
            continue
        asset_change_df = pd.DataFrame(get_changes_from_switching_table(station))
        if len(asset_change_df) > 0:
            change_trafos_lines_in_ucte(trafos, asset_change_df)
            change_trafos_lines_in_ucte(lines, asset_change_df)
        coupler_state_df = pd.DataFrame(get_coupler_state_ucte(station.couplers))
        change_busbar_coupler_state(lines, coupler_state_df)

    # handle order of elements in the ucte file
    handle_duplicated_grid_ids(trafos)
    handle_duplicated_grid_ids(lines)

    output_ucte_str = make_ucte(preamble, nodes, lines, trafos, trafo_reg, postamble)
    with open(grid_model_file_output, "w") as f:
        f.write(output_ucte_str)


def change_trafos_lines_in_ucte(ucte_df: pd.DataFrame, change_df: pd.DataFrame) -> pd.DataFrame:
    """Change the 'from' and 'to' columns of the trafos or line DataFrame based on the change_df.

    Parameters
    ----------
    ucte_df : pd.DataFrame
        The ucte trafos or line DataFrame
        Note: The DataFrame should have 'from', 'to', 'order' columns
        Note: The DataFrame is modified in place
    change_df : pd.DataFrame
        The change_df, containing the 'grid_model_id', 'initial_busbar' and 'final_busbar' columns

    Returns
    -------
    pd.DataFrame
        The updated trafos DataFrame
    """
    # Create a new column 'grid_model_id' in the trafos DataFrame
    ucte_df["grid_model_id"] = ucte_df.apply(lambda row: f"{row['from']} {row['to']} {row['order']}", axis=1)
    ucte_df["index"] = ucte_df.index
    # Merge the ucte_df DataFrame with the change_df DataFrame and apply the update_busbars function
    ucte_df_w_changes = ucte_df.merge(change_df, on="grid_model_id", how="inner", suffixes=("", "_change")).set_index(
        "index"
    )

    ucte_df_w_changes_reassign = ucte_df_w_changes[ucte_df_w_changes["final_busbar"].notnull()]
    ucte_df_w_changes_disconnect = ucte_df_w_changes[ucte_df_w_changes["final_busbar"].isnull()]

    # Update 'from' and 'to' columns based on 'initial_busbar' and 'final_busbar'
    if ucte_df_w_changes_reassign.shape[0] > 0:
        ucte_df_w_changes_reassign = ucte_df_w_changes_reassign.apply(update_busbar_name, axis=1)
    if ucte_df_w_changes_disconnect.shape[0] > 0:
        ucte_df_w_changes_disconnect = ucte_df_w_changes_disconnect.apply(disconnect_line_from_ucte, axis=1)
    ucte_df_w_changes = pd.concat([ucte_df_w_changes_reassign, ucte_df_w_changes_disconnect])
    # update ucte_df
    ucte_df.drop(columns=["grid_model_id", "index"], inplace=True)
    ucte_df.loc[ucte_df_w_changes.index, ["from", "to", "status"]] = ucte_df_w_changes[["from", "to", "status"]]


# Update 'from' and 'to' columns based on 'initial_busbar' and 'final_busbar'
def update_busbar_name(row: pd.Series) -> pd.Series:
    """Update 'from' and 'to' columns based on 'initial_busbar' and 'final_busbar'.

    Parameters
    ----------
    row : pd.Series
        A row of the DataFrame

    Returns
    -------
    pd.Series
        The updated row
    """
    row["from"] = row["from"].replace(row["initial_busbar"], row["final_busbar"])
    row["to"] = row["to"].replace(row["initial_busbar"], row["final_busbar"])
    return row


def disconnect_line_from_ucte(line_row: pd.Series) -> pd.Series:
    """Disconnect a line from UCTE.

    Note: a switch is modeled as a line in UCTE.
    To differentiate between a switch and a line, different status codes are used.

    Parameters
    ----------
    line_row : pd.Series
        A row of the line DataFrame from the parse_ucte() function

    Returns
    -------
    pd.Series
        The updated row can be used to update the line DataFrame
    """
    if UCTE_STATUS_CODES[int(line_row["status"])]["name"] == "in_service":
        line_row["status"] = str(UCTE_STATUS_CODES[int(line_row["status"])]["opposite"])
    return line_row


def change_busbar_coupler_state(lines_df: pd.DataFrame, change_df: pd.DataFrame) -> None:
    """Change the 'status' columns of the lines DataFrame based on the change_df.

    Parameters
    ----------
    lines_df : pd.DataFrame
        The lines DataFrame
        Note: The DataFrame should have 'from', 'to', 'status' columns
        Note: The DataFrame is modified in place
    change_df : pd.DataFrame
        The change_df, containing the 'grid_model_id' and 'coupler_state_ucte' columns
    """
    # Create a new column 'grid_model_id' in the lines DataFrame
    lines_df["grid_model_id"] = lines_df.apply(lambda row: f"{row['from']} {row['to']} {row['order']}", axis=1)
    lines_df["index"] = lines_df.index
    # Merge the lines DataFrame with the change_df DataFrame and apply the update_coupler_state function
    lines_df_w_changes = lines_df.merge(change_df, on="grid_model_id", how="inner", suffixes=("", "_change")).set_index(
        "index"
    )
    lines_df_w_changes = lines_df_w_changes.apply(update_coupler_state, axis=1)
    # update lines_df
    lines_df.drop(columns=["grid_model_id", "index"], inplace=True)
    lines_df.loc[lines_df_w_changes.index, "status"] = lines_df_w_changes["status"]


def update_coupler_state(row: pd.Series) -> pd.Series:
    """Update 'status' column based on 'coupler_state_ucte'.

    Parameters
    ----------
    row : pd.Series
        A row of the DataFrame

    Returns
    -------
    pd.Series
        The updated row
    """
    if int(row["status"]) not in UCTE_STATUS_CODE_SWITCH.values():
        initial_status_name = UCTE_STATUS_CODES[int(row["status"])]["name"]
        new_status_name = UCTE_STATUS_CODES[int(row["coupler_state_ucte"])]["name"]
        if initial_status_name != new_status_name:
            logger.warning(
                f"Line '{row['grid_model_id']}' has a status different from 2 or 7 with status: "
                + f"{row['status']}. Trying to switch a none busbar coupler. Status will be changed "
                + f"to {UCTE_STATUS_CODES[int(row['status'])]['opposite']}"
            )
            row["status"] = str(UCTE_STATUS_CODES[int(row["status"])]["opposite"])

    else:
        row["status"] = str(row["coupler_state_ucte"])
    return row


def get_coupler_state_ucte(couplers: BusbarCoupler) -> list[dict[str, Union[str, int]]]:
    """Get coupler ucte state of from a BusbarCoupler.

    Parameters
    ----------
    couplers : BusbarCoupler
        BusbarCoupler object from the asset topology model

    Returns
    -------
    list[dict[str, Union[str, int]]]
        List of dictionaries containing the coupler_name and the state of the coupler in UCTE format
        2: busbar coupler in operation (definition: R=0, X=0, B=0)
        7: busbar coupler out of operation (definition: R=0, X=0, B=0)
    """
    coupler_state_ucte = [  # TODO: make a dataclass for this (code style)
        {
            "grid_model_id": coupler.grid_model_id,
            "coupler_state_ucte": UCTE_STATUS_CODE_SWITCH[coupler.open],
        }
        for coupler in couplers
    ]
    return coupler_state_ucte


def get_changes_from_switching_table(
    station: Station,
) -> list[dict[str, Union[str, None]]]:
    """Get changes from switching table.

    Parameters
    ----------
    station : Station
        Station object with switching table, busbars and assets

    Returns
    -------
    list[dict[str, Union[str, None]]]
        List of tuples with asset name, initial_busbar and final_busbar
        Note: initial_busbar and final_busbar can both be None if asset is disconnected
    """
    switching_table = station.asset_switching_table
    busbar_name_list = [busbar.grid_model_id for busbar in station.busbars]
    asset_list = station.assets
    change_list = []  # TODO: make a dataclass for this (code style)
    # loop over assets -> by column
    for asset_index, asset_in_table in enumerate(switching_table.T):
        asset_name = asset_list[asset_index].grid_model_id
        asset_type = asset_list[asset_index].type
        if asset_in_table.sum() > 1:
            raise ValueError(
                f"Asset {asset_list[asset_index].grid_model_id} is connected to multiple"
                + " busbars. This is not supported for the UCTE format"
            )
        if asset_in_table.sum() == 0:
            # asset is disconnected
            change_list.append(
                {
                    "grid_model_id": asset_name,
                    "initial_busbar": None,
                    "final_busbar": None,
                    "asset_type": asset_type,
                }
            )
            continue
        busbar_initial = [busbar for busbar in busbar_name_list if busbar in asset_name]
        # asset is connected, check if busbar assignment is changed
        for busbar_index, asset_connected in enumerate(asset_in_table):
            if not asset_connected:
                continue
            busbar_name = busbar_name_list[busbar_index]

            if len(busbar_initial) == 0:
                raise ValueError(f"Asset {asset_name} busbar connection is not found, busbar_name_list: {busbar_name_list}")

            if busbar_name not in asset_name:
                if len(busbar_initial) > 1:
                    raise ValueError(
                        f"Asset {asset_name} is connected to multiple busbars within the same station. "
                        + "Asset can not be reassigned."
                    )
                change_list.append(
                    {
                        "grid_model_id": asset_name,
                        "initial_busbar": busbar_initial[0],
                        "final_busbar": busbar_name,
                        "asset_type": asset_type,
                    }
                )

    return change_list


def handle_duplicated_grid_ids(ucte_df: pd.DataFrame) -> None:
    """Handle duplicated grid ids in the ucte file.

    The function will increment the order of the duplicated grid ids by 1.

    Parameters
    ----------
    ucte_df : pd.DataFrame
        The ucte DataFrame to be updated
        Note: The DataFrame should have 'from', 'to', 'order' columns
        Note: The DataFrame is modified in place
    """
    ucte_df["grid_model_id"] = ucte_df.apply(lambda row: f"{row['from']} {row['to']} {row['order']}", axis=1)
    ucte_df["index"] = ucte_df.index
    duplicated_ids = ucte_df[ucte_df["grid_model_id"].duplicated(keep="first")]
    run_count = 0
    max_runs = 20
    while duplicated_ids.shape[0] > 0:
        # Find duplicated grid_model_ids
        duplicated_ids["order"] = duplicated_ids.apply(lambda row: f"{int(row['order']) + 1}", axis=1)
        duplicated_ids.set_index("index")
        ucte_df.loc[duplicated_ids.index, "order"] = duplicated_ids["order"]

        # set new grid_model_id
        ucte_df["grid_model_id"] = ucte_df.apply(lambda row: f"{row['from']} {row['to']} {row['order']}", axis=1)
        duplicated_ids = ucte_df[ucte_df["grid_model_id"].duplicated(keep="first")]

        run_count += 1
        if run_count > max_runs:
            raise ValueError("Duplicated grid_model_ids could not be resolved. More than 20 iterations have been reached.")
    ucte_df.drop(columns=["grid_model_id", "index"], inplace=True)
