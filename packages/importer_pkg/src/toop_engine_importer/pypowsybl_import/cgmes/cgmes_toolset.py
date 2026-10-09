# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Additional functions for the pypowsybl library, related to CGMES data."""

import pandas as pd
from beartype.typing import Literal, Optional
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_helpers import get_voltage_level_with_region


def get_region_for_df(
    network: Network,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Get the region for each element in a DataFrame.

    This function retrieves the region for each element in a DataFrame using the substation region.

    Parameters
    ----------
    network: Network
        The network for which the regions should be retrieved.
    df: pd.DataFrame
        The DataFrame for which the regions should be retrieved.
        The df can be any type of element DataFrame, e.g. line, bus, transformer, switch, etc.
        Note: "region" will be added for "voltage_level_id"
              "region_1" and "region_2" will be added for "voltage_level1_id" and "voltage_level2_id"


    Returns
    -------
    pd.DataFrame
        A DataFrame with the elements and their regions.
        added columns:  "region" for "voltage_level_id"
                        "region_1" and "region_2" for "voltage_level1_id" and "voltage_level2_id"
    """
    voltage_level = get_voltage_level_with_region(network, attributes=["region"])
    id_columns = ["voltage_level_id", "voltage_level1_id", "voltage_level2_id"]
    for id_column in id_columns:
        if id_column in df.columns:
            df = df.merge(voltage_level, left_on=id_column, right_on="id", how="left", suffixes=("_1", "_2")).set_index(
                df.index
            )
    return df


def get_busbar_sections_with_in_service(
    network: Network, attributes: Optional[list[str]] = None, all_attributes: Optional[Literal[True, False]] = None
) -> pd.DataFrame:
    """Get the busbar sections with their in_service status.

    This function is an extension to the network.get_busbar_sections() function. It matches the signature of the
    network.get_busbar_sections() function, and adds the in_service status to the busbar sections.
    "in_service" is a standard column to be returned by this function.

    Get an in_service status by checking the connected_component and the connected status of the busbar section.
    The column "in_service" is defined as follows:
    - busbar section is connected (connected column is True)
    - busbar section is connected to the main grid (connected_component = 0)

    Parameters
    ----------
    network: Network
        The network for which the busbar sections should be retrieved.
    attributes: Optional[list[str]]
        The attributes that should be retrieved for the busbar sections.
        Behaves like the attributes parameter in network.get_busbar_sections().
    all_attributes: Optional[Union[True,False]]
        If True, all attributes are retrieved for the busbar sections.
        Behaves like the all_attributes parameter in network.get_busbar_sections().

    Returns
    -------
    pd.DataFrame
        A DataFrame with the busbar sections and their in_service status.
    """
    if attributes is not None and all_attributes is not None:
        raise ValueError("Only one of 'attributes' and 'all_attributes' can be specified")

    # missing "in_service" attribute -> default function
    if attributes is not None and "in_service" not in attributes:
        return network.get_busbar_sections(attributes=attributes)

    if all_attributes:
        busbar_sections = network.get_busbar_sections(all_attributes=True)
        attributes = list(busbar_sections.columns)
    elif attributes is None:
        busbar_sections = network.get_busbar_sections()
        attributes = list(busbar_sections.columns)
    elif attributes == ["in_service"]:
        busbar_sections = network.get_busbar_sections()
    else:
        # to be able to merge with the buses, we need to add the bus_id
        # to assess the in_service status, we need the connected status
        attributes_merge = attributes
        if "bus_id" not in attributes:
            attributes_merge = [*attributes_merge, "bus_id"]
        if "connected" not in attributes_merge:
            attributes_merge = [*attributes_merge, "connected"]
        if "in_service" in attributes_merge:
            attributes_merge.remove("in_service")
        busbar_sections = network.get_busbar_sections(attributes=attributes_merge)
    # set default value for in_service
    busbar_sections["in_service"] = True

    # get bus information
    buses = network.get_buses(attributes=["connected_component"])
    busbar_sections = busbar_sections.merge(
        buses, left_on="bus_id", right_index=True, how="left", suffixes=("", "_bus")
    ).set_index(busbar_sections.index)

    # set in_service to False if connected_component is not in service
    not_connected_to_main_grid = busbar_sections["connected_component"] != 0
    not_in_service_condition = not_connected_to_main_grid | ~busbar_sections["connected"]
    busbar_sections.loc[not_in_service_condition, "in_service"] = False

    # get attributes columns
    if "in_service" not in attributes:
        attributes = [*attributes, "in_service"]
    busbar_sections = busbar_sections[attributes]

    return busbar_sections  # ty: ignore[unsound-return-statement] # pandas column selection typed as Unknown by ty
