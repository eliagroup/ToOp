# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Asset Topology helper for CGMES."""

import pandas as pd
from pypowsybl.network.impl.network import Network


def get_bus_info_from_topology(network: Network, station_elements: pd.DataFrame) -> pd.DataFrame:
    """Get the info for all busbars that are part of the bus.

    Parameters
    ----------
    network: Network
        The powsybl network object
    station_elements: pd.DataFrame
        Dataframe with elements, including BUSBAR_SECTION
        Comes from BusBreakerTopology.elements

    Returns
    -------
    station_busses: pd.DataFrame
        DataFrame with the busbars of the specified bus.
        Note: The DataFrame columns are the same as in the pydantic model.
    """
    station_busses = station_elements[station_elements["type"] == "BUSBAR_SECTION"].copy().sort_index()
    station_busses["grid_model_id"] = station_busses.index
    station_busses = station_busses.merge(
        network.get_busbar_sections()[["name", "connected"]], left_on="id", right_on="id", how="left"
    )

    # get bus df
    station_busses = (
        station_busses.reset_index().reset_index().rename(columns={"index": "int_id", "connected": "in_service"})
    )
    station_busses = station_busses[["grid_model_id", "name", "int_id", "in_service", "bus_id"]]

    return station_busses
