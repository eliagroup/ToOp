"""Asset Topology helper for UCTE."""

import pandas as pd


def get_bus_info_from_topology(station_busses: pd.DataFrame, bus_id: str) -> pd.DataFrame:
    """Get the info for all busbars that are part of the bus.

    Parameters
    ----------
    station_busses: pd.DataFrame
        Dataframe with all busbars of the station and which bus they are connected to.
        Comes from BusBreakerTopology.buses
    bus_id: str
        Bus id for which the busses should be retrieved.

    Returns
    -------
    station_busses: pd.DataFrame
        DataFrame with the busbars of the specified bus.
        Note: The DataFrame columns are the same as
        interfaces.asset_topology.Busbar.__args__.
    """
    station_busses = station_busses[station_busses["bus_id"] == bus_id].copy()
    # for UCTE model: if not in service, asset will not appear
    station_busses["in_service"] = True

    # get bus df
    station_busses = (
        station_busses.sort_index().reset_index().reset_index().rename(columns={"index": "int_id", "id": "grid_model_id"})
    )
    station_busses = station_busses[["grid_model_id", "name", "int_id", "in_service"]]

    return station_busses
