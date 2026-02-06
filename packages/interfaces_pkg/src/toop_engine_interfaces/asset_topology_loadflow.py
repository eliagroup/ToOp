# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides an interface for storing loadflow results in the asset topology.

It inherits from the asset_topology interface but adds additional fields for storing loadflow
results.
"""

import math

from beartype.typing import Callable, Optional
from pydantic import field_validator
from toop_engine_interfaces.asset_topology import (
    Busbar,
    Station,
    SwitchableAsset,
    Topology,
)


class SwitchableAssetWithLF(SwitchableAsset):
    """A switchable asset with additional fields for storing loadflow results.

    All fields are optional because json does not support nan/inf values.
    If a value is nan/inf, it will be converted to None by the field_validtators
    """

    p: Optional[float]
    """The active power flow over the asset in MW."""

    q: Optional[float]
    """The reactive power flow over the asset in MVar."""

    i: Optional[float]
    """The current flow over the asset in kA."""

    i_max: Optional[float]
    """The maximum current allowed over the asset, if present. For assets without a current
    limit, this is None"""

    @field_validator("p", "q", "i", "i_max")
    @classmethod
    def convert_nan(cls, value: float) -> Optional[float]:
        """Replace nan/inf values with None

        Parameters
        ----------
        value : float
            The value to check for nan/inf

        Returns
        -------
        Optional[float]
            The value, or None if it was nan/inf
        """
        if value is None or math.isnan(value) or math.isinf(value):
            return None
        return value


class BusbarWithLF(Busbar):
    """A busbar with additional fields for storing loadflow results."""

    va: Optional[float]
    """The voltage angle at the busbar in degrees"""

    vm: Optional[float]
    """The voltage magnitude at the busbar in kV"""

    @field_validator("va", "vm")
    @classmethod
    def convert_nan(cls, value: float) -> Optional[float]:
        """Replace nan/inf values with None

        Parameters
        ----------
        value : float
            The value to check for nan/inf

        Returns
        -------
        Optional[float]
            The value, or None if it was nan/inf
        """
        if value is None or math.isnan(value) or math.isinf(value):
            return None
        return value


class StationWithLF(Station):
    """A station with additional fields for storing loadflow results."""

    busbars: list[BusbarWithLF]
    """The busbars, overloaded from station replacing the Busbar class with BusbarWithLF"""

    assets: list[SwitchableAssetWithLF]
    """The assets, overloaded from station replacing the SwitchableAsset class with
    SwitchableAssetWithLF"""


class TopologyWithLF(Topology):
    """A topology with additional fields for storing loadflow results."""

    stations: list[StationWithLF]
    """The stations, overloaded from topology replacing the Station class with StationWithLF"""


def map_loadflow_results_station(
    station: Station,
    node_extractor: Callable[[Busbar], tuple[Optional[float], Optional[float]]],
    asset_extractor: Callable[
        [SwitchableAsset],
        tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
    ],
) -> StationWithLF:
    """Map loadflow results onto a station without loadflows.

    This also converts nan/inf values to None to be compatible with json serialization

    Parameters
    ----------
    station : Station
        The station to map loadflow results onto, using the plain asset_topology classes
    node_extractor : Callable[[Busbar], tuple[Optional[float], Optional[float]]]
        A function that extracts voltage angle and voltage magnitude for a busbar from some loadflow
        results table. If any of the values can not be extracted, the extractor is free to return None
    asset_extractor : Callable[[SwitchableAsset], tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
        A function that extracts active power, reactive power, current and maximum current for
        an asset from some loadflow results table. If any of the values can not be extracted, the
        extractor is free to return None

    Returns
    -------
    StationWithLF
        The station with loadflow results mapped onto it
    """
    busbars = []
    for busbar in station.busbars:
        if not busbar.in_service:
            busbars.append(
                BusbarWithLF(
                    **busbar.model_dump(),
                    vm=None,
                    va=None,
                )
            )
        else:
            va, vm = node_extractor(busbar)
            busbars.append(
                BusbarWithLF(
                    **busbar.model_dump(),
                    vm=vm,
                    va=va,
                )
            )

    assets = []
    for asset in station.assets:
        if not asset.in_service:
            assets.append(
                SwitchableAssetWithLF(
                    **asset.model_dump(),
                    p=None,
                    q=None,
                    i=None,
                    i_max=None,
                )
            )
        else:
            p, q, i, i_max = asset_extractor(asset)
            assets.append(
                SwitchableAssetWithLF(
                    **asset.model_dump(),
                    p=p,
                    q=q,
                    i=i,
                    i_max=i_max,
                )
            )

    return StationWithLF(
        **station.model_dump(exclude=["busbars", "assets"]),
        busbars=busbars,
        assets=assets,
    )


def map_loadflow_results_topology(
    topology: Topology,
    node_extractor: Callable[[Busbar], tuple[Optional[float], Optional[float]]],
    asset_extractor: Callable[
        [SwitchableAsset],
        tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
    ],
) -> TopologyWithLF:
    """Map loadflow results onto a topology without loadflows

    This also converts nan/inf values to None to be compatible with json serialization

    Parameters
    ----------
    topology : Topology
        The topology to map loadflow results onto, using the plain asset_topology classes
    node_extractor : Callable[[Busbar], tuple[Optional[float], Optional[float]]]
        A function that extracts voltage angle and voltage magnitude for a busbar from some loadflow
        results table. If any of the values can not be extracted, the extractor is free to return None
    asset_extractor : Callable[[SwitchableAsset], tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
        A function that extracts active power, reactive power, current and maximum current for
        an asset from some loadflow results table. If any of the values can not be extracted, the
        extractor is free to return None

    Returns
    -------
    TopologyWithLF
        The topology with loadflow results mapped onto it
    """
    stations = []
    for station in topology.stations:
        stations.append(map_loadflow_results_station(station, node_extractor, asset_extractor))

    return TopologyWithLF(**topology.model_dump(exclude=["stations"]), stations=stations)
