# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Merge UCTE and CGMES network models.

Steps to merge the UCTE and CGMES network models:
1. Get the border tie lines between the UCTE and CGMES network.
2. Remove the tie lines from the UCTE network model.
3. Remove the CGMES area from the UCTE network model.
4. Merge the UCTE and CGMES network model.

The merge only works if the border lines are represented as dangling lines in both network models.
E.g. inner German will not merge with this function. Convert the inner German lines to dangling lines first.

"""

from typing import Optional

import logbook
import pandas as pd
import pandera as pa
import pandera.typing as pat
from pydantic import BaseModel, Field, model_validator
from pypowsybl.network.impl.network import Network
from typing_extensions import Self

logger = logbook.Logger(__name__)


class TieLineSchema(pa.DataFrameModel):
    """A Schema for net.get_tie_lines()."""

    pairing_key: pat.Index[str]
    """The pairing key of the tie lines."""

    dangling_line1_id: pat.Series[str]
    """The first dangling line id of the tie lines."""

    dangling_line2_id: pat.Series[str]
    """The second dangling line id of the tie lines."""


class DanglingLineSchema(pa.DataFrameModel):
    """A Schema for net.get_dangling_lines."""

    pairing_key: pat.Index[str]
    """The pairing key of the tie lines."""

    connected: pat.Series[bool]
    """The connection status of the tie lines."""

    paired: pat.Series[bool]
    """The pairing status of the tie lines."""

    tie_line_id: pat.Series[str]
    """The tie line id of the tie lines."""


class UcteCgmesMerge(BaseModel):
    """The border line merge configuration and information."""

    ucte_area_name: Optional[str]
    """ The name of the UCTE area to remove.
    If the UCTE area name is not provided, the country name must be provided.
    will look for net.get_substations().index.str.startswith(ucte_area_name)
    # """
    country_name: Optional[str]
    """ The name of the country to remove.
    If the country name is not provided, the UCTE area name must be provided.
    id for countries from net_ucte.get_areas().index
    """

    ucte_border_lines: pat.DataFrame[TieLineSchema]
    """ The expected border lines between the UCTE and CGMES network.
    Note: contains the filtered DataFrame from net_ucte.get_tie_lines()
    """

    cgmes_dangling_lines: pat.DataFrame[DanglingLineSchema]
    """ All dangling lines from the CGMES grid model
    net_cgmes.get_dangling_lines()
    """

    removed_tie_lines: Optional[list[str]] = Field(default_factory=list)
    """ The removed tie lines from the ucte file."""

    removed_dangling_lines: Optional[list[str]] = Field(default_factory=list)
    """ removed dangling lines from the ucte file.
    Can exceed the removed tie lines, if the dangling line was not connected.
    """

    statistics: Optional[dict] = Field(default_factory=dict)
    """ The statistics of the merge quality."""

    @model_validator(mode="after")
    def validate_remove_area(self) -> Self:
        """Validate ucte_area_name and country_name."""
        if not self.ucte_area_name and not self.country_name:
            raise ValueError("Either ucte_area_name or country_name must be set.")
        return self

    @model_validator(mode="after")
    def validate_dataframes(self) -> Self:
        """Validate the Dataframes model."""
        # validate the input data
        self.cgmes_dangling_lines = DanglingLineSchema.validate(self.cgmes_dangling_lines)
        self.ucte_border_lines = TieLineSchema.validate(self.ucte_border_lines)
        return self


def run_merge_ucte_cgmes(
    net_cgmes: Network, net_ucte: Network, ucte_area_name: Optional[str] = None, country_name: Optional[str] = None
) -> tuple[Network, UcteCgmesMerge]:
    """Merge the UCTE and CGMES network model.

    The merge is performed by removing the given area from the UCTE network model and
    merging the UCTE and CGMES network model.

    Parameters
    ----------
    net_ucte : Network
        The UCTE network model.
        Note: is modified in place.
    net_cgmes : Network
        The CGMES network model.
        Note: is modified in place.
    ucte_area_name : Optional[str]
        The name of the UCTE area to remove.
        If the UCTE area name is not provided, the country name must be provided.
        will look for net.get_substations().index.str.startswith(ucte_area_name)
    country_name : Optional[str]
        The name of the country to remove.
        If the country name is not provided, the UCTE area name must be provided.
        id for countries from net_ucte.get_areas().index

    Returns
    -------
    net_merged : Network
        The merged network model
    ucte_cgmes_merge_info : UcteCgmesMerge
        The merge information to store the removed tie lines and dangling lines.
    """
    ucte_cgmes_merge_info = UcteCgmesMerge(
        ucte_area_name=ucte_area_name,
        country_name=country_name,
        ucte_border_lines=get_ucte_border_tie_lines(net_ucte=net_ucte, net_cgmes=net_cgmes),
        cgmes_dangling_lines=net_cgmes.get_dangling_lines(),
        removed_tie_lines=[],
        removed_dangling_lines=[],
    )

    remove_area_from_ucte(net_ucte=net_ucte, ucte_cgmes_merge_info=ucte_cgmes_merge_info)
    net_merged = merge_ucte_and_cgmes(net_ucte=net_ucte, net_cgmes=net_cgmes)
    validate_merge_quality(net_merged=net_merged, ucte_cgmes_merge_info=ucte_cgmes_merge_info)
    return net_merged, ucte_cgmes_merge_info


def remove_station(net: Network, station_name: str, ucte_cgmes_merge_info: Optional[UcteCgmesMerge] = None) -> None:
    """Remove a station with all its elements from the network.

    Parameters
    ----------
    net : Network
        The network model to remove the station.
    station_name : str
        The name of the station to remove.
        id from net.get_substations().index
    ucte_cgmes_merge_info : Optional[UcteCgmesMerge]
        The merge information to store the removed tie lines and dangling lines.
        Note: gets modified in place.
    """
    voltage_level = net.get_voltage_levels()
    voltage_level = voltage_level[voltage_level["substation_id"] == station_name]
    element_ids = []
    tie_line_ids = []
    # A station can have multiple voltage levels if a coupler is open
    for station_vl in voltage_level.index:
        bus_breaker_topology = net.get_bus_breaker_topology(station_vl)
        elements = bus_breaker_topology.elements
        element_ids += list(elements.index)
        element_ids += list(bus_breaker_topology.switches.index)
        # handle existing tie lines
        if any(elements["type"].isin(["DANGLING_LINE"])):
            dangling_id = elements[elements["type"] == "DANGLING_LINE"].index
            if ucte_cgmes_merge_info is not None:
                ucte_cgmes_merge_info.removed_dangling_lines += list(dangling_id)
            dangling_lines = net.get_dangling_lines().loc[dangling_id]
            dangling_lines = dangling_lines[dangling_lines["tie_line_id"] != ""]
            tie_line_ids += list(dangling_lines["tie_line_id"])

    # remove duplicates, as the remove_elements raises an error if the element is already removed
    # Note: tie lines must be removed first, otherwise there is and parent/child error
    net.remove_elements(list(set(tie_line_ids)))
    net.remove_elements(list(set(element_ids)))
    net.remove_elements(station_name)
    if ucte_cgmes_merge_info is not None:
        ucte_cgmes_merge_info.removed_tie_lines += tie_line_ids


def get_ucte_border_tie_lines(net_ucte: Network, net_cgmes: Network) -> pd.DataFrame:
    """Get border lines between UCTE and CGMES network.

    The border lines are identified by the pairing key of the dangling lines in the CGMES network.

    Parameters
    ----------
    net_ucte : Network
        The UCTE network model.
    net_cgmes : Network
        The CGMES network model.

    Returns
    -------
    border_lines : pd.DataFrame
        The border lines between the UCTE and CGMES network.
        Note: contains the filtered DataFrame from net_ucte.get_tie_lines()
    """
    dangling_replace_grid = net_cgmes.get_dangling_lines()
    pairing_key_replace_grid = dangling_replace_grid["pairing_key"].values

    tie_outer_grid = net_ucte.get_tie_lines()
    border_lines = tie_outer_grid[tie_outer_grid["pairing_key"].isin(pairing_key_replace_grid)]
    return border_lines


def remove_area_from_ucte(net_ucte: Network, ucte_cgmes_merge_info: UcteCgmesMerge) -> None:
    """Remove the area from the UCTE network model.

    The CGMES area is identified by the name of the area or the country and removed from the UCTE network model.

    Parameters
    ----------
    net_ucte : Network
        The UCTE network model.
    ucte_cgmes_merge_info : Optional[UcteCgmesMerge]
        The merge information to store the removed tie lines and dangling lines.
        Note: gets modified in place.
    """
    substations = net_ucte.get_substations()
    if ucte_cgmes_merge_info.country_name is not None:
        substation_ids = substations[substations["country"] == ucte_cgmes_merge_info.country_name].index
    else:
        substation_ids = substations[substations.index.str.startswith(ucte_cgmes_merge_info.ucte_area_name)].index
    for station in substation_ids:
        remove_station(net=net_ucte, station_name=station, ucte_cgmes_merge_info=ucte_cgmes_merge_info)


def merge_ucte_and_cgmes(net_ucte: Network, net_cgmes: Network) -> Network:
    """Merge the UCTE and CGMES network model.

    Expects the tie lines and cgmes are removed.
    Note: this merge only works if the border lines are represented as dangling lines in both network models.
    E.g. inner German will not merge with this function. Convert the inner German lines to dangling lines first.
    Note: the input networks are modified in place and will not be useable after the merge.

    Parameters
    ----------
    net_ucte : Network
        The UCTE network model.
        note: is modified in place.
    net_cgmes : Network
        The CGMES network model.
        note: is modified in place.

    Returns
    -------
    net_merged : Network
        The merged network model.
    """
    # Note: the other way around seems not to work.
    net_cgmes.merge(net_ucte)
    # Note: the there are now new tie lines for the pairing keys
    return net_cgmes


def validate_merge_quality(net_merged: Network, ucte_cgmes_merge_info: UcteCgmesMerge) -> None:
    """Validate the merge quality.

    Creates statistics about the merge quality.

    Parameters
    ----------
    net_merged : Network
        The merged network model.
    ucte_cgmes_merge_info : UcteCgmesMerge
        The merge information to store the removed tie lines and dangling lines.
        Note: gets modified in place.
    """
    paring_key_cgmes = ucte_cgmes_merge_info.cgmes_dangling_lines["pairing_key"].values
    merged_dangling_lines = net_merged.get_dangling_lines()
    merged_dangling_lines = merged_dangling_lines[merged_dangling_lines["pairing_key"].isin(paring_key_cgmes)]
    merged_tie_line = merged_dangling_lines[merged_dangling_lines["tie_line_id"] != ""]

    ucte_cgmes_merge_info.statistics["tie_lines_replaced_ratio"] = len(ucte_cgmes_merge_info.removed_tie_lines) / len(
        merged_tie_line.values
    )
    ucte_cgmes_merge_info.statistics["n_paired_dangling_lines"] = merged_dangling_lines["paired"].sum()
    ucte_cgmes_merge_info.statistics["n_connected_but_not_paired_dangling_lines"] = (
        merged_dangling_lines["connected"] & ~merged_dangling_lines["paired"]
    ).sum()
    ucte_cgmes_merge_info.statistics["failed_pairing_keys"] = merged_dangling_lines[
        merged_dangling_lines["connected"] & ~merged_dangling_lines["paired"]
    ]["pairing_key"].values
    ucte_cgmes_merge_info.statistics["replaced_tie_lines"] = merged_tie_line["tie_line_id"].values
    ucte_cgmes_merge_info.statistics["removed_tie_lines_not_in_border_lines"] = [
        line for line in ucte_cgmes_merge_info.removed_tie_lines if line not in ucte_cgmes_merge_info.ucte_border_lines.index
    ]
