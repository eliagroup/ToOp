# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Classes that represent the applied topology.

This is the station-local view of the topology
with assets and asset bays, including the differences to the original topology.
"""

from pydantic import BaseModel, Field
from toop_engine_interfaces.asset_topology.asset_topology import MasterAssetTopology
from toop_engine_interfaces.asset_topology.assets import BusbarCoupler
from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeBusGroup
from typing_extensions import deprecated


@deprecated("RealizedTopology is deprecated; use RuntimeTopology (AssetTopology) instead.")
class RealizedTopology(BaseModel):
    """A realized topology, including the new topology and the changes made to the original topology.

    DeprecationWarning:
    This model is deprecated and should be replaced by Topology (AssetTopology)
    in new code.

    This is similar to AppliedStation but holding information for all stations in the topology.
    The diffs are include a station identifier that shows which station in the topology was affected by the
    diff.
    """

    master_data: MasterAssetTopology | None = None
    """Canonical master data associated with the realized runtime stations when available."""

    bus_groups: list[RuntimeBusGroup] = Field(default_factory=list)
    """The realized asset stations that were directly applied or compared."""

    coupler_diff: list[tuple[str, BusbarCoupler]]
    """A list of couplers that have been switched. Each tuple contains the station id
    and the coupler that was switched."""

    branch_reassignment_diff: list[tuple[str, int, int, bool]]
    """Branch reassignments as ``(station_id, branch_index, busbar_index, connected)`` tuples."""

    injection_reassignment_diff: list[tuple[str, int, int, bool]]
    """Injection reassignments as ``(station_id, injection_index, busbar_index, connected)`` tuples."""

    branch_disconnection_diff: list[tuple[str, int]]
    """Branch disconnections as ``(station_id, branch_index)`` tuples."""

    injection_disconnection_diff: list[tuple[str, int]]
    """Injection disconnections as ``(station_id, injection_index)`` tuples."""


@deprecated("AppliedStation is deprecated; use RuntimeStation (AssetTopology) instead.")
class AppliedStation(BaseModel):
    """A realized station, including the new station and the changes made to the original station."""

    bus_group: RuntimeBusGroup
    """The realized asset station object"""

    coupler_diff: list[BusbarCoupler]
    """A list of couplers that have been switched."""

    branch_reassignment_diff: list[tuple[int, int, bool]]
    """Branch reassignments as ``(branch_index, busbar_index, connected)`` tuples."""

    injection_reassignment_diff: list[tuple[int, int, bool]]
    """Injection reassignments as ``(injection_index, busbar_index, connected)`` tuples."""

    branch_disconnection_diff: list[int]
    """Branch indices that were disconnected."""

    injection_disconnection_diff: list[int]
    """Injection indices that were disconnected."""
