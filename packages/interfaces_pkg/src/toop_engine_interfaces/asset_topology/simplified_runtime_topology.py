# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Runtime topology subclasses marking already simplified station views."""

from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeAssetTopology, RuntimeBusGroup


class SimplifiedBusGroup(RuntimeBusGroup):
    """Runtime bus-group subclass marking stations that passed preprocessing simplification."""


class SimplifiedAssetTopology(RuntimeAssetTopology):
    """Runtime topology subclass carrying already simplified station views."""

    stations: list[SimplifiedBusGroup]
    """Simplified runtime station snapshots for the topology view."""


def to_simplified_bus_group(station: RuntimeBusGroup) -> SimplifiedBusGroup:
    """Convert a runtime station to the simplified subtype without reserializing nested assets."""
    return SimplifiedBusGroup.model_validate(station, from_attributes=True)
