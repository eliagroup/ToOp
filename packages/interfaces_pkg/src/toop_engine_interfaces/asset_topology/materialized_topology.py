# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Classes that represent the materialized topology.

This is the station-local view of the topology populated with all necessary info
on assets and asset bays.
"""

import numpy as np
from beartype.typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from toop_engine_interfaces.asset_topology.asset_types import BranchEnd
from toop_engine_interfaces.asset_topology.assets import AssetBay, SwitchableAsset
from toop_engine_interfaces.asset_topology.station_models import (
    _merged_round_trip_payload,
    _StationStructure,
    _validate_station_physical_assignments,
    _validate_station_switching_tables,
)


class MaterializedAssetConnection(BaseModel):
    """Station-local association between a switching-table column and a materialized asset payload."""

    asset: SwitchableAsset
    """Station-local asset payload aligned with one switching-table column."""

    branch_end: Optional[BranchEnd] = None
    """Optional branch-end metadata for this station-local asset occurrence."""

    asset_bay: Optional[AssetBay] = None
    """Optional station-local asset bay payload for this station-local asset occurrence."""

    def get_sr_switch(self) -> Optional[dict[str, str]]:
        """Return the sr_switch_grid_model_id dict from the asset bay if it exists."""
        if self.asset_bay is not None:
            return self.asset_bay.sr_switch_grid_model_id
        return None


class MaterializedStation(_StationStructure):
    """Station data describing a single materialized station.

    The station identity refers to a single bus-branch model bus_id that represents one splitable
    station view.
    A physical substation or voltage level may contain multiple bus-branch model bus_ids.
    The station assets are aligned with the switching tables and describe the assets visible in that
    station view; they are not intended to define a topology-owned canonical asset list.
    """

    branch_connections: list[MaterializedAssetConnection] = Field(default_factory=list)
    """Station-local branch payloads aligned with ``branch_switching_table``."""

    injection_connections: list[MaterializedAssetConnection] = Field(default_factory=list)
    """Station-local injection payloads aligned with ``injection_switching_table``."""

    @model_validator(mode="after")
    def check_asset_shapes(self: "MaterializedStation") -> "MaterializedStation":
        """Check if switching-table-aligned station-local assets match the matrix shapes."""
        _validate_station_switching_tables(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.branch_connections),
            asset_switching_table=self.branch_switching_table,
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_station_physical_assignments(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            asset_switching_table=self.branch_switching_table,
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_station_switching_tables(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.injection_connections),
            asset_switching_table=self.injection_switching_table,
            asset_connectivity=self.injection_connectivity,
            asset_kind="injection",
        )
        _validate_station_physical_assignments(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            asset_switching_table=self.injection_switching_table,
            asset_connectivity=self.injection_connectivity,
            asset_kind="injection",
        )
        return self

    @model_validator(mode="after")
    def check_asset_bay(self: "MaterializedStation") -> "MaterializedStation":
        """Check if the asset bay bus is in busbars."""
        busbar_grid_model_id = [busbar.grid_model_id for busbar in self.busbars]
        for asset_connection in [*self.branch_connections, *self.injection_connections]:
            asset = asset_connection.asset
            asset_bay = asset_connection.asset_bay
            if asset_bay is not None:
                for busbar_id in asset_bay.sr_switch_grid_model_id.keys():
                    if busbar_id not in busbar_grid_model_id:
                        raise ValueError(
                            f"busbar_id {busbar_id} in asset {asset.grid_model_id} does not exist in busbars"
                            f" Station_id: {self.grid_model_id}, Name: {self.name}"
                        )

        return self

    def __eq__(self, other: object) -> bool:
        """Check if two stations are equal.

        Parameters
        ----------
        other : object
            The other station to compare to.

        Returns
        -------
        bool
            True if the stations are equal, False otherwise.
        """
        if not isinstance(other, MaterializedStation):
            return False
        return (
            self.grid_model_id == other.grid_model_id
            and self.region == other.region
            and self.busbars == other.busbars
            and self.couplers == other.couplers
            and self.branch_connections == other.branch_connections
            and self.injection_connections == other.injection_connections
            and np.array_equal(self.branch_switching_table, other.branch_switching_table)
            and np.array_equal(self.injection_switching_table, other.injection_switching_table)
            and (
                np.array_equal(self.branch_connectivity, other.branch_connectivity)
                if (self.branch_connectivity is not None and other.branch_connectivity is not None)
                else self.branch_connectivity == other.branch_connectivity
            )
            and (
                np.array_equal(self.injection_connectivity, other.injection_connectivity)
                if (self.injection_connectivity is not None and other.injection_connectivity is not None)
                else self.injection_connectivity == other.injection_connectivity
            )
        )

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "MaterializedStation":
        """Copy and revalidate the station."""
        payload = _merged_round_trip_payload(self, update, deep=deep)
        return type(self).model_validate(payload)

    def get_connected_assets(
        self,
        busbar_index: int,
        topology_assets: Optional[list[SwitchableAsset]] = None,
        asset_scope: Literal["all", "branch", "injection"] = "all",
    ) -> list[SwitchableAsset]:
        """Return in-service assets connected to one busbar.

        Parameters
        ----------
        busbar_index : int
            Row index into the station switching tables.
        topology_assets : Optional[list[SwitchableAsset]]
            Ignored for materialized stations because payloads are embedded locally.
        asset_scope : Literal["all", "branch", "injection"]
            Restrict the lookup to branch or injection connections.

        Returns
        -------
        list[SwitchableAsset]
            Connected in-service assets for the requested busbar and scope.
        """
        del topology_assets
        if asset_scope == "branch":
            return [
                asset_connection.asset
                for asset_connection, is_connected in zip(
                    self.branch_connections,
                    self.branch_switching_table[busbar_index],
                    strict=True,
                )
                if is_connected and asset_connection.asset.in_service
            ]
        if asset_scope == "injection":
            return [
                asset_connection.asset
                for asset_connection, is_connected in zip(
                    self.injection_connections,
                    self.injection_switching_table[busbar_index],
                    strict=True,
                )
                if is_connected and asset_connection.asset.in_service
            ]
        return [
            *self.get_connected_assets(busbar_index, asset_scope="branch"),
            *self.get_connected_assets(busbar_index, asset_scope="injection"),
        ]
