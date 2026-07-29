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
    """Station-local association between a switching-table column and a materialized asset payload.

    Attributes
    ----------
    asset : SwitchableAsset
        Station-local runtime asset payload aligned with one switching-table column.
    branch_end : Optional[BranchEnd]
        Optional canonical branch-end metadata for the station-local occurrence.
    asset_bay : Optional[AssetBay]
        Optional station-local asset-bay payload describing the physical switch path.
    """

    asset: SwitchableAsset
    """Station-local asset payload aligned with one switching-table column."""

    branch_end: Optional[BranchEnd] = None
    """Optional branch-end metadata for this station-local asset occurrence."""

    asset_bay: Optional[AssetBay] = None
    """Optional station-local asset bay payload for this station-local asset occurrence."""

    def get_sr_switch(self) -> Optional[dict[str, str]]:
        """Return the selector-switch mapping of the asset bay, if available.

        Returns
        -------
        Optional[dict[str, str]]
            Mapping from busbar id to selector-switch id, or ``None`` when the
            station-local asset has no asset-bay payload.
        """
        if self.asset_bay is not None:
            return self.asset_bay.sr_switch_grid_model_id
        return None


class MaterializedStation(_StationStructure):
    """Station data describing a single materialized station.

    The station identity refers to a bus-group or station-view identifier.
    A physical substation or voltage level may contain multiple bus-branch model bus ids.
    The station assets are aligned with the switching tables and describe the assets visible in that
    station view; they are not intended to define a topology-owned canonical asset list.
    """

    branch_connections: list[MaterializedAssetConnection] = Field(default_factory=list)
    """Station-local branch payloads aligned with ``branch_switching_table``."""

    injection_connections: list[MaterializedAssetConnection] = Field(default_factory=list)
    """Station-local injection payloads aligned with ``injection_switching_table``."""

    @model_validator(mode="after")
    def check_asset_shapes(self: "MaterializedStation") -> "MaterializedStation":
        """Check if switching-table-aligned station-local assets match the matrix shapes.

        Returns
        -------
        MaterializedStation
            The validated station instance.

        Raises
        ------
        ValueError
            If switching tables, connectivity tables, or asset-bay busbar references
            do not align with the station-local structure.
        """
        _validate_station_switching_tables(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.branch_connections),
            asset_switching_table=self.branch_switching_table,
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_station_physical_assignments(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            asset_switching_table=self.branch_switching_table,
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_station_switching_tables(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.injection_connections),
            asset_switching_table=self.injection_switching_table,
            asset_connectivity=self.injection_connectivity,
            asset_kind="injection",
        )
        _validate_station_physical_assignments(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            asset_switching_table=self.injection_switching_table,
            asset_connectivity=self.injection_connectivity,
            asset_kind="injection",
        )
        busbar_ids = {busbar.grid_model_id for busbar in self.busbars}
        for asset_connection in [*self.branch_connections, *self.injection_connections]:
            if asset_connection.asset_bay is None:
                continue
            for busbar_id in asset_connection.asset_bay.sr_switch_grid_model_id:
                if busbar_id not in busbar_ids:
                    raise ValueError(
                        f"busbar_id {busbar_id} in asset {asset_connection.asset.grid_model_id} does not exist in busbars"
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
            self.bus_group_id == other.bus_group_id
            and self.voltage_level_id == other.voltage_level_id
            and self.region == other.region
            and self.busbars == other.busbars
            and self.bus_branch_bus_ids == other.bus_branch_bus_ids
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
        """Copy and revalidate the station.

        Parameters
        ----------
        update : Optional[dict[str, Any]], optional
            Field updates to merge into the copied station.
        deep : bool, default=False
            Whether to deep-copy nested structures before validation.

        Returns
        -------
        MaterializedStation
            Copied and revalidated station instance.
        """
        if update and ("asset_switching_table" in update or "asset_connectivity" in update):
            normalized_update = dict(update)

            if "asset_switching_table" in normalized_update:
                asset_switching_table = normalized_update.pop("asset_switching_table")
                branch_count = len(self.branch_connections)
                normalized_update["branch_switching_table"] = asset_switching_table[:, :branch_count]
                normalized_update["injection_switching_table"] = asset_switching_table[:, branch_count:]

            if "asset_connectivity" in normalized_update:
                asset_connectivity = normalized_update.pop("asset_connectivity")
                if asset_connectivity is None:
                    normalized_update["branch_connectivity"] = None
                    normalized_update["injection_connectivity"] = None
                else:
                    branch_count = len(self.branch_connections)
                    normalized_update["branch_connectivity"] = asset_connectivity[:, :branch_count]
                    normalized_update["injection_connectivity"] = asset_connectivity[:, branch_count:]

            update = normalized_update

        payload = _merged_round_trip_payload(self, update, deep=deep)
        return type(self).model_validate(payload)

    @property
    def grid_model_id(self) -> str:
        """Backward-compatible alias for the station identifier."""
        return self.bus_group_id

    @property
    def assets(self) -> list[SwitchableAsset]:
        """Return combined station-local assets in legacy column order."""
        return [
            *(asset_connection.asset for asset_connection in self.branch_connections),
            *(asset_connection.asset for asset_connection in self.injection_connections),
        ]

    @property
    def asset_switching_table(self) -> np.ndarray:
        """Return the combined switching table in legacy branch-then-injection order."""
        return np.concatenate([self.branch_switching_table, self.injection_switching_table], axis=1)

    @property
    def asset_connectivity(self) -> Optional[np.ndarray]:
        """Return the combined connectivity matrix in legacy branch-then-injection order."""
        if self.branch_connectivity is None and self.injection_connectivity is None:
            return None
        branch_connectivity = self.branch_connectivity
        injection_connectivity = self.injection_connectivity
        if branch_connectivity is None:
            branch_connectivity = np.zeros_like(self.branch_switching_table, dtype=bool)
        if injection_connectivity is None:
            injection_connectivity = np.zeros_like(self.injection_switching_table, dtype=bool)
        return np.concatenate([branch_connectivity, injection_connectivity], axis=1)

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
