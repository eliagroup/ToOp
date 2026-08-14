# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Classes that represent runtime station-local topology structures.

The asset-topology pipeline evolves from master data to runtime views and then to
specialized projections:

- ``MasterAssetTopology`` stores the structural station design keyed by ``bus_group_id``.
- ``RuntimeAssetTopology`` and ``RuntimeBusGroup`` add live switch state and runtime bus ids.
- Simplified runtime views project the runtime state to the reduced DC-solver asset scope.
"""

import numpy as np
from beartype.typing import Any, Iterator, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from toop_engine_interfaces.asset_topology._model_utils import merged_round_trip_payload
from toop_engine_interfaces.asset_topology.asset_topology import (
    BusGroupSwitchingArray,
    CircuitGroup,
)
from toop_engine_interfaces.asset_topology.asset_types import BranchEnd
from toop_engine_interfaces.asset_topology.assets import AssetBay, BranchAsset, InjectionAsset
from toop_engine_interfaces.asset_topology.assets_runtime import RuntimeBusbar, RuntimeBusbarCoupler, RuntimeSwitchableAsset


def _validate_busgroup_switching_tables(
    station_grid_model_id: str,
    station_name: Optional[str],
    busbar_count: int,
    asset_count: int,
    asset_switching_table: np.ndarray,
    asset_connectivity: Optional[np.ndarray],
    asset_kind: str,
) -> None:
    """Validate switching-table shapes against the station dimensions."""
    if asset_switching_table.shape != (busbar_count, asset_count):
        raise ValueError(
            f"{asset_kind}_switching_table shape {asset_switching_table.shape} does not match busbars "
            f"{busbar_count} and {asset_kind} assets {asset_count}"
            f" Station_id: {station_grid_model_id}, Name: {station_name}"
        )

    if asset_connectivity is not None and asset_connectivity.shape != (busbar_count, asset_count):
        raise ValueError(
            f"{asset_kind}_connectivity shape {asset_connectivity.shape} does not match busbars "
            f"{busbar_count} and {asset_kind} assets {asset_count}"
            f" Station_id: {station_grid_model_id}, Name: {station_name}"
        )


def _validate_busgroup_physical_assignments(
    station_grid_model_id: str,
    station_name: Optional[str],
    asset_switching_table: np.ndarray,
    asset_connectivity: Optional[np.ndarray],
    asset_kind: str,
) -> None:
    """Validate that all current assignments are physically allowed."""
    if asset_connectivity is not None:
        if np.logical_and(asset_switching_table, np.logical_not(asset_connectivity)).any():
            raise ValueError(
                f"Not all current {asset_kind} assignments are physically allowed "
                f"Station_id: {station_grid_model_id}, Name: {station_name}"
            )


class RuntimeAssetConnection(BaseModel):
    """Busgroup-local association between a switching-table column and a materialized asset payload.

    Attributes
    ----------
    asset : RuntimeSwitchableAsset
        Station-local runtime asset payload aligned with one switching-table column.
    branch_end : Optional[BranchEnd]
        Optional canonical branch-end metadata for the station-local occurrence.
    asset_bay : Optional[AssetBay]
        Optional station-local asset-bay payload describing the physical switch path.
    """

    asset: RuntimeSwitchableAsset
    """Station-local asset payload aligned with one switching-table column."""

    branch_end: Optional[BranchEnd] = None
    """Optional branch-end metadata for this station-local asset occurrence."""

    asset_bay: Optional[AssetBay] = None
    """Optional station-local asset bay payload for this station-local asset occurrence."""

    def get_busbar_disconnector(self) -> Optional[dict[str, str]]:
        """Return the selector-switch mapping of the asset bay, if available.

        Returns
        -------
        Optional[dict[str, str]]
            Mapping from busbar id to selector-switch id, or ``None`` when the
            station-local asset has no asset-bay payload.
        """
        if self.asset_bay is not None:
            return self.asset_bay.busbar_disconnector_grid_model_id
        return None


class RuntimeBusGroup(BaseModel):
    """Busgroup data describing a single materialized station.

    The station identity refers to a bus-group or station-view identifier.
    A physical substation or voltage level may contain multiple bus-branch model bus ids.
    The station assets are aligned with the switching tables and describe the assets visible in that
    station view; they are not intended to define a topology-owned canonical asset list.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bus_group_id: str
    """The unique identifier of the station view or bus group.

    This is a station-view identifier and may be synthetic.
    Runtime electrical bus ids are tracked separately on the busbars.

    Included are all assets, busbars and couplers that are connectable via switches.
    Buses in the same station that are connected via branches are excluded in this specific bus.

    This means, that two stations/buses can have the same elements if the station is currently split.
    """

    voltage_level_id: Optional[str] = None
    """Voltage level identifier backing this station view in the source grid."""

    name: Optional[str] = None
    """The name of the station."""

    station_type: Optional[str] = None
    """The type of the station."""

    region: Optional[str] = None
    """The region of the station."""

    voltage_level: Optional[float] = None
    """The voltage level of the station."""

    busbars: list[RuntimeBusbar]
    """The list of busbars at the station."""

    bus_branch_bus_ids: list[str] = Field(default_factory=list)
    """Unique non-empty bus-branch bus ids currently represented by this station view."""

    couplers: list[RuntimeBusbarCoupler]
    """The list of couplers at the station."""

    branch_switching_table: BusGroupSwitchingArray
    """Holds the switching of each branch asset to each busbar, shape (n_bus, n_branch_asset).

    An entry is true if the asset is connected to the busbar.
    Note: An asset can be connected to multiple busbars, in which case a closed coupler is assumed
    to be present between these busbars.
    Note: An asset can be connected to none of the busbars. In this case, the asset is intentionally
    disconnected as part of a transmission line switching action. In practice, this usually involves
    a separate switch from the asset-to-busbar couplers, as each asset usually has a switch that
    completely disconnects it from the station. These switches are not modelled here, a
    postprocessing routine needs to do the translation to this physical layout. Do not use
    in_service for intentional disconnections.
    """

    injection_switching_table: BusGroupSwitchingArray
    """Holds the switching of each injection asset to each busbar, shape (n_bus, n_injection_asset)."""

    branch_connectivity: Optional[BusGroupSwitchingArray] = None
    """Holds all physically possible branch layouts, shape (n_bus, n_branch_asset)."""

    injection_connectivity: Optional[BusGroupSwitchingArray] = None
    """Holds all physically possible injection layouts, shape (n_bus, n_injection_asset)."""

    model_log: Optional[list[str]] = None
    """Holds log messages from the model creation process.

    This can be used to store information about the model creation process, e.g. warnings or errors.
    A potential use case is to inform the user about data quality issues e.g. missing the Asset Bay switches.
    """

    branch_connections: list[RuntimeAssetConnection] = Field(default_factory=list)
    """Station-local branch payloads aligned with ``branch_switching_table``."""

    injection_connections: list[RuntimeAssetConnection] = Field(default_factory=list)
    """Station-local injection payloads aligned with ``injection_switching_table``."""

    @field_validator(
        "branch_switching_table",
        "injection_switching_table",
        "branch_connectivity",
        "injection_connectivity",
        mode="before",
    )
    @classmethod
    def normalize_station_tables(
        cls,
        v: object | None,
    ) -> Optional[np.ndarray]:
        """Normalize switching and connectivity table inputs to boolean arrays."""
        if v is None:
            return None
        return np.asarray(v, dtype=bool)

    @field_validator("busbars")
    @classmethod
    def check_int_id_unique(cls, v: list[RuntimeBusbar]) -> list[RuntimeBusbar]:
        """Check if the int_ids of the busbars are unique."""
        int_ids = [busbar.int_id for busbar in v]
        if len(int_ids) != len(set(int_ids)):
            raise ValueError("busbar int_ids must be unique per station")
        return v

    @field_validator("busbars", mode="before")
    @classmethod
    def normalize_runtime_busbars(cls, v: object) -> list[RuntimeBusbar]:
        """Validate station busbars as runtime busbars."""
        if not isinstance(v, list):
            return v
        runtime_busbars: list[RuntimeBusbar] = []
        for busbar in v:
            if isinstance(busbar, RuntimeBusbar):
                runtime_busbars.append(busbar)
            elif isinstance(busbar, dict):
                runtime_busbars.append(RuntimeBusbar.model_validate(busbar))
            else:
                runtime_busbars.append(RuntimeBusbar(**busbar.model_dump()))
        return runtime_busbars

    @field_validator("bus_branch_bus_ids", mode="before")
    @classmethod
    def normalize_bus_branch_bus_ids(cls, v: Optional[list[str]]) -> list[str]:
        """Normalize explicit bus-branch bus ids to a unique sorted list."""
        if v is None:
            return []
        return sorted({bus_id for bus_id in v if bus_id not in {None, ""}})

    @field_validator("couplers")
    @classmethod
    def check_coupler_busbars_different(cls, v: list[RuntimeBusbarCoupler]) -> list[RuntimeBusbarCoupler]:
        """Check if the couplers do not connect the same busbar on both ends."""
        for coupler in v:
            if coupler.busbar_from_id == coupler.busbar_to_id:
                raise ValueError(f"Coupler {coupler.grid_model_id} connects the same busbar on both ends")
        return v

    @field_validator("couplers", mode="before")
    @classmethod
    def normalize_runtime_couplers(cls, v: object) -> list[RuntimeBusbarCoupler]:
        """Validate station couplers as runtime couplers."""
        if not isinstance(v, list):
            return v
        runtime_couplers: list[RuntimeBusbarCoupler] = []
        for coupler in v:
            if isinstance(coupler, RuntimeBusbarCoupler):
                runtime_couplers.append(coupler)
            elif isinstance(coupler, dict):
                runtime_couplers.append(RuntimeBusbarCoupler.model_validate(coupler))
            else:
                runtime_couplers.append(RuntimeBusbarCoupler(**coupler.model_dump()))
        return runtime_couplers

    @model_validator(mode="after")
    def check_coupler_busbars_exist(self: "RuntimeBusGroup") -> "RuntimeBusGroup":
        """Check if all coupler busbar ids exist in busbars."""
        busbar_ids = [busbar.int_id for busbar in self.busbars]
        for coupler in self.couplers:
            if coupler.busbar_from_id not in busbar_ids or coupler.busbar_to_id not in busbar_ids:
                raise ValueError(
                    f"Coupler {coupler.grid_model_id} references non-existing busbars"
                    f" Station_id: {self.bus_group_id}, Name: {self.name}"
                )
        return self

    @model_validator(mode="after")
    def check_coupler_references(self: "RuntimeBusGroup") -> "RuntimeBusGroup":
        """Check if all closed couplers reference in-service busbars."""
        busbar_state_map = {busbar.int_id: busbar.in_service for busbar in self.busbars}
        for coupler in self.couplers:
            if coupler.open or not coupler.in_service:
                continue
            if busbar_state_map[coupler.busbar_from_id] != busbar_state_map[coupler.busbar_to_id]:
                raise ValueError(
                    f"Closed coupler {coupler.grid_model_id} connects out-of-service busbar with in-service busbar."
                    f" Station_id: {self.bus_group_id}, Name: {self.name}"
                )
        return self

    @model_validator(mode="after")
    def sync_bus_branch_bus_ids(self: "RuntimeBusGroup") -> "RuntimeBusGroup":
        """Store the unique bus-branch bus ids contained in the station busbars."""
        bus_branch_bus_ids = sorted(
            {busbar.bus_branch_bus_id for busbar in self.busbars if busbar.bus_branch_bus_id not in {None, ""}}
        )
        self.bus_branch_bus_ids = bus_branch_bus_ids if bus_branch_bus_ids else self.bus_branch_bus_ids
        return self

    @model_validator(mode="after")
    def check_asset_shapes(self: "RuntimeBusGroup") -> "RuntimeBusGroup":
        """Check if switching-table-aligned station-local assets match the matrix shapes.

        Returns
        -------
        RuntimeBusGroup
            The validated station instance.

        Raises
        ------
        ValueError
            If switching tables, connectivity tables, or asset-bay busbar references
            do not align with the station-local structure.
        """
        _validate_busgroup_switching_tables(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.branch_connections),
            asset_switching_table=self.branch_switching_table,
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_busgroup_physical_assignments(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            asset_switching_table=self.branch_switching_table,
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_busgroup_switching_tables(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.injection_connections),
            asset_switching_table=self.injection_switching_table,
            asset_connectivity=self.injection_connectivity,
            asset_kind="injection",
        )
        _validate_busgroup_physical_assignments(
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
            for busbar_id in asset_connection.asset_bay.busbar_disconnector_grid_model_id:
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
        if not isinstance(other, RuntimeBusGroup):
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

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "RuntimeBusGroup":
        """Copy and revalidate the station.

        Parameters
        ----------
        update : Optional[dict[str, Any]], optional
            Field updates to merge into the copied station.
        deep : bool, default=False
            Whether to deep-copy nested structures before validation.

        Returns
        -------
        RuntimeBusGroup
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

        payload = merged_round_trip_payload(self, update, deep=deep)
        return type(self).model_validate(payload)

    def is_split(self) -> bool:
        """Return whether the station view spans more than one non-empty bus-branch bus id."""
        return len(self.bus_branch_bus_ids) > 1

    @property
    def grid_model_id(self) -> str:
        """Backward-compatible alias for the station identifier."""
        return self.bus_group_id

    @property
    def assets(self) -> list[RuntimeSwitchableAsset]:
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
        topology_assets: Optional[list[RuntimeSwitchableAsset]] = None,
        asset_scope: Literal["all", "branch", "injection"] = "all",
    ) -> list[RuntimeSwitchableAsset]:
        """Return in-service assets connected to one busbar.

        Parameters
        ----------
        busbar_index : int
            Row index into the station switching tables.
        topology_assets : Optional[list[RuntimeSwitchableAsset]]
            Ignored for materialized stations because payloads are embedded locally.
        asset_scope : Literal["all", "branch", "injection"]
            Restrict the lookup to branch or injection connections.

        Returns
        -------
        list[RuntimeSwitchableAsset]
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


class RuntimeAssetTopology(BaseModel):
    """Runtime topology payload grouped independently from canonical master data.

    The wrapper carries runtime station snapshots and optional runtime-visible
    circuit-group metadata aligned with the same topology view. It is the runtime
    companion of ``MasterAssetTopology`` and intentionally carries no topology-owned
    canonical asset collections itself.
    """

    bus_groups: list[RuntimeBusGroup]
    """Runtime station snapshots for the topology view."""

    circuit_groups: Optional[list[CircuitGroup]] = None
    """Optional circuit-group metadata carried alongside the runtime stations."""

    @field_validator("bus_groups")
    @classmethod
    def check_bus_group_ids_unique(cls, v: list[RuntimeBusGroup]) -> list[RuntimeBusGroup]:
        """Validate uniqueness of runtime bus-group identifiers.

        Parameters
        ----------
        v : list[RuntimeBusGroup]
            Runtime bus groups assigned to the wrapper.

        Returns
        -------
        list[RuntimeBusGroup]
            Validated runtime stations.
        """
        station_ids = [station.bus_group_id for station in v]
        if len(station_ids) != len(set(station_ids)):
            raise ValueError("bus_group_id must be unique for runtime topology stations")
        return v


def iter_bus_group_asset_references(
    stations: list[RuntimeBusGroup],
) -> Iterator[tuple[str, Literal["branch", "injection"], str, str | None]]:
    """Yield normalized runtime station asset references."""
    for station in stations:
        for asset_connection in station.branch_connections:
            asset_bay = asset_connection.asset_bay
            yield (
                station.bus_group_id,
                "branch",
                asset_connection.asset.grid_model_id,
                asset_bay.asset_bay_id if asset_bay is not None else None,
            )
        for asset_connection in station.injection_connections:
            asset_bay = asset_connection.asset_bay
            yield (
                station.bus_group_id,
                "injection",
                asset_connection.asset.grid_model_id,
                asset_bay.asset_bay_id if asset_bay is not None else None,
            )


def validate_runtime_bus_group_asset_references(
    stations: list[RuntimeBusGroup],
    branch_assets: list[BranchAsset],
    injection_assets: list[InjectionAsset],
    asset_bays: list[AssetBay],
) -> None:
    """Validate runtime station references against canonical topology payloads."""
    branch_asset_ids = {asset.grid_model_id for asset in branch_assets}
    injection_asset_ids = {asset.grid_model_id for asset in injection_assets}
    asset_bay_ids = {asset_bay.asset_bay_id for asset_bay in asset_bays}

    allowed_asset_ids = {
        "branch": branch_asset_ids,
        "injection": injection_asset_ids,
    }
    error_prefix = {
        "branch": "Branch",
        "injection": "Injection",
    }

    for station_id, asset_kind, asset_id, asset_bay_id in iter_bus_group_asset_references(stations):
        if asset_id not in allowed_asset_ids[asset_kind]:
            raise ValueError(
                f"{error_prefix[asset_kind]} asset grid_model_id {asset_id} referenced by station "
                f"{station_id} does not exist in topology assets"
            )
        if asset_bay_id is not None and asset_bay_id not in asset_bay_ids:
            raise ValueError(
                f"asset_bay_id {asset_bay_id} referenced by station {station_id} does not exist in topology asset bays"
            )


def get_asset_bay_ids_for_bus_group_asset(stations: list[RuntimeBusGroup], asset_grid_model_id: str) -> list[str]:
    """Return ordered unique asset-bay ids for one asset."""
    asset_bay_ids: list[str] = []
    seen_ids: set[str] = set()
    for _, _, asset_id, asset_bay_id in iter_bus_group_asset_references(stations):
        if asset_id != asset_grid_model_id or asset_bay_id is None or asset_bay_id in seen_ids:
            continue
        seen_ids.add(asset_bay_id)
        asset_bay_ids.append(asset_bay_id)
    return asset_bay_ids


def get_asset_bays_for_bus_group_asset(
    stations: list[RuntimeBusGroup],
    asset_bays: list[AssetBay],
    asset_grid_model_id: str,
) -> list[AssetBay]:
    """Return ordered unique asset-bay payloads for one asset."""
    asset_bay_map = {asset_bay.asset_bay_id: asset_bay for asset_bay in asset_bays}
    return [
        asset_bay_map[asset_bay_id] for asset_bay_id in get_asset_bay_ids_for_bus_group_asset(stations, asset_grid_model_id)
    ]
