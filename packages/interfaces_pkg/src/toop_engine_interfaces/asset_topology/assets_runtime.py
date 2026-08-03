# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Runtime asset topology models.

These models extend the canonical asset-topology objects with live runtime state,
such as service flags, live bus assignments, and resolved coupler endpoints.
"""

from beartype.typing import Optional
from toop_engine_interfaces.asset_topology.assets import (
    BranchAsset,
    Busbar,
    BusbarCoupler,
    InjectionAsset,
    SwitchableAsset,
)


class RuntimeBusbar(Busbar):
    """Runtime busbar with live service and electrical-bus state."""

    in_service: bool = True
    """Whether the busbar is in service in the current runtime state."""

    bus_branch_bus_id: Optional[str] = None
    """Runtime-only electrical bus id for this physical busbar."""


class RuntimeBusbarCoupler(BusbarCoupler):
    """Runtime coupler with live open and service state."""

    busbar_from_id: int
    """Resolved runtime busbar int id on the from side."""

    busbar_to_id: int
    """Resolved runtime busbar int id on the to side."""

    open: bool = False
    """Whether the coupler is open in the current runtime state."""

    in_service: bool = True
    """Whether the coupler is in service in the current runtime state."""


class RuntimeSwitchableAsset(SwitchableAsset):
    """Runtime switchable asset with live service state."""

    in_service: bool = True
    """Whether the asset is in service in the current runtime state."""


class RuntimeBranchAsset(BranchAsset, RuntimeSwitchableAsset):
    """Runtime switchable asset representing a branch-type element."""


class RuntimeInjectionAsset(InjectionAsset, RuntimeSwitchableAsset):
    """Runtime switchable asset representing an injection-type element."""
