# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Electrical circuit group helpers for Powsybl outage handling."""

from toop_engine_grid_helpers.powsybl.electrical_circuit_groups.electrical_circuit_groups import (
    identify_circuit_groups,
)
from toop_engine_grid_helpers.powsybl.electrical_circuit_groups.helper_functions import (
    build_branch_circuit_group_lookup,
    build_busbar_circuit_group_lookup,
    build_circuit_group_lookup_index,
    build_circuit_group_map,
    get_failing_elements_by_branch_ids,
    get_failing_elements_by_busbar_ids,
    get_failing_switches_by_branch_ids,
    get_failing_switches_by_busbar_ids,
    preprocess_circuit_group_lookup,
)
from toop_engine_grid_helpers.powsybl.electrical_circuit_groups.types import (
    ElectricalCircuitGroup,
    ElectricalCircuitGroupIdentification,
)

__all__ = [
    "ElectricalCircuitGroup",
    "ElectricalCircuitGroupIdentification",
    "build_branch_circuit_group_lookup",
    "build_busbar_circuit_group_lookup",
    "build_circuit_group_lookup_index",
    "build_circuit_group_map",
    "get_failing_elements_by_branch_ids",
    "get_failing_elements_by_busbar_ids",
    "get_failing_switches_by_branch_ids",
    "get_failing_switches_by_busbar_ids",
    "identify_circuit_groups",
    "preprocess_circuit_group_lookup",
]
