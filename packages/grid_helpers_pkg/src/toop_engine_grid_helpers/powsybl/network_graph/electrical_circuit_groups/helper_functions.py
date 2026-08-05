# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for lookup-oriented outage queries."""

import numpy as np
import pandas as pd
import pandera.typing as pat
from toop_engine_grid_helpers.powsybl.network_graph.electrical_circuit_groups.types import (
    BranchElectricalCircuitGroupSchema,
    CircuitGroupLookupIndex,
    ElectricalCircuitGroup,
    ElectricalCircuitGroupMap,
    FailingElementsByLookupId,
    FailingSwitchesByLookupId,
    InjectionElectricalCircuitGroupSchema,
    PreparedCircuitGroupLookupData,
    SwitchElectricalCircuitGroupSchema,
)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Remove duplicates while keeping the first occurrence order."""
    return list(dict.fromkeys(values))


def _group_values_by_key(keys: np.ndarray, values: np.ndarray) -> dict[int, list[str]]:
    """Group values by integer keys using array sorting instead of pandas groupby."""
    valid_mask = pd.notna(keys)
    if not np.any(valid_mask):
        return {}

    valid_keys = np.asarray(keys[valid_mask], dtype=np.int64)
    valid_values = np.asarray(values[valid_mask], dtype=object)
    order = np.argsort(valid_keys, kind="stable")
    sorted_keys = valid_keys[order]
    sorted_values = valid_values[order]
    unique_keys, start_indices = np.unique(sorted_keys, return_index=True)
    end_indices = np.r_[start_indices[1:], len(sorted_keys)]
    return {
        int(group_id): sorted_values[start_index:end_index].tolist()
        for group_id, start_index, end_index in zip(unique_keys, start_indices, end_indices, strict=False)
    }


def _build_busbar_lookup_maps(
    switches: pat.DataFrame[SwitchElectricalCircuitGroupSchema],
    injection: pat.DataFrame[InjectionElectricalCircuitGroupSchema],
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Build busbar-to-primary-group and busbar-to-asset-groups lookup maps."""
    busbar_sections = injection[injection["type"] == "BUSBAR_SECTION"]
    busbar_to_primary_group = {
        busbar_id: int(group_id) for busbar_id, group_id in busbar_sections["electrical_circuit_group"].astype(int).items()
    }
    busbar_group_by_bus = (
        busbar_sections[["bus_breaker_bus_id", "electrical_circuit_group"]]
        .drop_duplicates(subset=["bus_breaker_bus_id"])
        .set_index("bus_breaker_bus_id")["electrical_circuit_group"]
    )

    switch_sides = switches.copy().reset_index()[
        [
            "id",
            "bus_breaker_bus1_id",
            "bus_breaker_bus2_id",
            "electrical_circuit_group_bus1",
            "electrical_circuit_group_bus2",
        ]
    ]
    side1_group = switch_sides["bus_breaker_bus1_id"].map(busbar_group_by_bus)
    side2_group = switch_sides["bus_breaker_bus2_id"].map(busbar_group_by_bus)
    side1_present = side1_group.notna().to_numpy(copy=False)
    side2_present = side2_group.notna().to_numpy(copy=False)
    asset_mask = side1_present ^ side2_present

    asset_groups_by_primary: dict[int, dict[int, None]] = {}
    if np.any(asset_mask):
        asset_rows = switch_sides.loc[asset_mask]
        primary_group = side1_group.loc[asset_mask].where(side1_group.loc[asset_mask].notna(), side2_group.loc[asset_mask])
        primary_group_values = primary_group.to_numpy(dtype=np.int64, copy=False)
        bus1_group_values = asset_rows["electrical_circuit_group_bus1"].to_numpy(dtype=np.int64, copy=False)
        bus2_group_values = asset_rows["electrical_circuit_group_bus2"].to_numpy(dtype=np.int64, copy=False)
        asset_group_values = np.where(bus1_group_values != primary_group_values, bus1_group_values, bus2_group_values)
        keep_mask = asset_group_values != primary_group_values

        for primary_group_id, asset_group_id in zip(
            primary_group_values[keep_mask], asset_group_values[keep_mask], strict=False
        ):
            asset_groups_by_primary.setdefault(int(primary_group_id), {}).setdefault(int(asset_group_id), None)

    busbar_to_asset_groups = {
        busbar_id: list(asset_groups_by_primary.get(primary_group_id, {}).keys())
        for busbar_id, primary_group_id in busbar_to_primary_group.items()
    }
    return busbar_to_primary_group, busbar_to_asset_groups


def preprocess_circuit_group_lookup(
    branches: pat.DataFrame[BranchElectricalCircuitGroupSchema],
    switches: pat.DataFrame[SwitchElectricalCircuitGroupSchema],
    injection: pat.DataFrame[InjectionElectricalCircuitGroupSchema],
) -> PreparedCircuitGroupLookupData:
    """Precompute shared mappings for circuit-group lookup construction.

    Parameters
    ----------
    branches : DataFrame[BranchElectricalCircuitGroupSchema]
        Branch table annotated with electrical circuit group ids.
    switches : DataFrame[SwitchElectricalCircuitGroupSchema]
        Switch table annotated with circuit group ids on both bus-breaker sides.
    injection : DataFrame[InjectionElectricalCircuitGroupSchema]
        Injection table annotated with electrical circuit group ids.

    Returns
    -------
    PreparedCircuitGroupLookupData
        Shared branch, switch, injection, and busbar mappings reused by the
        busbar and branch lookup builders.
    """
    busbar_to_primary_group, busbar_to_asset_groups = _build_busbar_lookup_maps(switches=switches, injection=injection)
    branch_group_values = branches["electrical_circuit_group"].to_numpy(copy=False)
    branch_ids = branches.index.to_numpy(dtype=object, copy=False)
    branch_mask = pd.notna(branch_group_values)
    branch_to_group = {
        branch_id: int(component_id)
        for branch_id, component_id in zip(
            branch_ids[branch_mask], np.asarray(branch_group_values[branch_mask], dtype=np.int64), strict=False
        )
    }
    group_to_branches = _group_values_by_key(branch_group_values, branch_ids)

    switch_ids = switches.index.to_numpy(dtype=object, copy=False)
    group_to_switches: dict[int, list[str]] = {}
    for grouped_switches in (
        _group_values_by_key(switches["electrical_circuit_group_bus1"].to_numpy(copy=False), switch_ids),
        _group_values_by_key(switches["electrical_circuit_group_bus2"].to_numpy(copy=False), switch_ids),
    ):
        for component_id, grouped_switch_ids in grouped_switches.items():
            group_to_switches.setdefault(component_id, []).extend(grouped_switch_ids)
    group_to_switches = {
        component_id: _dedupe_preserve_order(switch_ids) for component_id, switch_ids in group_to_switches.items()
    }

    busbar_mask = (injection["type"] == "BUSBAR_SECTION").to_numpy(copy=False)
    injection_ids = injection.index.to_numpy(dtype=object, copy=False)
    injection_group_values = injection["electrical_circuit_group"].to_numpy(copy=False)
    group_to_busbar_sections = _group_values_by_key(injection_group_values[busbar_mask], injection_ids[busbar_mask])
    group_to_injections = _group_values_by_key(injection_group_values[~busbar_mask], injection_ids[~busbar_mask])

    return PreparedCircuitGroupLookupData(
        branch_to_group=branch_to_group,
        busbar_to_primary_group=busbar_to_primary_group,
        busbar_to_asset_groups=busbar_to_asset_groups,
        group_to_branches=group_to_branches,
        group_to_switches=group_to_switches,
        group_to_injections=group_to_injections,
        group_to_busbar_sections=group_to_busbar_sections,
    )


def build_busbar_circuit_group_lookup(preprocessed: PreparedCircuitGroupLookupData) -> CircuitGroupLookupIndex:
    """Build the busbar-oriented portion of the circuit-group lookup index.

    Parameters
    ----------
    preprocessed : PreparedCircuitGroupLookupData
        Shared mappings prepared from the branch, switch, and injection tables.

    Returns
    -------
    CircuitGroupLookupIndex
        Partial lookup index containing busbar mappings and the failing element
        and switch expansions reachable from each busbar section.
    """
    busbar_to_failing_elements: dict[str, list[str]] = {}
    busbar_to_failing_switches: dict[str, list[str]] = {}
    for busbar_id, asset_group_ids in preprocessed.busbar_to_asset_groups.items():
        failing_elements: list[str] = []
        failing_switches: list[str] = []
        for asset_group_id in asset_group_ids:
            failing_elements.extend(preprocessed.group_to_branches.get(asset_group_id, []))
            failing_elements.extend(preprocessed.group_to_injections.get(asset_group_id, []))
            failing_switches.extend(preprocessed.group_to_switches.get(asset_group_id, []))
        busbar_to_failing_elements[busbar_id] = _dedupe_preserve_order(failing_elements)
        busbar_to_failing_switches[busbar_id] = _dedupe_preserve_order(failing_switches)

    return CircuitGroupLookupIndex(
        busbar_to_primary_group=preprocessed.busbar_to_primary_group,
        busbar_to_asset_groups=preprocessed.busbar_to_asset_groups,
        group_to_branches=preprocessed.group_to_branches,
        group_to_switches=preprocessed.group_to_switches,
        group_to_injections=preprocessed.group_to_injections,
        group_to_busbar_sections=preprocessed.group_to_busbar_sections,
        busbar_to_failing_elements=busbar_to_failing_elements,
        busbar_to_failing_switches=busbar_to_failing_switches,
    )


def build_branch_circuit_group_lookup(
    preprocessed: PreparedCircuitGroupLookupData,
    busbar_lookup_index: CircuitGroupLookupIndex,
) -> CircuitGroupLookupIndex:
    """Build the branch-oriented portion of the circuit-group lookup index.

    Parameters
    ----------
    preprocessed : PreparedCircuitGroupLookupData
        Shared mappings prepared from the branch, switch, and injection tables.
    busbar_lookup_index : CircuitGroupLookupIndex
        Partial lookup index containing the busbar-level failing element and
        switch expansions needed to expand full group failures.

    Returns
    -------
    CircuitGroupLookupIndex
        Partial lookup index containing branch-to-group mappings and the full
        failing element and switch expansions for each electrical circuit group.
    """
    all_group_ids = (
        set(preprocessed.group_to_branches)
        | set(preprocessed.group_to_switches)
        | set(preprocessed.group_to_injections)
        | set(preprocessed.group_to_busbar_sections)
    )
    group_to_failing_elements: dict[int, list[str]] = {}
    group_to_failing_switches: dict[int, list[str]] = {}
    for component_id in all_group_ids:
        failing_elements = list(preprocessed.group_to_branches.get(component_id, []))
        failing_switches = list(preprocessed.group_to_switches.get(component_id, []))
        for busbar_id in preprocessed.group_to_busbar_sections.get(component_id, []):
            failing_elements.extend(busbar_lookup_index.busbar_to_failing_elements.get(busbar_id, []))
            failing_switches.extend(busbar_lookup_index.busbar_to_failing_switches.get(busbar_id, []))
        group_to_failing_elements[component_id] = _dedupe_preserve_order(failing_elements)
        group_to_failing_switches[component_id] = _dedupe_preserve_order(failing_switches)

    return CircuitGroupLookupIndex(
        branch_to_group=preprocessed.branch_to_group,
        group_to_branches=preprocessed.group_to_branches,
        group_to_switches=preprocessed.group_to_switches,
        group_to_injections=preprocessed.group_to_injections,
        group_to_busbar_sections=preprocessed.group_to_busbar_sections,
        group_to_failing_elements=group_to_failing_elements,
        group_to_failing_switches=group_to_failing_switches,
    )


def build_circuit_group_lookup_index(
    branches: pat.DataFrame[BranchElectricalCircuitGroupSchema],
    switches: pat.DataFrame[SwitchElectricalCircuitGroupSchema],
    injection: pat.DataFrame[InjectionElectricalCircuitGroupSchema],
) -> CircuitGroupLookupIndex:
    """Build the complete circuit-group lookup index used by query helpers.

    Parameters
    ----------
    branches : DataFrame[BranchElectricalCircuitGroupSchema]
        Branch table annotated with electrical circuit group ids.
    switches : DataFrame[SwitchElectricalCircuitGroupSchema]
        Switch table annotated with circuit group ids on both bus-breaker sides.
    injection : DataFrame[InjectionElectricalCircuitGroupSchema]
        Injection table annotated with electrical circuit group ids.

    Returns
    -------
    CircuitGroupLookupIndex
        Full lookup representation for resolving circuit-group contents and the
        failing elements or switches induced by branch and busbar lookups.
    """
    preprocessed = preprocess_circuit_group_lookup(branches=branches, switches=switches, injection=injection)
    busbar_lookup_index = build_busbar_circuit_group_lookup(preprocessed=preprocessed)
    branch_lookup_index = build_branch_circuit_group_lookup(
        preprocessed=preprocessed,
        busbar_lookup_index=busbar_lookup_index,
    )

    return CircuitGroupLookupIndex(
        branch_to_group=preprocessed.branch_to_group,
        busbar_to_primary_group=preprocessed.busbar_to_primary_group,
        busbar_to_asset_groups=preprocessed.busbar_to_asset_groups,
        group_to_branches=preprocessed.group_to_branches,
        group_to_switches=preprocessed.group_to_switches,
        group_to_injections=preprocessed.group_to_injections,
        group_to_busbar_sections=preprocessed.group_to_busbar_sections,
        group_to_failing_elements=branch_lookup_index.group_to_failing_elements,
        group_to_failing_switches=branch_lookup_index.group_to_failing_switches,
        busbar_to_failing_elements=busbar_lookup_index.busbar_to_failing_elements,
        busbar_to_failing_switches=busbar_lookup_index.busbar_to_failing_switches,
    )


def build_circuit_group_map(identified_circuit_groups: CircuitGroupLookupIndex | object) -> ElectricalCircuitGroupMap:
    """Build a group-id keyed circuit-group map from lookup data or a result bundle.

    Parameters
    ----------
    identified_circuit_groups : CircuitGroupLookupIndex | object
        Either a direct lookup index or an object exposing a ``lookup_index``
        attribute, such as ``ElectricalCircuitGroupIdentification``.

    Returns
    -------
    ElectricalCircuitGroupMap
        Mapping from electrical circuit group id to the grouped branch,
        switch, injection, and busbar-section ids.
    """
    lookup_index = getattr(identified_circuit_groups, "lookup_index", identified_circuit_groups)
    all_group_ids = (
        set(lookup_index.group_to_branches)
        | set(lookup_index.group_to_switches)
        | set(lookup_index.group_to_injections)
        | set(lookup_index.group_to_busbar_sections)
    )
    return {
        group_id: ElectricalCircuitGroup(
            branches=list(lookup_index.group_to_branches.get(group_id, [])),
            switches=list(lookup_index.group_to_switches.get(group_id, [])),
            injections=list(lookup_index.group_to_injections.get(group_id, [])),
            busbar_section=list(lookup_index.group_to_busbar_sections.get(group_id, [])),
        )
        for group_id in all_group_ids
    }


def get_failing_elements_by_branch_ids(
    branch_ids: list[str], lookup_index: CircuitGroupLookupIndex, include_busbar_id: bool = False
) -> FailingElementsByLookupId:
    """Get failing element ids for each requested branch id.

    Parameters
    ----------
    branch_ids : list[str]
        Branch ids to resolve.
    lookup_index : CircuitGroupLookupIndex
        Precomputed circuit-group lookup index.
    include_busbar_id : bool, optional
        Whether to append directly connected busbar-section ids to each returned
        failing-element list.

    Returns
    -------
    FailingElementsByLookupId
        Mapping from each branch id to the list of failing branch, injection,
        and optionally busbar-section ids.

    Raises
    ------
    ValueError
        If one or more branch ids are not present in the lookup index.
    """
    missing_branch_ids = [branch_id for branch_id in branch_ids if branch_id not in lookup_index.branch_to_group]
    if missing_branch_ids:
        raise ValueError(f"Branch IDs not found in circuit-group lookup index: {missing_branch_ids}")

    if not include_busbar_id:
        return {
            branch_id: lookup_index.group_to_failing_elements[lookup_index.branch_to_group[branch_id]]
            for branch_id in branch_ids
        }

    failing_elements_with_busbar_by_group: dict[int, list[str]] = {}
    for component_id in {lookup_index.branch_to_group[branch_id] for branch_id in branch_ids}:
        failing_elements = list(lookup_index.group_to_failing_elements.get(component_id, []))
        failing_elements.extend(lookup_index.group_to_busbar_sections.get(component_id, []))
        failing_elements_with_busbar_by_group[component_id] = _dedupe_preserve_order(failing_elements)

    return {
        branch_id: failing_elements_with_busbar_by_group[lookup_index.branch_to_group[branch_id]] for branch_id in branch_ids
    }


def get_failing_switches_by_branch_ids(
    branch_ids: list[str], lookup_index: CircuitGroupLookupIndex
) -> FailingSwitchesByLookupId:
    """Get failing switch ids for each requested branch id.

    Parameters
    ----------
    branch_ids : list[str]
        Branch ids to resolve.
    lookup_index : CircuitGroupLookupIndex
        Precomputed circuit-group lookup index.

    Returns
    -------
    FailingSwitchesByLookupId
        Mapping from each branch id to the list of failing switch ids.

    Raises
    ------
    ValueError
        If one or more branch ids are not present in the lookup index.
    """
    missing_branch_ids = [branch_id for branch_id in branch_ids if branch_id not in lookup_index.branch_to_group]
    if missing_branch_ids:
        raise ValueError(f"Branch IDs not found in circuit-group lookup index: {missing_branch_ids}")

    return {
        branch_id: lookup_index.group_to_failing_switches[lookup_index.branch_to_group[branch_id]]
        for branch_id in branch_ids
    }


def get_failing_elements_by_busbar_ids(
    busbar_section_ids: list[str], lookup_index: CircuitGroupLookupIndex
) -> FailingElementsByLookupId:
    """Get failing element ids for each requested busbar-section id.

    Parameters
    ----------
    busbar_section_ids : list[str]
        Busbar-section ids to resolve.
    lookup_index : CircuitGroupLookupIndex
        Precomputed circuit-group lookup index.

    Returns
    -------
    FailingElementsByLookupId
        Mapping from each busbar-section id to the list of failing branch and
        injection ids connected through the outage expansion.

    Raises
    ------
    ValueError
        If one or more busbar-section ids are not present in the lookup index.
    """
    missing_busbar_ids = [
        busbar_section_id
        for busbar_section_id in busbar_section_ids
        if busbar_section_id not in lookup_index.busbar_to_failing_elements
    ]
    if missing_busbar_ids:
        raise ValueError(f"Busbar sections not found in circuit-group lookup index: {missing_busbar_ids}")

    return {
        busbar_section_id: lookup_index.busbar_to_failing_elements[busbar_section_id]
        for busbar_section_id in busbar_section_ids
    }


def get_failing_switches_by_busbar_ids(
    busbar_section_ids: list[str], lookup_index: CircuitGroupLookupIndex
) -> FailingSwitchesByLookupId:
    """Get failing switch ids for each requested busbar-section id.

    Parameters
    ----------
    busbar_section_ids : list[str]
        Busbar-section ids to resolve.
    lookup_index : CircuitGroupLookupIndex
        Precomputed circuit-group lookup index.

    Returns
    -------
    FailingSwitchesByLookupId
        Mapping from each busbar-section id to the list of failing switch ids.

    Raises
    ------
    ValueError
        If one or more busbar-section ids are not present in the lookup index.
    """
    missing_busbar_ids = [
        busbar_section_id
        for busbar_section_id in busbar_section_ids
        if busbar_section_id not in lookup_index.busbar_to_failing_switches
    ]
    if missing_busbar_ids:
        raise ValueError(f"Busbar sections not found in circuit-group lookup index: {missing_busbar_ids}")

    return {
        busbar_section_id: lookup_index.busbar_to_failing_switches[busbar_section_id]
        for busbar_section_id in busbar_section_ids
    }
