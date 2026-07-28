# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for busbar-section outage expansion."""

import pandera as pa
import pandera.typing as pat
from toop_engine_grid_helpers.powsybl.network_graph.electrical_circuit_groups.types import (
    AssetBreakerSchema,
    BranchElectricalCircuitGroupSchema,
    BusbarCouplerSchema,
    BusbarSectionOutageGroup,
    BusbarSectionOutageGroups,
    ElectricalCircuitGroupMap,
    InjectionElectricalCircuitGroupSchema,
    SwitchElectricalCircuitGroupSchema,
)


@pa.check_types
def extract_secondary_busbar_section_circuit_groups(
    switches: pat.DataFrame[SwitchElectricalCircuitGroupSchema],
    injection: pat.DataFrame[InjectionElectricalCircuitGroupSchema],
) -> tuple[pat.DataFrame[BusbarCouplerSchema], pat.DataFrame[AssetBreakerSchema]]:
    """Extract secondary busbar section circuit groups from switches and busbar sections.

    If a busbar section is outaged, one may want to also know which connected branches and injections are affected.

    Parameters
    ----------
    switches : pd.DataFrame
        DataFrame containing switches with their electrical circuit groups.
    injection : pd.DataFrame
        DataFrame containing injections, including busbar sections with their electrical circuit groups.

    Returns
    -------
    tuple[DataFrame[BusbarCouplerSchema], DataFrame[AssetBreakerSchema]]
        Busbar couplers and asset breakers with their corresponding circuit groups.
    """
    busbar_sections = injection[injection["type"] == "BUSBAR_SECTION"]
    switch_sides = switches.reset_index().merge(
        busbar_sections[["bus_breaker_bus_id", "electrical_circuit_group"]],
        how="left",
        left_on="bus_breaker_bus1_id",
        right_on="bus_breaker_bus_id",
        suffixes=["_side1", "_side2"],
    )
    switch_sides = switch_sides.merge(
        busbar_sections[["bus_breaker_bus_id", "electrical_circuit_group"]],
        how="left",
        left_on="bus_breaker_bus2_id",
        right_on="bus_breaker_bus_id",
        suffixes=["_side1", "_side2"],
    )
    # get busbar couplers where both sides have a busbar section
    busbar_coupler = switch_sides[
        (switch_sides["electrical_circuit_group_side1"].notnull())
        & (switch_sides["electrical_circuit_group_side2"].notnull())
    ].copy()
    busbar_coupler = busbar_coupler[
        [
            "id",
            "bus_breaker_bus1_id",
            "bus_breaker_bus2_id",
            "kind",
            "electrical_circuit_group_bus1",
            "electrical_circuit_group_bus2",
        ]
    ]

    # get asset breakers where one side has a busbar section and the other side does not
    cond_side1 = switch_sides["electrical_circuit_group_side1"].notnull()
    cond_side2 = switch_sides["electrical_circuit_group_side2"].notnull()
    asset_breakers = switch_sides[(cond_side1 | cond_side2) & ~(cond_side1 & cond_side2)].copy()
    asset_breakers["electrical_circuit_group_busbar"] = asset_breakers["electrical_circuit_group_side1"]
    cond = asset_breakers["electrical_circuit_group_busbar"].isnull()
    asset_breakers.loc[cond, "electrical_circuit_group_busbar"] = asset_breakers.loc[cond, "electrical_circuit_group_side2"]
    # Identify the non-busbar-side circuit group for each asset breaker.
    cond_side1 = asset_breakers["electrical_circuit_group_bus1"] != asset_breakers["electrical_circuit_group_busbar"]
    cond_side2 = asset_breakers["electrical_circuit_group_bus2"] != asset_breakers["electrical_circuit_group_busbar"]
    asset_breakers.loc[cond_side1, "asset_circuit_group"] = asset_breakers.loc[cond_side1, "electrical_circuit_group_bus1"]
    asset_breakers.loc[cond_side2, "asset_circuit_group"] = asset_breakers.loc[cond_side2, "electrical_circuit_group_bus2"]
    asset_breakers["asset_circuit_group"] = asset_breakers["asset_circuit_group"].astype(int)
    asset_breakers["electrical_circuit_group_busbar"] = asset_breakers["electrical_circuit_group_busbar"].astype(int)
    asset_breakers = asset_breakers[
        [
            "id",
            "bus_breaker_bus1_id",
            "bus_breaker_bus2_id",
            "kind",
            "electrical_circuit_group_busbar",
            "asset_circuit_group",
        ]
    ]

    # set index back to id for both dataframes
    busbar_coupler.set_index("id", inplace=True)
    asset_breakers.set_index("id", inplace=True)
    return busbar_coupler, asset_breakers


@pa.check_types
def get_outage_group_ids_by_busbar_section_id(
    injection: pat.DataFrame[InjectionElectricalCircuitGroupSchema],
    switches: pat.DataFrame[SwitchElectricalCircuitGroupSchema],
) -> BusbarSectionOutageGroups:
    """Get outage group IDs for a given busbar section.

    Parameters
    ----------
    injection : pd.DataFrame
        DataFrame containing injection information.
    switches : pd.DataFrame
        DataFrame containing switch information.

    Returns
    -------
    dict[str, BusbarSectionOutageGroup]
        Busbar-section outage expansion metadata keyed by busbar section id.
    """
    busbar_outage_groups: BusbarSectionOutageGroups = {}
    busbar_coupler, asset_breakers = extract_secondary_busbar_section_circuit_groups(switches=switches, injection=injection)

    busbar_sections = injection[injection["type"] == "BUSBAR_SECTION"]
    for busbar_id, row in busbar_sections.iterrows():
        busbar_circuit_group = int(row["electrical_circuit_group"])
        # get all busbar couplers
        busbar_couplers = busbar_coupler[
            (busbar_coupler["electrical_circuit_group_bus1"] == busbar_circuit_group)
            | (busbar_coupler["electrical_circuit_group_bus2"] == busbar_circuit_group)
        ]
        busbar_couplers_ids = list(busbar_couplers.index)
        # get all asset breakers
        busbar_asset_breakers = asset_breakers[(asset_breakers["electrical_circuit_group_busbar"] == busbar_circuit_group)]
        asset_breakers_ids = list(busbar_asset_breakers.index)
        asset_circuit_groups = [int(group_id) for group_id in busbar_asset_breakers["asset_circuit_group"].tolist()]

        busbar_outage_groups[busbar_id] = BusbarSectionOutageGroup(
            primary_circuit_group=busbar_circuit_group,
            busbar_couplers=busbar_couplers_ids,
            primary_asset_breakers=asset_breakers_ids,
            asset_circuit_groups=asset_circuit_groups,
        )
    return busbar_outage_groups


def get_all_failing_elements_by_busbar_section_id(
    busbar_section_id: str, busbar_outage_groups: BusbarSectionOutageGroups, outage_groups: ElectricalCircuitGroupMap
) -> list[str]:
    """Get all failing elements for a given busbar section.

    Parameters
    ----------
    busbar_section_id : str
        The ID of the busbar section.
    busbar_outage_groups : BusbarSectionOutageGroups
        A dictionary mapping busbar section IDs to their corresponding outage group information.
    outage_groups : ElectricalCircuitGroupMap
        A dictionary containing outage groups with their corresponding branch IDs and switch IDs.

    Returns
    -------
    list[str]
        A list of all failing element IDs for the given busbar section.
    """
    if busbar_section_id not in busbar_outage_groups:
        raise ValueError(f"Busbar section {busbar_section_id} not found in busbar_outage_groups.")

    busbar_info = busbar_outage_groups[busbar_section_id]
    asset_circuit_groups = busbar_info.asset_circuit_groups

    failing_elements = []

    # Add branches and switches from the asset circuit groups
    for asset_circuit_group in asset_circuit_groups:
        if asset_circuit_group in outage_groups:
            failing_elements.extend(outage_groups[asset_circuit_group].branches)
            failing_elements.extend(outage_groups[asset_circuit_group].injections)
    # de-duplicate the list of failing elements
    failing_elements = list(set(failing_elements))
    return failing_elements


def get_all_failing_switches_by_busbar_section_id(
    busbar_section_id: str, busbar_outage_groups: BusbarSectionOutageGroups, outage_groups: ElectricalCircuitGroupMap
) -> list[str]:
    """Get all failing switches for a given busbar section.

    Parameters
    ----------
    busbar_section_id : str
        The ID of the busbar section.
    busbar_outage_groups : dict
        A dictionary mapping busbar section IDs to their corresponding outage group information.
    outage_groups : dict
        A dictionary containing outage groups with their corresponding branch IDs and switch IDs.

    Returns
    -------
    list[str]
        A list of all failing switch IDs for the given busbar section.
    """
    if busbar_section_id not in busbar_outage_groups:
        raise ValueError(f"Busbar section {busbar_section_id} not found in busbar_outage_groups.")

    busbar_info = busbar_outage_groups[busbar_section_id]
    asset_circuit_groups = busbar_info.asset_circuit_groups

    failing_switches = []

    # Add switches from the asset circuit groups
    for asset_circuit_group in asset_circuit_groups:
        if asset_circuit_group in outage_groups:
            failing_switches.extend(outage_groups[asset_circuit_group].switches)

    # de-duplicate the list of failing switches
    failing_switches = list(set(failing_switches))

    return failing_switches


@pa.check_types
def get_all_failing_elements_by_branch_id(
    branch_id: str,
    outage_groups: ElectricalCircuitGroupMap,
    branches: pat.DataFrame[BranchElectricalCircuitGroupSchema],
    busbar_outage_groups: BusbarSectionOutageGroups,
    include_busbar_id: bool = False,
) -> list[str]:
    """Get all failing elements for a given busbar section by branch ID.

    Get all failing branches of this group and if a busbar is directly connected -> add outage cases as well.

    Parameters
    ----------
    branch_id : str
        The ID of the branch.
    outage_groups : dict
        A dictionary containing outage groups with their corresponding branch IDs and switch IDs.
    branches : pd.DataFrame
        DataFrame containing branch information.
    busbar_outage_groups : BusbarSectionOutageGroups
        Busbar-section outage expansion metadata keyed by busbar section id.
    include_busbar_id : bool, optional
        Whether to include the busbar section ID in the list of failing elements. Default is False

    Returns
    -------
    list[str]
        A list of all failing element IDs for the given busbar section.
    """
    if branch_id not in branches.index:
        raise ValueError(f"Branch {branch_id} not found in branches DataFrame.")

    branch_info = branches.loc[branch_id]
    branch_circuit_groups = int(branch_info["electrical_circuit_group"])

    failing_elements = []

    # Add branches from the asset circuit groups
    if branch_circuit_groups in outage_groups:
        failing_elements.extend(outage_groups[branch_circuit_groups].branches)
    # check if a busbar is directly connected -> add outage cases as well
    if len(outage_groups[branch_circuit_groups].busbar_section) > 0:
        busbar_sections = outage_groups[branch_circuit_groups].busbar_section
        for busbar_id in busbar_sections:
            if busbar_id in busbar_outage_groups:
                failing_elements.extend(
                    get_all_failing_elements_by_busbar_section_id(
                        busbar_section_id=busbar_id, busbar_outage_groups=busbar_outage_groups, outage_groups=outage_groups
                    )
                )
    if include_busbar_id:
        failing_elements.extend(busbar_sections)

    # de-duplicate the list of failing elements
    failing_elements = list(set(failing_elements))

    return failing_elements


@pa.check_types
def get_all_failing_switches_by_branch_id(
    branch_id: str,
    outage_groups: ElectricalCircuitGroupMap,
    branches: pat.DataFrame[BranchElectricalCircuitGroupSchema],
    busbar_outage_groups: BusbarSectionOutageGroups,
) -> list[str]:
    """Get all failing switches for a given busbar section by branch ID.

    Get all failing switches of this group and if a busbar is directly connected -> add outage cases as well.

    Parameters
    ----------
    branch_id : str
        The ID of the branch.
    outage_groups : dict
        A dictionary containing outage groups with their corresponding branch IDs and switch IDs.
    branches : pd.DataFrame
        DataFrame containing branch information.
    busbar_outage_groups : BusbarSectionOutageGroups
        Busbar-section outage expansion metadata keyed by busbar section id.

    Returns
    -------
    list[str]
        A list of all failing switch IDs for the given busbar section.
    """
    if branch_id not in branches.index:
        raise ValueError(f"Branch {branch_id} not found in branches DataFrame.")

    branch_info = branches.loc[branch_id]
    branch_circuit_groups = int(branch_info["electrical_circuit_group"])

    failing_switches = []

    # Add switches from the asset circuit groups
    if branch_circuit_groups in outage_groups:
        failing_switches.extend(outage_groups[branch_circuit_groups].switches)
    # check if a busbar is directly connected -> add outage cases as well
    if len(outage_groups[branch_circuit_groups].busbar_section) > 0:
        busbar_sections = outage_groups[branch_circuit_groups].busbar_section
        for busbar_id in busbar_sections:
            if busbar_id in busbar_outage_groups:
                failing_switches.extend(
                    get_all_failing_switches_by_busbar_section_id(busbar_id, busbar_outage_groups, outage_groups)
                )

    # de-duplicate the list of failing switches
    failing_switches = list(set(failing_switches))

    return failing_switches
