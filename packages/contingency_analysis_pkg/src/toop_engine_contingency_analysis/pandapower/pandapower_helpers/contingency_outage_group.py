# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""
Utilities for mapping contingencies to outage groups in a pandapower network.

The module builds connected-component-based outage sets, extracts elements
belonging to those components, and converts them into PandapowerElements objects.
Used to expand initial contingencies into full topology-aware outage groups.
"""

import uuid
from collections import defaultdict

import pandapower as pp
from beartype.typing import Iterable, Set, Tuple
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    PandapowerContingencyGroup,
    PandapowerElements,
)
from toop_engine_grid_helpers.pandapower.outage_group import (
    OUTAGE_GROUP_SEPARATOR,
    build_connected_components_for_contingency_analysis,
    elem_node_id,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id

BUS_RELATED_ELEMENT_TYPES = {"gen", "ext_grid", "sgen", "shunt", "ward", "xward"}


def get_grid_element(
    net: pp.pandapowerNet,
    element: int,
    el_type: str,
) -> PandapowerElements:
    """
    Resolve a pandapower element to its effective table representation and

    return it as a PandapowerElements object.

    Some element types (e.g. gen, ext_grid, sgen, shunt, ward, xward) are
    internally mapped to their corresponding bus. In such cases:
        - table is set to "bus"
        - table_id becomes the associated bus index
        - unique_id is generated for the bus
        - name is taken from the bus

    For all other element types:
        - table remains el_type
        - table_id is the original element index
        - unique_id is generated for that element
        - name is taken from the element

    Args:
        net (pp.pandapowerNet): The pandapower network.
        element (int): Index of the element in its table.
        el_type (str): Name of the pandapower element table (e.g. "line", "bus", "gen").

    Returns
    -------
        PandapowerElements: Normalized representation of the element.
    """
    el_df = getattr(net, el_type)
    el = el_df.loc[element]

    if el_type in BUS_RELATED_ELEMENT_TYPES:
        bus_id = int(el.bus)
        bus = net.bus.loc[bus_id]

        return PandapowerElements(
            unique_id=get_globally_unique_id(bus_id, "bus"),
            table="bus",
            table_id=bus_id,
            name=str(bus["name"]),
        )

    return PandapowerElements(
        unique_id=get_globally_unique_id(element, el_type),
        table=el_type,
        table_id=element,
        name=str(el["name"]),
    )


def _elements_in_component(comps: Iterable[Iterable[str]], comp_idx: int) -> Set[Tuple[int, str]]:
    """
    Extract grid elements belonging to a given connected component.

    The component is expected to contain node identifiers encoded as strings:
      - Element nodes: "e_<etype>_<idx>"
      - Bus nodes:     "b_<idx>"

    These are converted into (id, type) tuples:
      - ("e&&line&&5")  -> (5, "line")
      - ("b&&12")      -> (12, "bus")

    Args:
        comps (Iterable[Iterable[str]]):
            Collection of connected components, where each component is an
            iterable of node ID strings.
        comp_idx (int):
            Index of the component to process.

    Returns
    -------
        Set[Tuple[int, str]]:
            A set of (element_id, element_type) tuples representing all grid
            elements contained in the specified component. Element types are
            derived from the node prefix ("e_" for general elements, "b_" for bus).
    """
    comp_nodes = comps[comp_idx]
    elems: Set[Tuple[int, str]] = set()

    for node in comp_nodes:
        if node.startswith(f"e{OUTAGE_GROUP_SEPARATOR}"):
            _, et, idx_s = node.split(OUTAGE_GROUP_SEPARATOR, 2)
            elems.add((int(idx_s), et))
        elif node.startswith(f"b{OUTAGE_GROUP_SEPARATOR}"):
            _, idx_s = node.split(OUTAGE_GROUP_SEPARATOR, 1)
            elems.add((int(idx_s), "bus"))

    return elems


def get_outage_group_for_contingency(
    net: pp.pandapowerNet, contingencies: list[PandapowerContingency]
) -> list[PandapowerContingencyGroup]:
    """
    Group contingencies by affected network connected components.

    This function maps each contingency to the connected component(s) of the
    network that its elements belong to. Contingencies affecting the same set
    of components are grouped together into a single outage group.

    For each resulting outage group:
    - All original contingencies sharing the same component signature are collected.
    - The resulting group elements are the union of all grid elements contained
      in the corresponding connected components.

    If an element cannot be mapped to any existing component, a new single-node
    component is created for it (e.g., isolated busbar outages).

    Args:
        net (pp.pandapowerNet):
            Pandapower network model used to compute connected components and
            retrieve grid elements.

        contingencies (list[PandapowerContingency]):
            List of contingencies, where each contingency contains a set of
            elements (identified by `table` and `table_id`) defining the outage.

    Returns
    -------
        list[PandapowerContingencyGroup]:
            A list of outage groups. Each group contains:
            - `contingencies`: original contingencies mapped to the same set
              of connected components.
            - `elements`: all grid elements belonging to those components.
    """
    # --- Step 1: Build a fast lookup map from node -> component index ---
    connected_components = build_connected_components_for_contingency_analysis(net)
    node_to_component = {node: comp_idx for comp_idx, component in enumerate(connected_components) for node in component}

    # --- Step 2: Track results ---
    grouped_contingencies: list[PandapowerContingencyGroup] = []
    total_outage_groups = defaultdict(list)

    # --- Step 3: Map each contingency to its component(s) ---
    for contingency in contingencies:
        contingency_components: set[int] = set()

        for element in contingency.elements:
            idx = int(element.table_id)
            etype = element.table

            # Convert element into a node ID
            node_id = elem_node_id("elem", idx, etype)

            # If the node is not in any component, create a new single-node component
            if node_id not in node_to_component:
                new_component = {elem_node_id("bus", idx)}
                connected_components.append(new_component)
                comp_idx = len(connected_components) - 1
                node_to_component[node_id] = comp_idx
            else:
                comp_idx = node_to_component[node_id]

            contingency_components.add(comp_idx)
        # Canonical outage-groups signature for this contingency
        signature = tuple(sorted(contingency_components))
        total_outage_groups[signature].append(contingency)

    # --- Step 4: Build grouped contingencies ---
    for outage_group, group_contingencies in total_outage_groups.items():
        res_elements = []
        for comp_idx in outage_group:
            res_elements.extend(_elements_in_component(connected_components, comp_idx))

        # Convert (id, type) tuples into actual grid elements
        elements = []
        for el_id, el_type in res_elements:
            elements.append(get_grid_element(net, el_id, el_type))
        grouped_contingencies.append(
            PandapowerContingencyGroup(
                contingencies=group_contingencies,
                elements=elements,
                outage_group_id=str(uuid.uuid4()),
            )
        )

    return grouped_contingencies
