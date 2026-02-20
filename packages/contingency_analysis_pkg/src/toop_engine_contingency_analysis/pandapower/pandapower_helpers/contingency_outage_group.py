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

from typing import Iterable, Set, Tuple

import pandapower as pp
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    PandapowerElements,
)
from toop_engine_grid_helpers.pandapower.outage_group import (
    OUTAGE_GROUP_SEPARATOR,
    build_connected_components_for_contingency_analysis,
    elem_node_id,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id


def get_grid_element(net: pp.pandapowerNet, element: int, el_type: str) -> PandapowerElements:
    """
    Retrieve a grid element from a pandapower network and wrap it into a PandapowerElements object.

    The function accesses the corresponding pandapower table (e.g. "line",
    "bus", "trafo") using the provided element type, extracts the row by index,
    and constructs a PandapowerElements instance with a globally unique ID.

    Args:
        net (pp.pandapowerNet):
            The pandapower network containing element tables.
        element (int):
            Row index of the element in the corresponding pandapower table.
        el_type (str):
            Name of the pandapower table where the element is stored
            (e.g. "bus", "line", "trafo", "switch").

    Returns
    -------
        PandapowerElements:
            A wrapped representation of the grid element containing:
            - unique_id: Globally unique identifier
            - table: Table name (element type)
            - table_id: Row index within the table
            - name: Element name from the pandapower table
    """
    el_df = getattr(net, el_type)
    el = el_df.loc[element]
    global_id = get_globally_unique_id(element, el_type)

    return PandapowerElements(unique_id=global_id, table=el_type, table_id=element, name=str(el["name"]))


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
) -> list[PandapowerContingency]:
    """
    Build outage groups for each contingency based on network connected components.

    For every contingency, all its elements are mapped to the connected component(s)
    they belong to. A new contingency is then created whose elements are the union
    of all grid elements contained in those component(s).

    If an element cannot be mapped to any existing component, a new single-node
    component is created for it (typically representing an isolated busbar outage).

    Args:
        net (pp.pandapowerNet):
            The pandapower network model used to compute connected components and
            retrieve grid elements.
        contingencies (List[PandapowerContingency]):
            List of contingencies. Each contingency contains a set of elements
            (with `id` and `type`) that define the initial outage.

    Returns
    -------
        List[PandapowerContingency]:
            A list of new contingencies where each contingency contains all elements
            belonging to the outage group(s) (connected component unions) associated
            with the original contingency. Metadata such as `unique_id`, `name`,
            and `va_diff_info` are preserved.
    """
    # --- Step 1: Build a fast lookup map from node -> component index ---
    connected_components = build_connected_components_for_contingency_analysis(net)
    node_to_component = {node: comp_idx for comp_idx, component in enumerate(connected_components) for node in component}

    # --- Step 2: Track results ---
    grouped_contingencies: list[PandapowerContingency] = []
    total_outage_groups: list[tuple[PandapowerContingency, set[int]]] = []

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

        total_outage_groups.append((contingency, contingency_components))

    # --- Step 4: Build grouped contingencies ---
    for contingency, outage_group in total_outage_groups:
        res_elements = []
        for comp_idx in outage_group:
            res_elements.extend(_elements_in_component(connected_components, comp_idx))

        # Convert (id, type) tuples into actual grid elements
        elements = []
        for el_id, el_type in res_elements:
            elements.append(get_grid_element(net, el_id, el_type))

        grouped_contingencies.append(
            PandapowerContingency(
                unique_id=contingency.unique_id,
                name=contingency.name,
                elements=elements,
                va_diff_info=contingency.va_diff_info,
            )
        )

    return grouped_contingencies
