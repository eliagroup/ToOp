# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Human-readable branch names, shared by the DC solver preprocessing and the input N-1 validation of the importer."""

import pandas as pd
import structlog
from beartype.typing import Literal
from pypowsybl.network import Network

logger = structlog.get_logger(__name__)

BranchType = Literal["LINE", "TWO_WINDINGS_TRANSFORMER", "TIE_LINE"]

# The source formats whose branch names are derived from the grid's own names
NAMED_SOURCE_FORMATS = ("UCTE", "CGMES", "hybrid")


def get_cgmes_ids(merged_net: Network) -> list[str]:
    """Get the CGMES IDs from the merged network.

    This function looks for the sub-network with CGMES format and returns the IDs of the elements in that sub-network.

    Parameters
    ----------
    merged_net: Network
        The merged powsybl network containing sub-networks of different formats.

    Returns
    -------
    list[str]
        The list of CGMES IDs from the CGMES sub-network.
    """
    cgmes_nets = []
    for sub_net_id in merged_net.get_sub_networks().index.values:
        sub_net = merged_net.get_sub_network(sub_net_id)
        if sub_net._source_format == "CGMES":
            cgmes_nets.append(sub_net)
    if len(cgmes_nets) == 0:
        raise ValueError("No CGMES sub-network found in the merged network.")
    if len(cgmes_nets) > 1:
        logger.warning(
            f"Multiple CGMES sub-networks found in the merged network. Returning IDs from the first one. "
            f"Sub-network IDs: {[net.id for net in cgmes_nets]}"
        )
    return cgmes_nets[0].get_identifiables().index.tolist()


def get_branch_names(net: Network, branch_type: BranchType) -> pd.Series:
    """Get the name of every branch of one type, depending on the source format of the grid.

    - UCTE: lines and transformers are named ``<id>: <elementName>``, tie lines by their id.
    - CGMES: lines and transformers keep their ``name``, tie lines are named ``<name 1> ## <name 2>`` after their
      boundary lines.
    - hybrid (merged UCTE and CGMES): each branch is named as in its own source format.
    - any other format: ``<bus 1> ## <bus 2> ## <name>``, and for tie lines
      ``<bus 1> ## <pairing key> ## <bus 2> ## <name 1> ## <name 2>``, using the bus-breaker bus ids and the
      ``elementName`` where the grid has one.

    Parameters
    ----------
    net : Network
        The powsybl network.
    branch_type : BranchType
        The Powsybl element type of the branches.

    Returns
    -------
    pd.Series
        The branch names, indexed by branch id.
    """
    if branch_type == "TIE_LINE":
        return _get_tie_line_names(net)
    branches = (
        net.get_lines(all_attributes=True) if branch_type == "LINE" else net.get_2_windings_transformers(all_attributes=True)
    )
    element_names = branches.get("elementName", branches["name"])
    source_format = net._source_format
    if source_format in ("UCTE", "hybrid"):
        ucte_names = branches.index.to_series() + ": " + element_names
        if source_format == "UCTE":
            return ucte_names
        return branches["name"].where(branches.index.isin(get_cgmes_ids(net)), ucte_names)
    if source_format == "CGMES":
        return branches["name"]
    return branches["bus_breaker_bus1_id"] + " ## " + branches["bus_breaker_bus2_id"] + " ## " + element_names


def _get_tie_line_names(net: Network) -> pd.Series:
    """Get the tie line names, see :func:`get_branch_names`."""
    tie_lines = net.get_tie_lines(all_attributes=True)
    if tie_lines.empty:
        return pd.Series(index=tie_lines.index, dtype=str)
    boundary_lines = net.get_boundary_lines(all_attributes=True)
    side_1 = boundary_lines.reindex(tie_lines["boundary_line1_id"]).set_axis(tie_lines.index)
    side_2 = boundary_lines.reindex(tie_lines["boundary_line2_id"]).set_axis(tie_lines.index)
    cgmes_names = side_1["name"] + " ## " + side_2["name"]
    source_format = net._source_format
    if source_format in ("UCTE", "hybrid"):
        ucte_names = tie_lines.index.to_series()
        if source_format == "UCTE":
            return ucte_names
        cgmes_ids = get_cgmes_ids(net)
        is_cgmes = tie_lines["boundary_line1_id"].isin(cgmes_ids) | tie_lines["boundary_line2_id"].isin(cgmes_ids)
        return cgmes_names.where(is_cgmes, ucte_names)
    if source_format == "CGMES":
        return cgmes_names
    return (
        side_1["bus_breaker_bus_id"]
        + " ## "
        + tie_lines["pairing_key"]
        + " ## "
        + side_2["bus_breaker_bus_id"]
        + " ## "
        + side_1.get("elementName", side_1["name"])
        + " ## "
        + side_2.get("elementName", side_2["name"])
    )
