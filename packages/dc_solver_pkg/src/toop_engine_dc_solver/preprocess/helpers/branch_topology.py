# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds functions to identify bridges inside a network."""

import numpy as np
from beartype.typing import Any, Sequence
from jaxtyping import Bool, Int


def zip_branch_lists(
    branches_from_node_list: list[Int[np.ndarray, " n_from_branch_at_node"]],
    branches_to_node_list: list[Int[np.ndarray, " n_to_branch_at_node"]],
) -> list[Int[np.ndarray, " n_branches_at_node"]]:
    """Get a combined list of branch arrays starting or leaving a given node.

    Parameters
    ----------
    branches_from_node_list : list[Int[np.ndarray, " n_from_branch_at_node"]]
        A list of the length of relevant nodes. Contains the branches leaving the node.
    branches_to_node_list: list[Int[np.ndarray, " n_to_branch_at_node"]]
        A list of the length of relevant nodes. Contains the branches entering the node.

    Returns
    -------
    list[Int[np.ndarray, " n_branches_at_node"]]
        A list of the length of relevant nodes. Contains arrays of all branches going from or towards the given node
    """
    return [
        np.concatenate([branches_from, branches_to])
        for branches_from, branches_to in zip(branches_from_node_list, branches_to_node_list, strict=True)
    ]


def get_branch_direction(
    branches_at_node_list: list[Int[np.ndarray, " n_branches_at_node"]],
    branches_from_node_list: list[Int[np.ndarray, " n_from_branch_at_node"]],
) -> list[Bool[np.ndarray, " n_branches_at_node"]]:
    """Get a list of boolean arrays indication if a branch is starting at a node (True) of leaves the node

    Parameters
    ----------
    branches_at_node_list : list[Int[np.ndarray, " n_branches_at_node"]]
        A list of the length of relevant nodes. Contains arrays of all branches going from or towards the given node
    branches_from_node_list : list[Int[np.ndarray, " n_from_branch_at_node"]]
        A list of the length of relevant nodes. Contains the branches leaving the node.

    Returns
    -------
    list[Bool[np.ndarray, " n_branches_at_node"]]
        A boolean Array indicating the wether the according branch
        in the branches_at_node list is entering or leaving the given node
    """
    branch_direction = [
        np.isin(branches_at_node, branches_from_node)
        for branches_at_node, branches_from_node in zip(branches_at_node_list, branches_from_node_list, strict=True)
    ]
    return branch_direction


def get_masked_elements(
    mask: Bool[np.ndarray, " n_elements"], types: Sequence[Any], ids: Sequence[Any]
) -> tuple[Sequence[Any], Sequence[Any]]:
    """Apply a boolean mask to a list of element types and ids and return the masked elements.

    Parameters
    ----------
    mask : Bool[np.ndarray, " n_elements"]
        A boolean mask to apply to the types and ids.
    types : Sequence[Any]
        A list of element types, length n_elements.
    ids : Sequence[Any]
        A list of element ids, length n_elements.

    Returns
    -------
    Sequence[Any]
        The masked element types.
    Sequence[Any]
        The masked element ids.
    """
    return np.array(types)[mask].tolist(), np.array(ids)[mask].tolist()
