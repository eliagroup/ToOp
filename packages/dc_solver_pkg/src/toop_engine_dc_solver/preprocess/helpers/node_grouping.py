# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds a function helping with grouping things by node (like injections and branches)"""

import numpy as np
from jaxtyping import Bool, Int


def group_by_node(
    node_vector: Int[np.ndarray, " n_elements"],
    relevant_node_ids: Int[np.ndarray, " n_filtered_nodes"],
) -> list[Int[np.ndarray, " n_elements_at_node"]]:
    """Get a list of arrays for each node.

    Each array contains the indices of the original array where the node occured

    Parameters
    ----------
    node_vector : Int[np.ndarray, " n_elements"]
        A vector of the nodes the elements are connected to
    relevant_node_ids: Int[np.ndarray, " n_filtered_nodes"]
        Filter to only calculate the elements connected to certain nodes

    Returns
    -------
    list[Int[np.ndarray, " n_elements_at_node"]]
        A list of length n_filtered_nodes.
        Contains the elements from the node_vector for each node.
        Arrays can be of different size depending on the amount of elements
    """
    # Compares f_stat with ids
    # (broadcasts to shape nStat_relevant x nBrh)
    find_node = node_vector == relevant_node_ids[:, np.newaxis]

    # Does flatnonzero (= find) for each row (each relevant station)
    element_by_node = [np.flatnonzero(row) for row in find_node]
    return element_by_node


def get_num_elements_per_node(
    elements_at_node_list: list[Int[np.ndarray, " n_elements_at_node"]],
) -> Int[np.ndarray, " n_nodes"]:
    """Get the number of elements for each node in the given list.

    Parameters
    ----------
    elements_at_node_list : list[Int[np.ndarray, " n_elements_at_node"]]
        A list of the length of relevant nodes. Contains arrays of all elements going from or towards the given node

    Returns
    -------
    Int[np.ndarray, " n_nodes"]
        An Int Array indicating the amount of elements starting or leaving the nodes in the given list
    """
    num_elements_per_node = [branch_array.size for branch_array in elements_at_node_list]
    return np.array(num_elements_per_node, dtype=int)


def convert_boolean_mask_to_index_array(
    mask: Bool[np.ndarray, " n_cases n_elements"], padding: int = -1
) -> Int[np.ndarray, " n_cases max_n_true_elements"]:
    """
    Convert a boolean mask to an index array

    Returns the smallest index array that contains the indices of the True elements in the mask.
    Pads with padding value in case multiple numbers of true elements are present.

    Parameters
    ----------
    mask : Bool[np.ndarray, " n_cases n_elements"]
        A boolean mask to be converted to an index array
    padding : int, optional
        The value to pad the index array with, by default -1

    Returns
    -------
    Int[np.ndarray, " n_cases max_n_true_elements"]
        An index array of the smallest shape possible with index values of the True elements in the mask
    """
    if mask.size == 0 or np.sum(mask) == 0:
        return np.zeros((mask.shape[0], 0), dtype=int)

    case_idx, element_idx = np.nonzero(mask)
    # Find out for each element which index of occurrence it is in this case
    occurrence = np.diag(np.cumsum(case_idx[None, :] == case_idx[:, None], axis=1)) - 1
    retval = np.full((mask.shape[0], occurrence.max() + 1), padding, dtype=int)
    # Then overwrite the padded array
    retval[case_idx, occurrence] = element_idx
    return retval
