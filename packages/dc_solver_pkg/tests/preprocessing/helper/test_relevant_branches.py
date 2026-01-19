# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from toop_engine_dc_solver.preprocess.helpers.relevant_branches import (
    get_relevant_branches,
)
from toop_engine_dc_solver.preprocess.network_data import NetworkData


def test_reduce_branch_dimension(
    network_data_filled: NetworkData,
) -> None:
    reduced_branches = get_relevant_branches(
        network_data_filled.from_nodes,
        network_data_filled.to_nodes,
        network_data_filled.relevant_node_mask,
        network_data_filled.monitored_branch_mask,
        network_data_filled.outaged_branch_mask,
        network_data_filled.multi_outage_branch_mask,
        np.zeros_like(network_data_filled.monitored_branch_mask, dtype=bool),
    )

    # Check for the same number of true values in all branch masks
    assert np.sum(network_data_filled.monitored_branch_mask) == np.sum(
        network_data_filled.monitored_branch_mask[reduced_branches]
    )
    assert np.sum(network_data_filled.outaged_branch_mask) == np.sum(
        network_data_filled.outaged_branch_mask[reduced_branches]
    )
    assert np.sum(network_data_filled.multi_outage_branch_mask) == np.sum(
        network_data_filled.multi_outage_branch_mask[:, reduced_branches]
    )

    # Check for the same number of branches that end in relevant nodes
    assert np.sum(network_data_filled.relevant_node_mask[network_data_filled.from_nodes]) == np.sum(
        network_data_filled.relevant_node_mask[network_data_filled.from_nodes[reduced_branches]]
    )
    assert np.sum(network_data_filled.relevant_node_mask[network_data_filled.to_nodes]) == np.sum(
        network_data_filled.relevant_node_mask[network_data_filled.to_nodes[reduced_branches]]
    )
