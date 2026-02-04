# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""JAX - GPU/TPU compatible- implementation of the DC load flow.

Paper references:
Accelerated DC loadflow solver for topology optimization
https://arxiv.org/pdf/2501.17529

Bus Split Distribution Factors
DOI:10.36227/techrxiv.22298950.v1

Unified algebraic deviation of distribution factors in linear power flow
https://doi.org/10.48550/arXiv.2412.16164
"""

from .aggregate_results import (
    aggregate_to_metric,
    aggregate_to_metric_batched,
    default_metric,
)
from .batching import (
    get_buffer_utilization,
    greedy_buffer_size_selection,
    greedy_n_subs_selection,
    upper_bound_buffer_size_injections,
)
from .compute_batch import (
    compute_batch,
    compute_symmetric_batch,
)
from .cross_coupler_flow import (
    compute_cross_coupler_flow_single,
    compute_cross_coupler_flows,
    get_unsplit_flows,
)
from .injections import (
    convert_action_index_to_numpy,
    convert_inj_candidates,
    convert_inj_topo_vect,
    default_injection,
    random_injection,
)
from .inputs import (
    convert_from_stat_bool,
    convert_tot_stat,
    deserialize_static_information,
    load_static_information,
    save_static_information,
    serialize_static_information,
    validate_static_information,
)
from .inspector import inspect_topology, is_valid_batch
from .nminus2_outage import n_2_analysis
from .result_storage import sparsify_results
from .topology_computations import (
    apply_limit_n_subs,
    convert_action_set_index_to_topo,
    convert_branch_topo_vect,
    convert_topo_sel_sorted,
    convert_topo_to_action_set_index,
    convert_topo_to_action_set_index_jittable,
    default_topology,
    random_topology,
)
from .topology_looper import (
    run_solver,
    run_solver_inj_bruteforce,
    run_solver_symmetric,
)
from .types import (
    AggregateMetricProtocol,
    AggregateOutputProtocol,
    InjectionComputations,
    SparseNMinus0,
    SparseNMinus1,
    SparseSolverOutput,
    StaticInformation,
    TopoVectBranchComputations,
)

__all__ = [
    "AggregateMetricProtocol",
    "AggregateOutputProtocol",
    "InjectionComputations",
    "SparseNMinus0",
    "SparseNMinus1",
    "SparseSolverOutput",
    "StaticInformation",
    "TopoVectBranchComputations",
    "aggregate_to_metric",
    "aggregate_to_metric_batched",
    "apply_limit_n_subs",
    "compute_batch",
    "compute_cross_coupler_flow_single",
    "compute_cross_coupler_flows",
    "compute_symmetric_batch",
    "convert_action_index_to_numpy",
    "convert_action_set_index_to_topo",
    "convert_branch_topo_vect",
    "convert_branch_topo_vect",
    "convert_from_stat_bool",
    "convert_inj_candidates",
    "convert_inj_topo_vect",
    "convert_topo_sel_sorted",
    "convert_topo_sel_sorted",
    "convert_topo_to_action_set_index",
    "convert_topo_to_action_set_index_jittable",
    "convert_tot_stat",
    "default_injection",
    "default_metric",
    "default_topology",
    "default_topology",
    "deserialize_static_information",
    "get_buffer_utilization",
    "get_unsplit_flows",
    "greedy_buffer_size_selection",
    "greedy_n_subs_selection",
    "inspect_topology",
    "is_valid_batch",
    "load_static_information",
    "n_2_analysis",
    "random_injection",
    "random_topology",
    "run_solver",
    "run_solver_inj_bruteforce",
    "run_solver_symmetric",
    "save_static_information",
    "serialize_static_information",
    "sparsify_results",
    "upper_bound_buffer_size_injections",
    "validate_static_information",
]
