# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Create scoring functions for the genetic algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Optional
from jax_dataclasses import replace
from jaxtyping import Array, Bool, Float, Int
from qdax.core.emitters.standard_emitters import EmitterState
from toop_engine_dc_solver.jax.aggregate_results import aggregate_to_metric_batched, get_worst_k_contingencies
from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
from toop_engine_dc_solver.jax.nodal_inj_optim import make_start_options
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    DynamicInformation,
    MetricType,
    NodalInjOptimResults,
    NodalInjStartOptions,
    SolverConfig,
    int_max,
)
from toop_engine_topology_optimizer.dc.genetic_functions.evolution_functions import (
    Genotype,
    deduplicate_genotypes,
    fix_dtypes,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_map_elites import (
    DiscreteMapElitesRepertoire,
)
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics, Topology

# The list of metrics available and the function used to combine the results of two timesteps.
METRICS = {
    "max_flow_n_0": jnp.maximum,
    "median_flow_n_0": jnp.maximum,
    "overload_energy_n_0": jnp.add,
    "overload_energy_limited_n_0": jnp.add,
    "underload_energy_n_0": jnp.add,
    "transport_n_0": jnp.add,
    "exponential_overload_energy_n_0": jnp.add,
    "exponential_overload_energy_limited_n_0": jnp.add,
    "critical_branch_count_n_0": jnp.maximum,
    "critical_branch_count_limited_n_0": jnp.maximum,
    "max_flow_n_1": jnp.maximum,
    "median_flow_n_1": jnp.maximum,
    "overload_energy_n_1": jnp.add,
    "overload_energy_limited_n_1": jnp.add,
    "underload_energy_n_1": jnp.add,
    "transport_n_1": jnp.add,
    "exponential_overload_energy_n_1": jnp.add,
    "exponential_overload_energy_limited_n_1": jnp.add,
    "critical_branch_count_n_1": jnp.maximum,
    "critical_branch_count_limited_n_1": jnp.maximum,
    "n0_n1_delta": jnp.add,
    "cross_coupler_flow": jnp.add,
    "switching_distance": jnp.maximum,
    "split_subs": jnp.maximum,
    "n_2_penalty": jnp.add,
}


# The scoring function runs the loadflow and computes the overload energy
def compute_overloads(
    topologies: Genotype,
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
    observed_metrics: tuple[MetricType, ...],
    n_worst_contingencies: int = 10,
) -> tuple[dict[str, Float[Array, " batch_size"]], Optional[NodalInjOptimResults], Bool[Array, " batch_size"]]:
    """Compute the overloads for a single timestep by invoking the solver and aggregating the results.

    Parameters
    ----------
    topologies : Genotype
        The topologies to score, where the first max_num_splits entries are the substations, the
        second max_num_splits entries are the branch topos and the last max_num_splits entries are
        the injection topos
    dynamic_information : DynamicInformation
        The dynamic information of the grid
    solver_config : SolverConfig
        The static solver configuration
    observed_metrics : tuple[MetricType, ...]
        The metrics to observe
    n_worst_contingencies : int, optional
        The number of worst contingencies to return, by default 10

    Returns
    -------
    dict[str, Float[Array, " batch_size"]]
        A dictionary with the overload energy, transport, max flow and other metrics, from aggregate_to_metric_batched
    NodalInjOptimResults
        The results of the nodal injection optimization
    Bool[Array, " batch_size"]
        Whether the topologies were successfully solved
    """
    topo_comp, disconnections, nodal_inj_start = translate_topology(topologies)

    lf_res, success = compute_symmetric_batch(
        topology_batch=topo_comp,
        disconnection_batch=disconnections,
        injections=None,  # Use from action set
        nodal_inj_start_options=nodal_inj_start,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )

    aggregates = {}
    for metric_name in observed_metrics:
        metric = aggregate_to_metric_batched(
            lf_res_batch=lf_res,
            branch_limits=dynamic_information.branch_limits,
            reassignment_distance=dynamic_information.action_set.reassignment_distance,
            n_relevant_subs=dynamic_information.n_sub_relevant,
            metric=metric_name,
        )
        metric = jnp.where(success, metric, jnp.inf)
        aggregates[metric_name] = metric

    # Note: compute_overloads is called for each timestep separately, so the results are not batched.
    # As we don't have multi timestep optimisation support yet, we compute the worst k contingencies
    # sequentially one timestep at a time. This means that the timestep dimension will always be 1.
    #  TODO This is a temporary solution until we have multi timestep support.
    worst_k_res = jax.vmap(get_worst_k_contingencies, in_axes=(None, 0, None))(
        n_worst_contingencies, lf_res.n_1_matrix, dynamic_information.branch_limits.max_mw_flow
    )
    aggregates["top_k_overloads_n_1"] = worst_k_res.top_k_overloads[:, 0]  # Take the first timestep only
    aggregates["case_indices"] = worst_k_res.case_indices[:, 0, :]

    return aggregates, lf_res.nodal_injections_optimized, success


def scoring_function(
    topologies: Genotype,
    random_key: jax.random.PRNGKey,
    dynamic_informations: tuple[DynamicInformation, ...],
    solver_configs: tuple[SolverConfig, ...],
    target_metrics: tuple[tuple[MetricType, float], ...],
    observed_metrics: tuple[MetricType, ...],
    descriptor_metrics: tuple[MetricType, ...],
    n_worst_contingencies: int = 10,
) -> tuple[
    Float[Array, " batch_size"],
    Int[Array, " batch_size n_dims"],
    dict,
    dict,
    jax.random.PRNGKey,
    Genotype,
]:
    """Create scoring function for the genetic algorithm.

    Parameters
    ----------
    topologies : Genotype
        The topologies to score
    random_key : jax.random.PRNGKey
        The random key to use for the scoring (currently not used)
    dynamic_informations : tuple[DynamicInformation, ...]
        The dynamic information of the grid for every timestep
    solver_configs : tuple[SolverConfig, ...]
        The solver configuration for every timestep
    target_metrics : tuple[tuple[MetricType, float], ...]
        The list of metrics to optimize for with their weights
    observed_metrics : tuple[MetricType, ...]
        The observed metrics
    descriptor_metrics : tuple[MetricType, ...]
        The metrics to use as descriptors
    n_worst_contingencies : int, optional
        The number of worst contingencies to consider for calculating
        top_k_overloads_n_1

    Returns
    -------
    Float[Array, " batch_size"]
        The metrics of the topologies
    Int[Array, " batch_size n_dims"]
        The descriptors of the topologies
    dict
        The extra scores
    dict
        Emitter Information
    jax.random.PRNGKey
        The random key that was passed in, unused
    Genotype
        The genotypes that were passed in, but updated to account for in-the-loop optimizations such as the nodal
        injection optimization.
    """
    n_topologies = len(topologies.action_index)

    metrics, nodal_injections_optimized, success = compute_overloads(
        topologies=topologies,
        dynamic_information=dynamic_informations[0],
        solver_config=solver_configs[0],
        observed_metrics=observed_metrics,
        n_worst_contingencies=n_worst_contingencies,
    )
    # Sequentially compute each subsequent timestep
    for dynamic_information, solver_config in zip(dynamic_informations[1:], solver_configs[1:], strict=True):
        metrics_local, _nodal_injections_optimized_local, success_local = compute_overloads(
            topologies=topologies,
            dynamic_information=dynamic_information,
            solver_config=solver_config,
            observed_metrics=observed_metrics,
        )
        success = success & success_local

        # TODO figure out how to stack nodal_inj optim results for multiple timesteps
        for key in observed_metrics:
            combine_fn = METRICS[key]
            metrics[key] = combine_fn(metrics[key], metrics_local[key])

    fitness = sum(-metrics[metric_name] * weight for metric_name, weight in target_metrics)

    emitter_info = {
        "n_branch_combis": jnp.array(n_topologies, dtype=int),
        "n_inj_combis": jnp.array(n_topologies, dtype=int),
        "n_split_grids": jnp.sum(~success),
    }

    descriptors = jnp.stack([metrics[key].astype(int) for key in descriptor_metrics], axis=1)

    topologies = replace(topologies, nodal_injections_optimized=nodal_injections_optimized)

    return (
        fitness,
        descriptors,
        metrics,
        emitter_info,
        random_key,
        topologies,
    )


def translate_topology(
    topology: Genotype,
) -> tuple[
    ActionIndexComputations,
    Int[Array, " batch_size max_num_disconnections"],
    NodalInjStartOptions | None,
]:
    """Translate the topology into the format used by the solver.

    Parameters
    ----------
    topology : Genotype
        The topology in genotype form

    Returns
    -------
    ActionIndexComputations
        The topology computations
    Int[Array, " batch_size max_num_disconnections"]
        The branch disconnections to apply
    NodalInjStartOptions | None
        The nodal injection optimization start options containing pst taps, or None if no PST optimization
    """
    batch_size, _max_num_splits = topology.action_index.shape

    topology = fix_dtypes(topology)

    # Branch actions can be read straight out of the branch actions array
    topo_comp = ActionIndexComputations(
        action=topology.action_index,
        pad_mask=jnp.ones((batch_size,), dtype=bool),
    )

    nodal_inj_start = make_start_options(topology.nodal_injections_optimized)

    return topo_comp, topology.disconnections, nodal_inj_start


def filter_repo(
    repertoire: DiscreteMapElitesRepertoire,
    initial_fitness: float,
) -> DiscreteMapElitesRepertoire:
    """Reduce the repertoire to only valid and deduplicated topologies.

    This will not return any topologies that are worse than the initial fitness and deduplicate the topologies.
    The function is currently not jax jit compatible.

    Parameters
    ----------
    repertoire : DiscreteMapElitesRepertoire
        The repertoire to reduce
    initial_fitness : float
        The initial fitness of the grid. This is used to filter out topologies that are worse than this fitness.

    Returns
    -------
    DiscreteMapElitesRepertoire
        The reduced repertoire with only valid and deduplicated topologies.
    """
    distributed = len(repertoire.fitnesses.shape) > 1
    if distributed:
        repertoire = repertoire[0]

    assert len(repertoire.fitnesses.shape) == 1, "Wrong shape on repertoire"
    # Deduplicate the repertoire and remove invalid entries
    valid_mask = jnp.isfinite(repertoire.fitnesses) & (repertoire.fitnesses > initial_fitness)
    repertoire = repertoire[valid_mask]

    _, indices = deduplicate_genotypes(repertoire.genotypes)
    repertoire = repertoire[indices]

    return repertoire


def convert_to_topologies(
    repertoire: DiscreteMapElitesRepertoire,
    contingency_ids: list[str],
    grid_model_low_tap: Int[Array, " n_controllable_psts"] | None = None,
) -> list[Topology]:
    """Take a repertoire and convert it to a list of kafka-sendable topologies.

    Parameters
    ----------
    repertoire : DiscreteMapElitesRepertoire
        The repertoire to convert. You might want to filter it using filter_repo first.
    contingency_ids : list[str]
        The contingency IDs for each topology
    grid_model_low_tap : Int[Array, " n_controllable_psts"] | None
        The lowest tap value in the grid model, used to convert the relative tap values in the genotype to absolute tap
        values that can be sent to the kafka topics. This will only be read if nodal_injection results are present
        in the genotype.

    Returns
    -------
    list[Topology]
        The list of topologies in the format used by the kafka topics.
    """
    topologies = []

    for i in range(len(repertoire.fitnesses)):
        iter_repertoire = repertoire[i]

        action_indices = [int(act) for act in iter_repertoire.genotypes.action_index if act != int_max()]

        disconnections = [int(disc) for disc in iter_repertoire.genotypes.disconnections if disc != int_max()]

        nodal_inj = iter_repertoire.genotypes.nodal_injections_optimized
        pst_setpoints = None
        if nodal_inj is not None:
            assert grid_model_low_tap is not None, (
                "grid_model_low_tap must be provided if nodal_injections_optimized is present"
            )
            assert len(nodal_inj.pst_taps.shape) == 2
            assert nodal_inj.pst_taps.shape[0] == 1, "Only one timestep is supported, but found shape " + str(
                nodal_inj.pst_taps.shape
            )
            tap_array = nodal_inj.pst_taps[0].astype(int) + grid_model_low_tap
            pst_setpoints = tap_array.tolist()

        case_indices = iter_repertoire.extra_scores.pop("case_indices", [])
        case_ids = np.array(contingency_ids)[case_indices].tolist()
        metrics = Metrics(
            fitness=float(iter_repertoire.fitnesses),
            extra_scores={key: value.item() for key, value in iter_repertoire.extra_scores.items()},
            worst_k_contingency_cases=case_ids,
        )

        topologies.append(
            Topology(
                actions=action_indices,
                disconnections=disconnections,
                pst_setpoints=pst_setpoints,
                metrics=metrics,
            )
        )
    return topologies


def summarize_repo(
    repertoire: DiscreteMapElitesRepertoire,
    initial_fitness: float,
    contingency_ids: list[str],
    grid_model_low_tap: Int[Array, " n_controllable_psts"] | None = None,
) -> list[Topology]:
    """Summarize the repertoire into a list of topologies.

    Parameters
    ----------
    repertoire : DiscreteMapElitesRepertoire
        The repertoire to summarize
    initial_fitness : float
        The initial fitness of the grid
    contingency_ids : list[str]
        The contingency IDs for each topology. Here we assume that this list is common for all the topologies
        in the repertoire.
        TODO: Fix me to have per topology contingency ids if needed
    grid_model_low_tap : Int[Array, " n_controllable_psts"] | None
        The lowest tap value in the grid model, from nodal_injection_information.grid_model_low_tap.

    Returns
    -------
    list[Topology]
        The summarized topologies
    """
    with jax.default_device(jax.devices("cpu")[0]):
        best_repo = filter_repo(
            repertoire=repertoire,
            initial_fitness=initial_fitness,
        )

        topologies = convert_to_topologies(best_repo, contingency_ids, grid_model_low_tap=grid_model_low_tap)

    return topologies


def summarize(
    repertoire: DiscreteMapElitesRepertoire,
    emitter_state: EmitterState,
    initial_fitness: float,
    initial_metrics: dict,
    contingency_ids: list[str],
    grid_model_low_tap: Int[Array, " n_controllable_psts"] | None = None,
) -> dict:
    """Summarize the repertoire and emitter state into a serializable dictionary.

    Parameters
    ----------
    repertoire : DiscreteMapElitesRepertoire
        The repertoire to summarize
    emitter_state : EmitterState
        The emitter state to summarize
    initial_fitness : float
        The initial fitness of the grid
    initial_metrics : dict
        The initial metrics of the grid
    contingency_ids : list[str]
        A list of contingency ids. Here we assume that the list of contingency ids is common for all the topologies
    grid_model_low_tap : Int[Array, " n_controllable_psts"] | None
        The lowest tap value in the grid model, from nodal_injection_information.grid_model_low_tap.

    Returns
    -------
    dict
        The summarized dictionary, json serializable
    """
    topologies = summarize_repo(
        repertoire=repertoire,
        initial_fitness=initial_fitness,
        contingency_ids=contingency_ids,
        grid_model_low_tap=grid_model_low_tap,
    )
    max_fitness = max(t.metrics.fitness for t in topologies) if len(topologies) > 0 else initial_fitness

    # Store the topologies
    best_topos = [t.model_dump() for t in topologies]
    retval = {k: v.item() for k, v in emitter_state.items()}
    retval.update(
        {
            "max_fitness": max_fitness,
            "best_topos": best_topos,
            "initial_fitness": initial_fitness,
            "initial_metrics": initial_metrics,
        }
    )
    return retval
