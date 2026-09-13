from beartype.typing import Callable, Tuple
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
from pydantic import PositiveInt
from qdax.core.emitters.emitter import Emitter
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Metrics
from scipy.special import comb
from toop_engine_dc_solver.jax.types import int_max
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.repertoire.discrete_map_elites import DiscreteMapElites, EmitterScores
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire


def next_subsample(current: jnp.ndarray, n: int, p: int) -> jnp.ndarray:
    """
    Generates a tensor of shape current.shape where each row is the next subset of the previous row, starting from initial. If a subset has no next subset, fills the row with -1.

    Args:

        current: Array of shape (batch_size, p) containing the initial subset.
        n: Total number of elements (from 0 to n-1).
        p: Size of the subsets.

    Returns:
        Tensor of shape (batch_size, p) containing the batch_size next subsets generated from inital combination current[-1,:].

    """

    batch_size = current.shape[0]

    def next_subsample_single(sub):
        """
        Finds the next combination from a given one.

        """

        mask = sub != (n - p + jnp.arange(p, dtype=jnp.int64))
        i = jnp.max(jnp.where(mask, jnp.arange(p, dtype=jnp.int64), -1))
        next_sub = jnp.where(mask, sub, -1)

        def update_sub(sub, i):
            sub = sub.at[i].add(1)
            def body_fun(j, sub_val):
                sub_val = sub_val.at[i + 1 + j].set(sub_val[i] + j + 1)
                return sub_val
            sub = lax.fori_loop(0, p - i - 1, body_fun, sub)
            return sub

        next_sub = update_sub(next_sub, i)

        next_sub = lax.cond(
            i == -1,
            lambda: jnp.zeros(p, dtype=jnp.int64),
            lambda: next_sub,
        )
        return next_sub

    initial = next_subsample_single(current[-1])
    current = current.at[0].set(initial)

    def body_fun(i, output):
        prev_sub = output[i - 1]
        new_sub = next_subsample_single(prev_sub)
        output = output.at[i].set(new_sub)
        return output

    output = lax.fori_loop(1, batch_size, body_fun, current)
    return output


def next_brute_force(current_genotypes, n_actions, n_disconnections, n_rel_subs, n_disconnectable_branches) -> tuple[Genotype, int]:

    def transform(disconnections):
        # transorm ToOp indexing to brute force generator indexing
        combi = jnp.flip(disconnections)
        combi = jnp.where(combi != int_max(), combi + 1, combi)
        combi = jnp.where(combi == int_max(), 0, combi)
        return combi

    def inverse_transform(combi):
        # Inverse transform of the latter
        disconnections = jnp.where(combi == 0, int_max(), combi)
        disconnections = jnp.where(disconnections != int_max(), disconnections - 1, disconnections)
        disconnections = jnp.flip(disconnections)
        return disconnections

    current_action_index = current_genotypes.action_index
    current_disconnections = current_genotypes.disconnections

    # Find next combinations for actions and disconnections

    next_action_index = transform(current_action_index)
    next_action_index = next_subsample(next_action_index, n_rel_subs + 1, n_actions)
    next_action_index = inverse_transform(next_action_index)

    next_disconnections = transform(current_disconnections)
    next_disconnections = next_subsample(next_disconnections, n_disconnectable_branches + 1, n_disconnections)
    next_disconnections = inverse_transform(next_disconnections)

    # If all actions have been tested, take next disconnections
    incremente = jnp.all(next_action_index == int_max())

    next_disconnections = jnp.where(incremente, int_max(), next_disconnections)
    next_action_index = jnp.where(incremente, next_action_index, current_action_index)

    return Genotype(action_index=next_action_index,
                    disconnections=next_disconnections,
                    nodal_injections_optimized=None)


def number_of_possible_combinations(n, p):
    """
    Returns the number of possible combinations 
    corresponding to the number of sub sample of integers from 0 to n
    of maximal size p
    """
    res = sum([comb(n, k, exact=True) for k in range(p+1)])
    return res


class BruteForceAlgo(DiscreteMapElites):

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, PRNGKeyArray, PyTree],
            Tuple[Fitness, Descriptor, ExtraScores, EmitterScores, PRNGKeyArray, Genotype],
        ],
        emitter: Emitter,
        metrics_function: Callable[[DiscreteMapElitesRepertoire], Metrics],
        n_cells_per_dim: tuple[int, ...],
        cell_depth: PositiveInt = 1,
        distributed: bool = False,
        max_num_splits: PositiveInt = 1,
        max_num_disconnections: PositiveInt = 1,
        number_of_combinations_to_evaluate: PositiveInt = 1
    ) -> None:
        super().__init__(scoring_function, emitter, metrics_function, n_cells_per_dim, cell_depth, distributed)
        self._max_num_splits = max_num_splits
        self._max_num_disconnections = max_num_disconnections
        self._number_of_combinations_to_evaluate = number_of_combinations_to_evaluate