import jax
import jax.numpy as jnp
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.standard_emitters import EmitterState, ExtraScores, MixingEmitter
from qdax.custom_types import ExtraScores, RNGKey, Descriptor, Fitness
from functools import partial
from typing import Callable, Tuple

from toop_engine_dc_solver.jax.types import int_max
from toop_engine_topology_optimizer.dc.ga_helpers import MixingEmitterState
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from beartype.typing import Optional

class BruteForceEmitterState(MixingEmitterState):
    """
    extends EmitterState for brute force emitter
    """
    genotypes: Genotype


class BruteForceEmitter(MixingEmitter):
    # WIP emitter
    def __init__(
        self,
        batch_size: int,
        next_brute_force_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    ) -> None:
        super().__init__(None, None, None, batch_size)
        self._next_brute_force_fn = next_brute_force_fn


    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey
    ) -> Tuple[Genotype, ExtraScores, RNGKey]:
        genotypes = self._next_brute_force_fn(emitter_state.genotypes)
        return genotypes, {}, random_key


    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size


class TrackingBruteForceEmmiter(BruteForceEmitter):
    def init(
        self,
        random_key: PRNGKeyArray,
        init_genotypes: Optional[Genotype],  # noqa: ARG002
    ) -> tuple[EmitterState, PRNGKeyArray]:
        """Overwrite the Emitter.init function to seed an EmitterState."""
        from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
        return BruteForceEmitterState(
            total_branch_combis=jnp.array(0, dtype=int),
            total_inj_combis=jnp.array(0, dtype=int),
            total_num_splits=jnp.array(0, dtype=int),
            genotypes=Genotype(action_index=jnp.full((self.batch_size, 3), int_max(), dtype=int),
                                disconnections=jnp.full((self.batch_size, 5), int_max(), dtype=int),
                                nodal_injections_optimized=None),
        ), random_key

    def state_update(
        self,
        emitter_state: Optional[EmitterState],
        repertoire: Optional[Repertoire | DiscreteMapElitesRepertoire],  # noqa: ARG002
        genotypes: Optional[Genotype],  # noqa: ARG002
        fitnesses: Optional[Fitness],  # noqa: ARG002
        descriptors: Optional[Descriptor],  # noqa: ARG002
        extra_scores: ExtraScores,
    ) -> EmitterState:
        """Overwrite the state update to store information for the running means."""
        assert emitter_state is not None
        assert extra_scores is not None
        return BruteForceEmitterState(
            total_branch_combis=emitter_state.total_branch_combis
            + extra_scores.get("n_branch_combis", jnp.array(0, dtype=int)).astype(int),
            total_inj_combis=emitter_state.total_inj_combis
            + extra_scores.get("n_inj_combis", jnp.array(0, dtype=int)).astype(int),
            total_num_splits=emitter_state.total_num_splits
            + extra_scores.get("n_split_grids", jnp.array(0, dtype=int)).astype(int),
            genotypes=genotypes
        )