# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Parent-selection strategies for MAP-Elites repertoires."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from beartype.typing import Literal, Tuple
from flax.struct import PyTreeNode
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

ParentSelectionMode = Literal[
    "uniform",
    "uniform_cell",
    "ucb",
    "ucb_batched",
    "ucb_snapshot",
    "exploitation",
    "exploration",
    "greedy",
]


class ParentSelectorState(PyTreeNode):
    """Runtime state threaded through the emitter for parent selection."""


class UCBParentSelectorState(ParentSelectorState):
    """Runtime state for feedback-driven cell-based parent selection."""

    selection_counts: Int[Array, " n_cells"]
    success_counts: Int[Array, " n_cells"]
    total_selections: Int[Array, " "]


class ParentSelector(ABC):
    """Policy object responsible for selecting parent repertoire indices."""

    @property
    @abstractmethod
    def mode(self) -> ParentSelectionMode:
        """Return the configured parent-selection mode."""

    def init_state(self) -> ParentSelectorState:
        """Create the initial runtime state for the selection policy."""
        return ParentSelectorState()

    def state_from_telemetry(
        self,
        parent_selector_state: ParentSelectorState,
        selection_counts: Int[Array, " n_cells"],  # noqa: ARG002
        success_counts: Int[Array, " n_cells"],  # noqa: ARG002
    ) -> ParentSelectorState:
        """Build selector state from the emitter's cumulative parent telemetry."""
        return parent_selector_state

    @abstractmethod
    def select(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,
    ) -> Tuple[Int[Array, " num_samples"], ParentSelectorState, PRNGKeyArray]:
        """Select occupied repertoire indices and return the updated runtime state."""

    def select_for_emission(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,
    ) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
        """Select all parents required by one emitter invocation.

        The default implementation preserves selector behavior while discarding the
        transient state used only during this emission.
        """
        selected_indices, _updated_state, random_key = self.select(
            occupied_mask=occupied_mask,
            parent_selector_state=parent_selector_state,
            random_key=random_key,
            num_samples=num_samples,
            fitnesses=fitnesses,
        )
        return selected_indices, random_key


@dataclass(frozen=True)
class UniformParentSelector(ParentSelector):
    """Uniform parent-selection policy over occupied repertoire entries."""

    @property
    def mode(self) -> ParentSelectionMode:
        """Return the configured parent-selection mode."""
        return "uniform"

    def select(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,  # noqa: ARG002
    ) -> Tuple[Int[Array, " num_samples"], ParentSelectorState, PRNGKeyArray]:
        """Select occupied repertoire indices uniformly."""
        selected_indices, random_key = select_uniform_parent_indices(
            occupied_mask=occupied_mask,
            random_key=random_key,
            num_samples=num_samples,
        )
        return selected_indices, parent_selector_state, random_key


@dataclass(frozen=True)
class UniformCellParentSelector(ParentSelector):
    """Uniform parent-selection policy over occupied MAP-Elites cells."""

    cell_depth: int

    @property
    def mode(self) -> ParentSelectionMode:
        """Return the configured parent-selection mode."""
        return "uniform_cell"

    def select(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,  # noqa: ARG002
    ) -> Tuple[Int[Array, " num_samples"], ParentSelectorState, PRNGKeyArray]:
        """Select a cell uniformly, then an occupied candidate within that cell."""
        selected_indices, random_key = select_uniform_cell_parent_indices(
            occupied_mask=occupied_mask,
            random_key=random_key,
            num_samples=num_samples,
            cell_depth=self.cell_depth,
        )
        return selected_indices, parent_selector_state, random_key


@dataclass(frozen=True)
class UCBParentSelector(ParentSelector):
    """Cell-based UCB parent-selection policy over occupied repertoire cells."""

    ucb_exploration_constant: float
    cell_depth: int

    def init_state(self) -> ParentSelectorState:
        """Create an empty feedback state sized on first repertoire update."""
        return UCBParentSelectorState(
            selection_counts=jnp.zeros((0,), dtype=int),
            success_counts=jnp.zeros((0,), dtype=int),
            total_selections=jnp.array(0, dtype=int),
        )

    @property
    def mode(self) -> ParentSelectionMode:
        """Return the configured parent-selection mode."""
        return "ucb"

    def state_from_telemetry(
        self,
        parent_selector_state: ParentSelectorState,  # noqa: ARG002
        selection_counts: Int[Array, " n_cells"],
        success_counts: Int[Array, " n_cells"],
    ) -> ParentSelectorState:
        """Reuse the emitter's feedback counters as the next UCB state."""
        return UCBParentSelectorState(
            selection_counts=selection_counts,
            success_counts=success_counts,
            total_selections=jnp.sum(selection_counts, dtype=int),
        )

    def select(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,  # noqa: ARG002
    ) -> Tuple[Int[Array, " num_samples"], ParentSelectorState, PRNGKeyArray]:
        """Select occupied repertoire indices and advance transient feedback counts."""
        num_cells = occupied_mask.shape[0] // self.cell_depth
        state = ensure_ucb_parent_selector_state(parent_selector_state, num_cells)
        selected_indices, random_key = select_cell_parent_indices(
            occupied_mask=occupied_mask,
            selection_counts=state.selection_counts,
            success_counts=state.success_counts,
            total_selections=state.total_selections,
            parent_selection_mode=self.mode,
            ucb_exploration_constant=self.ucb_exploration_constant,
            random_key=random_key,
            num_samples=num_samples,
            cell_depth=self.cell_depth,
        )
        selected_cells = repertoire_indices_to_cell_indices(selected_indices, num_cells)
        updated_state = UCBParentSelectorState(
            selection_counts=state.selection_counts.at[selected_cells].add(1),
            success_counts=state.success_counts,
            total_selections=state.total_selections + jnp.array(num_samples, dtype=int),
        )
        return selected_indices, updated_state, random_key

    def select_for_emission(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,  # noqa: ARG002
    ) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
        """Select one full parent emission without materializing discarded state."""
        num_cells = occupied_mask.shape[0] // self.cell_depth
        state = ensure_ucb_parent_selector_state(parent_selector_state, num_cells)
        return select_cell_parent_indices(
            occupied_mask=occupied_mask,
            selection_counts=state.selection_counts,
            success_counts=state.success_counts,
            total_selections=state.total_selections,
            parent_selection_mode=self.mode,
            ucb_exploration_constant=self.ucb_exploration_constant,
            random_key=random_key,
            num_samples=num_samples,
            cell_depth=self.cell_depth,
        )


@dataclass(frozen=True)
class BatchedUCBParentSelector(UCBParentSelector):
    """Cell-based UCB selector that updates virtual counts in vectorized blocks."""

    selection_block_size: int

    @property
    def mode(self) -> ParentSelectionMode:
        """Return the configured parent-selection mode."""
        return "ucb_batched"

    def select(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,  # noqa: ARG002
    ) -> Tuple[Int[Array, " num_samples"], ParentSelectorState, PRNGKeyArray]:
        """Select occupied repertoire indices using batched virtual UCB pulls."""
        num_cells = occupied_mask.shape[0] // self.cell_depth
        state = ensure_ucb_parent_selector_state(parent_selector_state, num_cells)
        selected_indices, random_key = select_batched_ucb_parent_indices(
            occupied_mask=occupied_mask,
            selection_counts=state.selection_counts,
            success_counts=state.success_counts,
            total_selections=state.total_selections,
            ucb_exploration_constant=self.ucb_exploration_constant,
            random_key=random_key,
            num_samples=num_samples,
            cell_depth=self.cell_depth,
            selection_block_size=self.selection_block_size,
        )
        selected_cells = repertoire_indices_to_cell_indices(selected_indices, num_cells)
        updated_state = UCBParentSelectorState(
            selection_counts=state.selection_counts.at[selected_cells].add(1),
            success_counts=state.success_counts,
            total_selections=state.total_selections + jnp.array(num_samples, dtype=int),
        )
        return selected_indices, updated_state, random_key

    def select_for_emission(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,  # noqa: ARG002
    ) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
        """Select one full virtual UCB emission without materializing discarded state."""
        num_cells = occupied_mask.shape[0] // self.cell_depth
        state = ensure_ucb_parent_selector_state(parent_selector_state, num_cells)
        return select_batched_ucb_parent_indices(
            occupied_mask=occupied_mask,
            selection_counts=state.selection_counts,
            success_counts=state.success_counts,
            total_selections=state.total_selections,
            ucb_exploration_constant=self.ucb_exploration_constant,
            random_key=random_key,
            num_samples=num_samples,
            cell_depth=self.cell_depth,
            selection_block_size=self.selection_block_size,
        )


@dataclass(frozen=True)
class SnapshotUCBParentSelector(UCBParentSelector):
    """GPU-friendly UCB policy that samples from one score snapshot per emission."""

    temperature: float

    @property
    def mode(self) -> ParentSelectionMode:
        """Return the configured parent-selection mode."""
        return "ucb_snapshot"

    def select(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,  # noqa: ARG002
    ) -> Tuple[Int[Array, " num_samples"], ParentSelectorState, PRNGKeyArray]:
        """Select parents from a fixed UCB score distribution."""
        num_cells = occupied_mask.shape[0] // self.cell_depth
        state = ensure_ucb_parent_selector_state(parent_selector_state, num_cells)
        selected_indices, random_key = select_snapshot_ucb_parent_indices(
            occupied_mask=occupied_mask,
            selection_counts=state.selection_counts,
            success_counts=state.success_counts,
            total_selections=state.total_selections,
            ucb_exploration_constant=self.ucb_exploration_constant,
            temperature=self.temperature,
            random_key=random_key,
            num_samples=num_samples,
            cell_depth=self.cell_depth,
        )
        selected_cells = repertoire_indices_to_cell_indices(selected_indices, num_cells)
        updated_state = UCBParentSelectorState(
            selection_counts=state.selection_counts.at[selected_cells].add(1),
            success_counts=state.success_counts,
            total_selections=state.total_selections + jnp.array(num_samples, dtype=int),
        )
        return selected_indices, updated_state, random_key

    def select_for_emission(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,  # noqa: ARG002
    ) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
        """Select one snapshot-scored parent emission without state updates."""
        num_cells = occupied_mask.shape[0] // self.cell_depth
        state = ensure_ucb_parent_selector_state(parent_selector_state, num_cells)
        return select_snapshot_ucb_parent_indices(
            occupied_mask=occupied_mask,
            selection_counts=state.selection_counts,
            success_counts=state.success_counts,
            total_selections=state.total_selections,
            ucb_exploration_constant=self.ucb_exploration_constant,
            temperature=self.temperature,
            random_key=random_key,
            num_samples=num_samples,
            cell_depth=self.cell_depth,
        )


class ExploitationParentSelector(UCBParentSelector):
    """Cell-based parent-selection policy that maximizes empirical survival rate."""

    def __init__(self, cell_depth: int) -> None:
        super().__init__(ucb_exploration_constant=0.0, cell_depth=cell_depth)

    @property
    def mode(self) -> ParentSelectionMode:
        """Return the configured parent-selection mode."""
        return "exploitation"


class ExplorationParentSelector(UCBParentSelector):
    """Cell-based parent-selection policy that prioritizes unvisited then least-selected cells."""

    def __init__(self, cell_depth: int) -> None:
        super().__init__(ucb_exploration_constant=1.0, cell_depth=cell_depth)

    @property
    def mode(self) -> ParentSelectionMode:
        """Return the configured parent-selection mode."""
        return "exploration"


class GreedyParentSelector(ParentSelector):
    """Parent-selection policy that samples uniformly among highest-fitness candidates."""

    @property
    def mode(self) -> ParentSelectionMode:
        """Return the configured parent-selection mode."""
        return "greedy"

    def select(
        self,
        occupied_mask: Bool[Array, " repertoire_size"],
        parent_selector_state: ParentSelectorState,
        random_key: PRNGKeyArray,
        num_samples: int,
        fitnesses: Float[Array, " repertoire_size"] | None = None,
    ) -> Tuple[Int[Array, " num_samples"], ParentSelectorState, PRNGKeyArray]:
        """Select candidates with the highest occupied repertoire fitness."""
        if fitnesses is None:
            raise ValueError("Greedy parent selection requires repertoire fitnesses.")
        selected_indices, random_key = select_greedy_parent_indices(
            occupied_mask=occupied_mask,
            fitnesses=fitnesses,
            random_key=random_key,
            num_samples=num_samples,
        )
        return selected_indices, parent_selector_state, random_key


def build_parent_selector(  # noqa: C901
    parent_selection_mode: ParentSelectionMode,
    ucb_exploration_constant: float,
    cell_depth: int,
    ucb_selection_block_size: int = 32,
    ucb_snapshot_temperature: float = 0.1,
) -> ParentSelector:
    """Build the configured parent-selection policy object.

    Parameters
    ----------
    parent_selection_mode : ParentSelectionMode
        Parent-selection strategy to instantiate.
    ucb_exploration_constant : float
        Exploration constant used by the UCB selector.
    cell_depth : int
        Number of repertoire slots stored per MAP-Elites cell.
    ucb_selection_block_size : int
        Number of virtual UCB pulls selected together by ``"ucb_batched"``.
    ucb_snapshot_temperature : float
        Softmax temperature used by ``"ucb_snapshot"`` after all unvisited cells
        have received one parent selection.

    Returns
    -------
    ParentSelector
        The configured parent-selector implementation.

    Raises
    ------
    ValueError
        If the requested parent-selection mode is not supported.
    """
    if parent_selection_mode == "uniform":
        parent_selector: ParentSelector = UniformParentSelector()
    elif parent_selection_mode == "uniform_cell":
        parent_selector = UniformCellParentSelector(cell_depth=cell_depth)
    elif parent_selection_mode == "ucb":
        parent_selector = UCBParentSelector(ucb_exploration_constant=ucb_exploration_constant, cell_depth=cell_depth)
    elif parent_selection_mode == "ucb_batched":
        if ucb_selection_block_size < 1:
            raise ValueError("ucb_selection_block_size must be at least 1.")
        parent_selector = BatchedUCBParentSelector(
            ucb_exploration_constant=ucb_exploration_constant,
            cell_depth=cell_depth,
            selection_block_size=ucb_selection_block_size,
        )
    elif parent_selection_mode == "ucb_snapshot":
        if ucb_snapshot_temperature <= 0:
            raise ValueError("ucb_snapshot_temperature must be positive.")
        parent_selector = SnapshotUCBParentSelector(
            ucb_exploration_constant=ucb_exploration_constant,
            cell_depth=cell_depth,
            temperature=ucb_snapshot_temperature,
        )
    elif parent_selection_mode == "exploitation":
        parent_selector = ExploitationParentSelector(cell_depth=cell_depth)
    elif parent_selection_mode == "exploration":
        parent_selector = ExplorationParentSelector(cell_depth=cell_depth)
    elif parent_selection_mode == "greedy":
        parent_selector = GreedyParentSelector()
    else:
        raise ValueError(f"Unsupported parent selection mode: {parent_selection_mode}")
    return parent_selector


def ensure_ucb_parent_selector_state(
    parent_selector_state: ParentSelectorState,
    num_cells: int,
) -> UCBParentSelectorState:
    """Resize or initialize feedback state to match the current repertoire cell count.

    Parameters
    ----------
    parent_selector_state : ParentSelectorState
        Current selector state threaded through the emitter.
    num_cells : int
        Number of MAP-Elites cells represented by the current repertoire.

    Returns
    -------
    UCBParentSelectorState
        A feedback state whose arrays match the requested number of cells.
    """
    if (
        isinstance(parent_selector_state, UCBParentSelectorState)
        and parent_selector_state.selection_counts.shape[0] == num_cells
    ):
        return parent_selector_state

    return UCBParentSelectorState(
        selection_counts=jnp.zeros((num_cells,), dtype=int),
        success_counts=jnp.zeros((num_cells,), dtype=int),
        total_selections=jnp.array(0, dtype=int),
    )


def repertoire_indices_to_cell_indices(
    repertoire_indices: Int[Array, " n_indices"],
    num_cells: int,
) -> Int[Array, " n_indices"]:
    """Project flat repertoire indices to their owning MAP-Elites cell indices.

    Parameters
    ----------
    repertoire_indices : Int[Array, " n_indices"]
        Flat indices into the depth-expanded repertoire storage.
    num_cells : int
        Number of logical MAP-Elites cells in the repertoire.

    Returns
    -------
    Int[Array, " n_indices"]
        Cell index for each flat repertoire index.
    """
    return jnp.mod(repertoire_indices, num_cells)


def get_occupied_cell_mask(
    occupied_mask: Bool[Array, " repertoire_size"],
    cell_depth: int,
) -> Bool[Array, " n_cells"]:
    """Collapse flat repertoire occupancy to one boolean per cell.

    Parameters
    ----------
    occupied_mask : Bool[Array, " repertoire_size"]
        Boolean mask marking occupied slots in the flat repertoire storage.
    cell_depth : int
        Number of slots stored per MAP-Elites cell.

    Returns
    -------
    Bool[Array, " n_cells"]
        Boolean mask indicating which logical MAP-Elites cells are occupied.
    """
    if cell_depth == 1:
        return occupied_mask
    num_cells = occupied_mask.shape[0] // cell_depth
    return occupied_mask.reshape((cell_depth, num_cells)).any(axis=0)


def compute_cell_scores(
    occupied_cell_mask: Bool[Array, " n_cells"],
    selection_counts: Int[Array, " n_cells"],
    success_counts: Int[Array, " n_cells"],
    total_selections: Int[Array, " "],
    parent_selection_mode: ParentSelectionMode,
    ucb_exploration_constant: Float[Array, " "],
) -> Float[Array, " n_cells"]:
    r"""Compute scores for occupied MAP-Elites cells.

    Parameters
    ----------
    occupied_cell_mask : Bool[Array, " n_cells"]
        Boolean mask indicating which MAP-Elites cells contain at least one genotype.
    selection_counts : Int[Array, " n_cells"]
        Number of times each cell has been selected as a parent so far.
    success_counts : Int[Array, " n_cells"]
        Number of surviving offspring attributed to each parent cell.
    total_selections : Int[Array, " "]
        Total number of parent selections accumulated across all cells.
    parent_selection_mode : ParentSelectionMode
        Cell-based mode used to calculate the score: ``"ucb"``, ``"exploitation"``,
        or ``"exploration"``.
    ucb_exploration_constant : Float[Array, " "]
        Multiplicative factor for the exploration term in the UCB score.

    Returns
    -------
    Float[Array, " n_cells"]
        Score for each cell, masked to ``-jnp.inf`` for unoccupied cells.

    Notes
    -----
    Let $s_i$ and $n_i$ be a cell's success and selection counts, and let $N$ be
    the total number of selections. ``"exploitation"`` uses $s_i / n_i$ with
    unvisited cells scored as zero. ``"exploration"`` uses
    $\sqrt{\log(N) / n_i}$, and ``"ucb"`` adds that term, scaled by
    ``ucb_exploration_constant``, to the exploitation score. Exploration and UCB
    score unvisited occupied cells as infinity.
    """
    safe_selection_counts = jnp.maximum(selection_counts, 1)
    exploitation = success_counts.astype(float) / safe_selection_counts.astype(float)
    exploration = ucb_exploration_constant * jnp.sqrt(
        jnp.log(jnp.maximum(total_selections.astype(float), 1.0)) / safe_selection_counts.astype(float)
    )
    if parent_selection_mode == "exploitation":
        scores = exploitation
    else:
        visited_scores = exploration if parent_selection_mode == "exploration" else exploitation + exploration
        scores = jnp.where(selection_counts == 0, jnp.inf, visited_scores)
    return jnp.where(occupied_cell_mask, scores, -jnp.inf)


def compute_ucb_cell_scores(
    occupied_cell_mask: Bool[Array, " n_cells"],
    selection_counts: Int[Array, " n_cells"],
    success_counts: Int[Array, " n_cells"],
    total_selections: Int[Array, " "],
    ucb_exploration_constant: Float[Array, " "],
) -> Float[Array, " n_cells"]:
    """Compute cell-based UCB scores for occupied MAP-Elites cells."""
    return compute_cell_scores(
        occupied_cell_mask=occupied_cell_mask,
        selection_counts=selection_counts,
        success_counts=success_counts,
        total_selections=total_selections,
        parent_selection_mode="ucb",
        ucb_exploration_constant=ucb_exploration_constant,
    )


@jax.jit
def sample_best_cell_index(
    cell_scores: Float[Array, " n_cells"],
    random_key: PRNGKeyArray,
) -> Tuple[Int[Array, " "], PRNGKeyArray]:
    """Sample one cell uniformly among the highest-scoring cells.

    Parameters
    ----------
    cell_scores : Float[Array, " n_cells"]
        Per-cell selection scores.
    random_key : PRNGKeyArray
        Random key used to break ties among equally scoring cells.

    Returns
    -------
    Tuple[Int[Array, " "], PRNGKeyArray]
        Sampled cell index and the updated random key.
    """
    candidate_mask = cell_scores == jnp.max(cell_scores)
    random_key, subkey = jax.random.split(random_key)
    candidate_rank = jax.random.randint(subkey, (), 0, jnp.sum(candidate_mask))
    sampled_cell = jnp.argmax(jnp.cumsum(candidate_mask.astype(int)) > candidate_rank).astype(int)
    return sampled_cell, random_key


def _select_batched_ucb_cells(
    occupied_cell_mask: Bool[Array, " n_cells"],
    selection_counts: Int[Array, " n_cells"],
    success_counts: Int[Array, " n_cells"],
    total_selections: Int[Array, " "],
    ucb_exploration_constant: Float[Array, " "],
    random_key: PRNGKeyArray,
    block_size: int,
) -> Tuple[Int[Array, " block_size"], PRNGKeyArray]:
    """Allocate one virtual UCB block with a vectorized top-k selection."""
    n_cells = occupied_cell_mask.shape[0]
    virtual_offsets = jnp.arange(block_size, dtype=int)
    virtual_selection_counts = selection_counts[:, None] + virtual_offsets[None, :]
    safe_virtual_selection_counts = jnp.maximum(virtual_selection_counts, 1)
    virtual_total_selections = total_selections + virtual_offsets

    exploitation = success_counts[:, None].astype(float) / safe_virtual_selection_counts.astype(float)
    exploration = ucb_exploration_constant * jnp.sqrt(
        jnp.log(jnp.maximum(virtual_total_selections.astype(float), 1.0))[None, :]
        / safe_virtual_selection_counts.astype(float)
    )
    finite_scores = exploitation + exploration
    max_occupied_score = jnp.max(jnp.where(occupied_cell_mask[:, None], finite_scores, -jnp.inf))
    unvisited_first_pull_mask = (selection_counts[:, None] == 0) & (virtual_offsets[None, :] == 0)
    scores = jnp.where(unvisited_first_pull_mask, max_occupied_score + 1.0, finite_scores)
    scores = jnp.where(occupied_cell_mask[:, None], scores, -jnp.inf)

    random_key, permutation_key, shuffle_key = jax.random.split(random_key, 3)
    cell_permutation = jax.random.permutation(permutation_key, jnp.arange(n_cells, dtype=int))
    _, selected_virtual_indices = jax.lax.top_k(scores[cell_permutation].reshape((-1,)), block_size)
    selected_cells = cell_permutation[selected_virtual_indices // block_size]
    return jax.random.permutation(shuffle_key, selected_cells), random_key


@partial(jax.jit, static_argnames=("num_samples", "cell_depth", "selection_block_size"))
def select_batched_ucb_parent_indices(
    occupied_mask: Bool[Array, " repertoire_size"],
    selection_counts: Int[Array, " n_cells"],
    success_counts: Int[Array, " n_cells"],
    total_selections: Int[Array, " "],
    ucb_exploration_constant: Float[Array, " "],
    random_key: PRNGKeyArray,
    num_samples: int,
    cell_depth: int,
    selection_block_size: int,
) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
    """Select parent indices with blockwise virtual UCB count updates.

    The exact UCB selector updates its virtual visit counts after every pull. This
    batched variant scores each cell's next virtual pulls at their corresponding
    global selection horizons, selects the highest-scoring pulls with ``top_k``,
    then updates all selected counts together. A block size of one uses the same
    UCB scores as the exact selector; larger blocks approximate its sequential
    recurrence while retaining vectorized operations.

    Parameters
    ----------
    occupied_mask : Bool[Array, " repertoire_size"]
        Boolean mask marking occupied repertoire slots.
    selection_counts : Int[Array, " n_cells"]
        Number of parent selections accumulated for each cell.
    success_counts : Int[Array, " n_cells"]
        Number of successful offspring attributed to each cell.
    total_selections : Int[Array, " "]
        Total number of parent selections accumulated across all cells.
    ucb_exploration_constant : Float[Array, " "]
        Multiplicative factor for the UCB exploration term.
    random_key : PRNGKeyArray
        Random key used to randomize ties and selected-parent order.
    num_samples : int
        Number of parent indices to select.
    cell_depth : int
        Number of repertoire slots stored per logical MAP-Elites cell.
    selection_block_size : int
        Number of virtual pulls allocated together before scores are refreshed.

    Returns
    -------
    Tuple[Int[Array, " num_samples"], PRNGKeyArray]
        Selected occupied repertoire indices and the updated random key.

    Raises
    ------
    ValueError
        If ``selection_block_size`` is not positive.
    """
    if selection_block_size < 1:
        raise ValueError("selection_block_size must be at least 1.")
    if num_samples == 0:
        return jnp.empty((0,), dtype=int), random_key
    if selection_block_size == 1:
        return select_cell_parent_indices(
            occupied_mask=occupied_mask,
            selection_counts=selection_counts,
            success_counts=success_counts,
            total_selections=total_selections,
            parent_selection_mode="ucb",
            ucb_exploration_constant=ucb_exploration_constant,
            random_key=random_key,
            num_samples=num_samples,
            cell_depth=cell_depth,
        )

    occupied_cell_mask = get_occupied_cell_mask(occupied_mask=occupied_mask, cell_depth=cell_depth)
    n_cells = occupied_cell_mask.shape[0]
    n_full_blocks, remainder = divmod(num_samples, selection_block_size)

    def select_full_block(
        carry: tuple[Int[Array, " n_cells"], Int[Array, " "], PRNGKeyArray],
        _unused: None,
    ) -> tuple[tuple[Int[Array, " n_cells"], Int[Array, " "], PRNGKeyArray], Int[Array, " selection_block_size"]]:
        current_selection_counts, current_total_selections, current_random_key = carry
        selected_cells, current_random_key = _select_batched_ucb_cells(
            occupied_cell_mask=occupied_cell_mask,
            selection_counts=current_selection_counts,
            success_counts=success_counts,
            total_selections=current_total_selections,
            ucb_exploration_constant=ucb_exploration_constant,
            random_key=current_random_key,
            block_size=selection_block_size,
        )
        current_selection_counts = current_selection_counts + jnp.bincount(selected_cells, length=n_cells).astype(int)
        current_total_selections = current_total_selections + jnp.array(selection_block_size, dtype=int)
        return (current_selection_counts, current_total_selections, current_random_key), selected_cells

    carry = (selection_counts, total_selections, random_key)
    if n_full_blocks > 0:
        carry, selected_full_blocks = jax.lax.scan(select_full_block, carry, xs=None, length=n_full_blocks)
        selected_cells = selected_full_blocks.reshape((-1,))
    else:
        selected_cells = jnp.empty((0,), dtype=int)

    current_selection_counts, current_total_selections, current_random_key = carry
    if remainder > 0:
        selected_remainder, current_random_key = _select_batched_ucb_cells(
            occupied_cell_mask=occupied_cell_mask,
            selection_counts=current_selection_counts,
            success_counts=success_counts,
            total_selections=current_total_selections,
            ucb_exploration_constant=ucb_exploration_constant,
            random_key=current_random_key,
            block_size=remainder,
        )
        selected_cells = jnp.concatenate((selected_cells, selected_remainder))

    return select_occupied_indices_in_cells(
        occupied_mask=occupied_mask,
        selected_cells=selected_cells,
        random_key=current_random_key,
        cell_depth=cell_depth,
    )


@partial(jax.jit, static_argnames=("num_samples", "cell_depth", "parent_selection_mode"))
def select_cell_parent_indices(
    occupied_mask: Bool[Array, " repertoire_size"],
    selection_counts: Int[Array, " n_cells"],
    success_counts: Int[Array, " n_cells"],
    total_selections: Int[Array, " "],
    parent_selection_mode: ParentSelectionMode,
    ucb_exploration_constant: Float[Array, " "],
    random_key: PRNGKeyArray,
    num_samples: int,
    cell_depth: int,
) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
    """Select repertoire indices by first choosing cells with the best score.

    Parameters
    ----------
    occupied_mask : Bool[Array, " repertoire_size"]
        Boolean mask marking occupied repertoire slots.
    selection_counts : Int[Array, " n_cells"]
        Number of parent selections accumulated for each cell.
    success_counts : Int[Array, " n_cells"]
        Number of successful offspring attributed to each cell.
    total_selections : Int[Array, " "]
        Total number of parent selections accumulated so far.
    parent_selection_mode : ParentSelectionMode
        Cell-based mode used to score parent cells.
    ucb_exploration_constant : Float[Array, " "]
        Multiplicative factor for the exploration term in the UCB score.
    random_key : PRNGKeyArray
        Random key used for stochastic tie breaking and layer sampling.
    num_samples : int
        Number of parent indices to select.
    cell_depth : int
        Number of repertoire slots stored per logical MAP-Elites cell.

    Returns
    -------
    Tuple[Int[Array, " num_samples"], PRNGKeyArray]
        Selected repertoire indices and the updated random key.
    """
    occupied_cell_mask = get_occupied_cell_mask(occupied_mask=occupied_mask, cell_depth=cell_depth)

    def select_one_parent(
        carry: tuple[Int[Array, " n_cells"], Int[Array, " "], PRNGKeyArray],
        _unused: None,
    ) -> tuple[tuple[Int[Array, " n_cells"], Int[Array, " "], PRNGKeyArray], Int[Array, " "]]:
        current_selection_counts, current_total_selections, current_random_key = carry
        cell_scores = compute_cell_scores(
            occupied_cell_mask=occupied_cell_mask,
            selection_counts=current_selection_counts,
            success_counts=success_counts,
            total_selections=current_total_selections,
            parent_selection_mode=parent_selection_mode,
            ucb_exploration_constant=ucb_exploration_constant,
        )
        selected_cell, current_random_key = sample_best_cell_index(
            cell_scores=cell_scores,
            random_key=current_random_key,
        )
        current_selection_counts = current_selection_counts.at[selected_cell].add(1)
        current_total_selections = current_total_selections + jnp.array(1, dtype=int)
        return (current_selection_counts, current_total_selections, current_random_key), selected_cell

    (_selection_counts, _total_selections, random_key), selected_cells = jax.lax.scan(
        select_one_parent,
        (selection_counts, total_selections, random_key),
        xs=None,
        length=num_samples,
    )

    return select_occupied_indices_in_cells(
        occupied_mask=occupied_mask,
        selected_cells=selected_cells,
        random_key=random_key,
        cell_depth=cell_depth,
    )


@partial(jax.jit, static_argnames=("num_samples", "cell_depth"))
def select_ucb_parent_indices(
    occupied_mask: Bool[Array, " repertoire_size"],
    selection_counts: Int[Array, " n_cells"],
    success_counts: Int[Array, " n_cells"],
    total_selections: Int[Array, " "],
    ucb_exploration_constant: Float[Array, " "],
    random_key: PRNGKeyArray,
    num_samples: int,
    cell_depth: int,
) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
    """Select repertoire indices using UCB cell scores."""
    return select_cell_parent_indices(
        occupied_mask=occupied_mask,
        selection_counts=selection_counts,
        success_counts=success_counts,
        total_selections=total_selections,
        parent_selection_mode="ucb",
        ucb_exploration_constant=ucb_exploration_constant,
        random_key=random_key,
        num_samples=num_samples,
        cell_depth=cell_depth,
    )


@partial(jax.jit, static_argnames=("num_samples", "cell_depth"))
def select_snapshot_ucb_parent_indices(
    occupied_mask: Bool[Array, " repertoire_size"],
    selection_counts: Int[Array, " n_cells"],
    success_counts: Int[Array, " n_cells"],
    total_selections: Int[Array, " "],
    ucb_exploration_constant: Float[Array, " "],
    temperature: Float[Array, " "],
    random_key: PRNGKeyArray,
    num_samples: int,
    cell_depth: int,
) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
    """Sample parents from one UCB score snapshot, prioritizing unvisited cells.

    Each unvisited occupied cell receives one parent selection before the remaining
    positions are independently sampled from a temperature-scaled distribution over
    the UCB scores computed at the start of the emission.
    """
    if num_samples == 0:
        return jnp.empty((0,), dtype=int), random_key

    occupied_cell_mask = get_occupied_cell_mask(occupied_mask=occupied_mask, cell_depth=cell_depth)
    n_cells = occupied_cell_mask.shape[0]
    unvisited_cell_mask = occupied_cell_mask & (selection_counts == 0)
    visited_cell_mask = occupied_cell_mask & ~unvisited_cell_mask
    has_visited_cells = jnp.any(visited_cell_mask)
    first_occupied_cell = jnp.argmax(occupied_cell_mask.astype(int))

    random_key, priority_key = jax.random.split(random_key)
    tie_breakers = jax.random.uniform(priority_key, shape=(n_cells,))
    ranked_unvisited_cells = jnp.argsort(jnp.where(unvisited_cell_mask, tie_breakers, jnp.inf))
    priority_pool_size = min(num_samples, n_cells)
    priority_cells = jnp.concatenate(
        (
            ranked_unvisited_cells[:priority_pool_size],
            jnp.full((num_samples - priority_pool_size,), first_occupied_cell, dtype=int),
        )
    )
    priority_count = jnp.minimum(jnp.sum(unvisited_cell_mask.astype(int)), num_samples)

    def sample_priority_indices(key: PRNGKeyArray) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
        """Sample one occupied repertoire entry for each priority cell."""
        return select_occupied_indices_in_cells(
            occupied_mask=occupied_mask,
            selected_cells=priority_cells,
            random_key=key,
            cell_depth=cell_depth,
        )

    def skip_priority_sampling(key: PRNGKeyArray) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
        """Avoid unused cell-depth sampling after every occupied cell was visited."""
        return jnp.full((num_samples,), first_occupied_cell, dtype=int), key

    priority_indices, random_key = jax.lax.cond(
        priority_count > 0,
        sample_priority_indices,
        skip_priority_sampling,
        random_key,
    )

    cell_scores = compute_ucb_cell_scores(
        occupied_cell_mask=occupied_cell_mask,
        selection_counts=selection_counts,
        success_counts=success_counts,
        total_selections=total_selections,
        ucb_exploration_constant=ucb_exploration_constant,
    )
    max_visited_score = jnp.max(jnp.where(visited_cell_mask, cell_scores, -jnp.inf))
    safe_max_visited_score = jnp.where(has_visited_cells, max_visited_score, 0.0)
    safe_cell_scores = jnp.where(visited_cell_mask, cell_scores, safe_max_visited_score)
    visited_weights = jnp.where(
        visited_cell_mask,
        jnp.exp((safe_cell_scores - safe_max_visited_score) / temperature),
        0.0,
    )
    visited_probabilities = visited_weights / jnp.maximum(jnp.sum(visited_weights), 1.0)
    uniform_probabilities = occupied_cell_mask.astype(float) / jnp.sum(occupied_cell_mask)
    cell_probabilities = jnp.where(has_visited_cells, visited_probabilities, uniform_probabilities)

    occupied_by_layer = occupied_mask.reshape((cell_depth, n_cells))
    occupied_per_cell = jnp.sum(occupied_by_layer, axis=0)
    slot_probabilities = jnp.tile(cell_probabilities / jnp.maximum(occupied_per_cell, 1), cell_depth)
    slot_probabilities = jnp.where(occupied_mask, slot_probabilities, 0.0)
    cumulative_probabilities = jnp.cumsum(slot_probabilities)

    random_key, sample_key = jax.random.split(random_key)
    random_values = jax.random.uniform(sample_key, shape=(num_samples,))
    sampled_indices = jnp.searchsorted(cumulative_probabilities, random_values, side="right").astype(int)
    sampled_indices = jnp.minimum(sampled_indices, occupied_mask.shape[0] - 1)
    is_priority_selection = jnp.arange(num_samples, dtype=int) < priority_count
    return jnp.where(is_priority_selection, priority_indices, sampled_indices), random_key


@partial(jax.jit, static_argnames=("cell_depth",))
def select_occupied_indices_in_cells(
    occupied_mask: Bool[Array, " repertoire_size"],
    selected_cells: Int[Array, " num_samples"],
    random_key: PRNGKeyArray,
    cell_depth: int,
) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
    """Sample one occupied candidate from each selected MAP-Elites cell."""
    if cell_depth == 1:
        return selected_cells, random_key

    num_cells = occupied_mask.shape[0] // cell_depth
    occupied_by_layer = occupied_mask.reshape((cell_depth, num_cells))
    occupied_layers = occupied_by_layer[:, selected_cells].T
    logits = jnp.where(occupied_layers, 0.0, -jnp.inf)
    random_key, subkey = jax.random.split(random_key)
    layer_keys = jax.random.split(subkey, selected_cells.shape[0])
    selected_layers = jax.vmap(jax.random.categorical)(layer_keys, logits).astype(int)
    return selected_cells + selected_layers * num_cells, random_key


@partial(jax.jit, static_argnames=("num_samples", "cell_depth"))
def select_uniform_cell_parent_indices(
    occupied_mask: Bool[Array, " repertoire_size"],
    random_key: PRNGKeyArray,
    num_samples: int,
    cell_depth: int,
) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
    """Select parent candidates uniformly across occupied MAP-Elites cells."""
    occupied_cell_mask = get_occupied_cell_mask(occupied_mask=occupied_mask, cell_depth=cell_depth)
    probabilities = occupied_cell_mask.astype(float)
    probabilities = probabilities / jnp.sum(probabilities)

    random_key, subkey = jax.random.split(random_key)
    selected_cells = jax.random.choice(
        subkey,
        jnp.arange(occupied_cell_mask.shape[0], dtype=int),
        shape=(num_samples,),
        p=probabilities,
    )
    return select_occupied_indices_in_cells(
        occupied_mask=occupied_mask,
        selected_cells=selected_cells,
        random_key=random_key,
        cell_depth=cell_depth,
    )


@partial(jax.jit, static_argnames=("num_samples",))
def select_uniform_parent_indices(
    occupied_mask: Bool[Array, " repertoire_size"],
    random_key: PRNGKeyArray,
    num_samples: int,
) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
    """Sample occupied repertoire entries uniformly.

    Parameters
    ----------
    occupied_mask : Bool[Array, " repertoire_size"]
        Boolean mask marking occupied repertoire entries.
    random_key : PRNGKeyArray
        Random key used for sampling.
    num_samples : int
        Number of repertoire entries to sample.

    Returns
    -------
    Tuple[Int[Array, " num_samples"], PRNGKeyArray]
        Sampled repertoire indices and the updated random key.
    """
    probabilities = occupied_mask.astype(float)
    probabilities = probabilities / jnp.sum(probabilities)

    random_key, subkey = jax.random.split(random_key)
    repertoire_indices = jnp.arange(occupied_mask.shape[0], dtype=int)
    sampled_indices = jax.random.choice(subkey, repertoire_indices, shape=(num_samples,), p=probabilities)

    return sampled_indices, random_key


@partial(jax.jit, static_argnames=("num_samples",))
def select_greedy_parent_indices(
    occupied_mask: Bool[Array, " repertoire_size"],
    fitnesses: Float[Array, " repertoire_size"],
    random_key: PRNGKeyArray,
    num_samples: int,
) -> Tuple[Int[Array, " num_samples"], PRNGKeyArray]:
    """Select candidates uniformly among the occupied highest-fitness entries."""
    masked_fitnesses = jnp.where(occupied_mask, fitnesses, -jnp.inf)
    best_fitness = jnp.max(masked_fitnesses)
    probabilities = (masked_fitnesses == best_fitness).astype(float)
    probabilities = probabilities / jnp.sum(probabilities)

    random_key, subkey = jax.random.split(random_key)
    sampled_indices = jax.random.choice(
        subkey,
        jnp.arange(occupied_mask.shape[0], dtype=int),
        shape=(num_samples,),
        p=probabilities,
    )
    return sampled_indices, random_key
