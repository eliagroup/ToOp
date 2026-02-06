# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
from toop_engine_dc_solver.jax.types import NodalInjOptimResults
from toop_engine_topology_optimizer.dc.genetic_functions.evolution_functions import (
    Genotype,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import (
    DiscreteMapElitesRepertoire,
    _init_default,
    add_to_repertoire,
    get_cell_index,
    get_cells_indices,
    init_repertoire,
)


def test_get_cells_indices() -> None:
    n_cells_per_dim = (4, 20)

    descriptor = jnp.array([0, 0])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 0

    descriptor = jnp.array([1, 0])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 20

    descriptor = jnp.array([0, 1])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 1

    descriptor = jnp.array([1, 1])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 21

    descriptor = jnp.array([3, 19])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 79

    descriptor = jnp.array([999, 999])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 79

    descriptor = jnp.array([-1, 1])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 1

    n_cells_per_dim = (4, 20, 10)

    descriptor = jnp.array([0, 0, 0])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 0

    descriptor = jnp.array([1, 0, 0])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 200

    descriptor = jnp.array([0, 1, 0])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 10

    descriptor = jnp.array([1, 1, 0])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 210

    descriptor = jnp.array([3, 19, 9])
    cell_index = get_cell_index(descriptor, n_cells_per_dim)
    assert cell_index == 799

    descriptors = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [3, 19, 9]])
    cell_indices = get_cells_indices(descriptors, n_cells_per_dim)
    assert jnp.array_equal(cell_indices, jnp.array([0, 200, 10, 210, 799]))


def test_create_repertoire() -> None:
    batch_size = 4
    genotypes = {
        "bla": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 10)),
        "blu": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 42)),
    }
    n_cells_per_dim = (4, 20, 10)
    descriptors = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 15, 8]], dtype=int)
    fitnesses = jax.random.normal(jax.random.PRNGKey(0), (batch_size,))

    repertoire = init_repertoire(
        genotypes=genotypes,
        n_cells_per_dim=n_cells_per_dim,
        descriptors=descriptors,
        fitnesses=fitnesses,
        extra_scores=None,
        cell_depth=1,
    )

    assert repertoire.fitnesses.shape == (4 * 20 * 10,)

    cell_indices = get_cells_indices(descriptors, n_cells_per_dim)
    for i, index in enumerate(cell_indices):
        assert repertoire.fitnesses[index] == fitnesses[i]
        assert repertoire.descriptors[index].tolist() == descriptors[i].tolist()
        assert repertoire.genotypes["bla"][index].tolist() == genotypes["bla"][i].tolist()
        assert repertoire.genotypes["blu"][index].tolist() == genotypes["blu"][i].tolist()


def test_create_repertoire_with_extra_scores() -> None:
    batch_size = 4
    genotypes = {
        "bla": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 10)),
        "blu": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 42)),
    }
    n_cells_per_dim = (4, 20, 10)
    descriptors = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=int)
    fitnesses = jax.random.normal(jax.random.PRNGKey(0), (batch_size,))
    extra_scores = {
        "score1": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 10)),
        "score2": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 42)),
    }

    repertoire = init_repertoire(
        genotypes=genotypes,
        n_cells_per_dim=n_cells_per_dim,
        descriptors=descriptors,
        fitnesses=fitnesses,
        extra_scores=extra_scores,
        cell_depth=1,
    )

    assert repertoire.fitnesses.shape == (4 * 20 * 10,)
    assert repertoire.extra_scores["score1"].shape == (4 * 20 * 10, 10)

    cell_indices = get_cells_indices(descriptors, n_cells_per_dim)
    for i, index in enumerate(cell_indices):
        assert repertoire.fitnesses[index] == fitnesses[i]
        assert repertoire.descriptors[index].tolist() == descriptors[i].tolist()
        assert repertoire.genotypes["bla"][index].tolist() == genotypes["bla"][i].tolist()
        assert repertoire.genotypes["blu"][index].tolist() == genotypes["blu"][i].tolist()
        assert repertoire.extra_scores["score1"][index].tolist() == extra_scores["score1"][i].tolist()
        assert repertoire.extra_scores["score2"][index].tolist() == extra_scores["score2"][i].tolist()


def test_create_repertoire_with_cell_depth() -> None:
    """
    Create a repertoire with a batch of 4 individuals and a cell depth of 2.
    Special cases :
    - Two individuals belong to the same cell
    """
    # Parameters
    batch_size = 4
    cell_depth = 2
    n_cells_per_dim = (3,)

    # Fake data
    nodal_injections_optimized = NodalInjOptimResults(
        pst_taps=jnp.zeros((batch_size, 1)),
    )
    genotypes = Genotype(  # 4 topologies
        action_index=jnp.array([[0], [1], [2], [0]]),
        disconnections=jnp.array([[0], [1], [2], [0]]),
        nodal_injections_optimized=nodal_injections_optimized,
    )
    extra_scores = {
        "score1": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 10)),
        "score2": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 42)),
        "score3": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 10)),
        "score4": jax.random.normal(jax.random.PRNGKey(0), (batch_size, 42)),
    }
    descriptors = jnp.array(
        [[0], [1], [0], [6]], dtype=int
    )  # cell 0 depth 1, cell 1 depth 0, cell 0 depth 0, cell 2 depth 0
    expected_fitnesses_positions = jnp.array([3, 1, 0, 2])  # cell 0 depth 1, cell 1 depth 0, cell 0 depth 0, cell 2 depth 0
    fitnesses = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Create repertoire
    with jax.disable_jit():
        repertoire = init_repertoire(
            genotypes=genotypes,
            n_cells_per_dim=n_cells_per_dim,
            descriptors=descriptors,
            fitnesses=fitnesses,
            extra_scores=extra_scores,
            cell_depth=cell_depth,
        )

    assert repertoire.fitnesses.shape[0] == (jnp.prod(jnp.array(n_cells_per_dim)) * cell_depth).item()

    assert jnp.array_equal(repertoire.fitnesses[expected_fitnesses_positions], fitnesses)


def test_add_to_repertoire_with_depth():
    """
    Tests the most common cases for adding an individual to a repertoire with a cell depth of 2.

    Cases :
        [ADD]
    - 0) A new individual needs to be added in the first layer
    - 0-deep) A new individual needs to be added in a deep layer

        [REPLACE]
    - 1) A new individual needs to replace a first layer individual, which needs to be pushed to a deep layer
    - 1-deep) A new individual needs to replace a last layer individual, which needs to be deleted

        [SIMULTANEOUS]
    - 2) Two individuals must be added to the same cell (same cell indices) at the same time

        [IGNORE]
    - 3) A new individual has worse fitness than the ones in its full cell
    - 3-same) Two individuals have the same genotype (should be ignored or will take space in the cell)  # decision was made to ignore this case
    """
    # Parameters
    cell_depth = 2
    n_cells_per_dim = (3,)  # possible indices are 0, 1, 2
    num_cells = jnp.prod(jnp.array(n_cells_per_dim)).item()

    def _change_layer(index: int, layer: int) -> int:
        """Get the index of the same cell in the nth layer"""
        return index + num_cells * layer

    # Create empty repertoire
    nodal_injections_optimized = NodalInjOptimResults(
        pst_taps=jnp.zeros(([1])),
    )
    initial_genotype = Genotype(
        action_index=jnp.array([0]),  # must only contain one genotype, not a list of genotype
        disconnections=jnp.array([0]),
        nodal_injections_optimized=nodal_injections_optimized,
    )
    initial_extra_score = {
        "score": jnp.array([0.0]),
    }
    empty_repertoire: DiscreteMapElitesRepertoire = _init_default(
        genotype=initial_genotype,
        extra_scores=initial_extra_score,
        n_cells_per_dim=n_cells_per_dim,
        cell_depth=cell_depth,
    )

    assert empty_repertoire.fitnesses.shape[0] == num_cells * cell_depth

    # ___________________________________________
    # Case 0) A new individual needs to be added in the first layer
    nodal_injections_optimized = NodalInjOptimResults(
        pst_taps=jnp.array([[0.0]]),
    )
    genotypes_zero = Genotype(  # this time it's a batch, so there is a list of genotypes
        action_index=jnp.array([[1]]),
        disconnections=jnp.array([[1]]),
        nodal_injections_optimized=nodal_injections_optimized,
    )
    descriptors_zero = jnp.array([[1]])
    index_zero = get_cells_indices(descriptors_zero, n_cells_per_dim)[0]
    fitnesses_zero = jnp.array([1.0])
    extra_scores_zero = None

    with jax.disable_jit():
        repertoire_zero: DiscreteMapElitesRepertoire = add_to_repertoire(
            repertoire=empty_repertoire,
            batch_of_genotypes=genotypes_zero,
            batch_of_descriptors=descriptors_zero,
            batch_of_fitnesses=fitnesses_zero,
            batch_of_extra_scores=extra_scores_zero,
        )

    assert empty_repertoire.fitnesses.shape[0] == repertoire_zero.fitnesses.shape[0]

    # First layer should be filled
    assert repertoire_zero.fitnesses[index_zero] == fitnesses_zero[0], "Case 0, first layer : wrong fitness"
    assert jnp.array_equal(repertoire_zero.genotypes.action_index[index_zero], genotypes_zero.action_index[0]), (
        "Case 0, first layer : wrong genotype"
    )

    # Second layer of the same cell should be empty
    second_layer_index = _change_layer(index_zero, 1)
    assert repertoire_zero.fitnesses[second_layer_index].item() == -jnp.inf, "Case 0, second layer : wrong fitness"
    assert jnp.array_equal(
        repertoire_zero.genotypes.action_index[second_layer_index],
        empty_repertoire.genotypes.action_index[second_layer_index],
    ), "Case 0, second layer : wrong genotype"

    # ___________________________________________
    # Case 0-deep) A new individual needs to be added in a deep layer
    """Plan : add a new individual to the same repertoire repertoire_zero but with a fitness that is half of the first one"""
    nodal_injections_optimized = NodalInjOptimResults(
        pst_taps=jnp.array([[1.0]]),
    )
    genotypes_zero_deep = Genotype(
        action_index=jnp.array([[2]]),
        disconnections=jnp.array([[1]]),
        nodal_injections_optimized=nodal_injections_optimized,
    )
    descriptors_zero_deep = descriptors_zero
    index_zero_deep = index_zero
    fitnesses_zero_deep = fitnesses_zero / 2
    extra_scores_zero_deep = None

    with jax.disable_jit():
        repertoire_zero_deep: DiscreteMapElitesRepertoire = add_to_repertoire(
            repertoire=repertoire_zero,
            batch_of_genotypes=genotypes_zero_deep,
            batch_of_descriptors=descriptors_zero_deep,
            batch_of_fitnesses=fitnesses_zero_deep,
            batch_of_extra_scores=extra_scores_zero_deep,
        )

    # First layer is unchanged
    assert jnp.array_equal(
        repertoire_zero.fitnesses[index_zero_deep],
        repertoire_zero_deep.fitnesses[index_zero_deep],
    ), "Case 0-deep, first layer : wrong fitness"
    assert jnp.array_equal(repertoire_zero.genotypes.action_index[index_zero_deep], genotypes_zero.action_index[0]), (
        "Case 0-deep, first layer : wrong genotype"
    )

    # Second layer should now be filled
    second_layer_index = _change_layer(index_zero_deep, 1)
    assert jnp.array_equal(repertoire_zero_deep.fitnesses[second_layer_index], fitnesses_zero_deep[0]), (
        "Case 0-deep, second layer : wrong fitness"
    )

    assert jnp.array_equal(
        repertoire_zero_deep.genotypes.action_index[second_layer_index],
        genotypes_zero_deep.action_index[0],
    ), "Case 0-deep, second layer : wrong genotype"

    # ___________________________________________
    # Case 1) A new individual needs to replace a first layer individual, which needs to be pushed to a deep layer
    """Plan : add a new individual to the repertoire repertoire_zero with a better fitness"""
    nodal_injections_optimized = NodalInjOptimResults(
        pst_taps=jnp.array([[1.0]]),
    )
    genotypes_one = Genotype(
        action_index=jnp.array([[3]]),
        disconnections=jnp.array([[1]]),
        nodal_injections_optimized=nodal_injections_optimized,
    )
    descriptors_one = descriptors_zero
    index_one = index_zero
    fitnesses_one = fitnesses_zero * 2  # Better fitness = store in first layer
    extra_scores_one = None

    with jax.disable_jit():
        repertoire_one: DiscreteMapElitesRepertoire = add_to_repertoire(
            repertoire=repertoire_zero,
            batch_of_genotypes=genotypes_one,
            batch_of_descriptors=descriptors_one,
            batch_of_fitnesses=fitnesses_one,
            batch_of_extra_scores=extra_scores_one,
        )

    # First layer should be updated and contain the new individual
    assert repertoire_one.fitnesses[index_one] == fitnesses_one[0], "Case 1, first layer : wrong fitness"
    assert jnp.array_equal(repertoire_one.genotypes.action_index[index_one], genotypes_one.action_index[0]), (
        "Case 1, first layer : wrong genotype"
    )

    # Second layer should be updated and contain the previous first layer individual
    second_layer_index = _change_layer(index_one, 1)
    assert repertoire_one.fitnesses[second_layer_index] == fitnesses_zero[0], "Case 1, second layer : wrong fitness"
    assert jnp.array_equal(repertoire_one.genotypes.action_index[second_layer_index], genotypes_zero.action_index[0]), (
        "Case 1, second layer : wrong genotype"
    )

    # ___________________________________________
    # Case 1-deep) A new individual needs to replace a last layer individual, which needs to be deleted
    """Plan : add a new individual to the repertoire repertoire_zero_deep, which already contains two individuals in the desired cell"""

    genotypes_one_deep = genotypes_one
    descriptors_one_deep = descriptors_one
    index_one_deep = index_one
    fitnesses_one_deep = (fitnesses_zero_deep + fitnesses_zero) / cell_depth  # fitness is in between the two others

    with jax.disable_jit():
        repertoire_one_deep: DiscreteMapElitesRepertoire = add_to_repertoire(
            repertoire=repertoire_zero_deep,
            batch_of_genotypes=genotypes_one_deep,
            batch_of_descriptors=descriptors_one_deep,
            batch_of_fitnesses=fitnesses_one_deep,
            batch_of_extra_scores=extra_scores_one,
        )

    # First layer should be unchanged
    assert jnp.array_equal(
        repertoire_one_deep.fitnesses[index_one_deep],
        repertoire_zero_deep.fitnesses[index_one_deep],
    ), "Case 1-deep, first layer : wrong fitness"
    assert jnp.array_equal(
        repertoire_one_deep.genotypes.action_index[index_one_deep],
        repertoire_zero_deep.genotypes.action_index[index_one_deep],
    ), "Case 1-deep, first layer : wrong genotype"

    # Second layer should have the new individual
    second_layer_index = _change_layer(index_one_deep, 1)
    assert repertoire_one_deep.fitnesses[second_layer_index] == fitnesses_one_deep[0], (
        "Case 1-deep, second layer : wrong fitness"
    )
    assert jnp.array_equal(
        repertoire_one_deep.genotypes.action_index[second_layer_index],
        genotypes_one_deep.action_index[0],
    ), "Case 1-deep, second layer : wrong genotype"

    # ___________________________________________
    # Case 2) Two individuals must be added to the same cell (same cell indices) at the same time
    """Plan : add genotypes_zero and genotypes_zero_deep to the empty repertoire and check that it is repertoire_zero_deep"""

    genotypes_two = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=0),
        genotypes_zero,
        genotypes_zero_deep,
    )
    descriptors_two = jnp.concatenate([descriptors_zero, descriptors_zero_deep], axis=0)
    index_two = index_zero
    fitnesses_two = jnp.concatenate([fitnesses_zero, fitnesses_zero_deep], axis=0)
    extra_scores_two = None

    with jax.disable_jit():
        repertoire_two: DiscreteMapElitesRepertoire = add_to_repertoire(
            repertoire=empty_repertoire,
            batch_of_genotypes=genotypes_two,
            batch_of_descriptors=descriptors_two,
            batch_of_fitnesses=fitnesses_two,
            batch_of_extra_scores=extra_scores_two,
        )

    # First layer should be filled with genotype_zero
    assert repertoire_two.fitnesses[index_two] == fitnesses_zero[0], "Case 2, first layer : wrong fitness"
    assert jnp.array_equal(repertoire_two.genotypes.action_index[index_two], genotypes_zero.action_index[0]), (
        "Case 2, first layer : wrong genotype"
    )

    # Second layer should be filled with genotype_zero_deep
    second_layer_index = _change_layer(index_two, 1)
    assert repertoire_two.fitnesses[second_layer_index] == fitnesses_zero_deep[0], "Case 2, second layer : wrong fitness"
    assert jnp.array_equal(
        repertoire_two.genotypes.action_index[second_layer_index],
        genotypes_zero_deep.action_index[0],
    ), "Case 2, second layer : wrong genotype"

    # ___________________________________________
    # Case 3) A new individual has worse fitness than the ones in its full cell
    """Plan : add a new individual to the repertoire_zero_deep, which has 2 individuals with better fitnesses"""
    nodal_injections_optimized = NodalInjOptimResults(
        pst_taps=jnp.array([[5.0]]),
    )
    genotypes_three = Genotype(
        action_index=jnp.array([[4]]),
        disconnections=jnp.array([[1]]),
        nodal_injections_optimized=nodal_injections_optimized,
    )
    descriptors_three = descriptors_zero
    index_three = index_zero
    fitnesses_three = fitnesses_zero - 1000.0  # Worse fitness

    with jax.disable_jit():
        repertoire_three: DiscreteMapElitesRepertoire = add_to_repertoire(
            repertoire=repertoire_zero_deep,
            batch_of_genotypes=genotypes_three,
            batch_of_descriptors=descriptors_three,
            batch_of_fitnesses=fitnesses_three,
            batch_of_extra_scores=extra_scores_one,
        )

    # First layer should be unchanged
    assert jnp.array_equal(
        repertoire_three.fitnesses[index_three],
        repertoire_zero_deep.fitnesses[index_three],
    ), "Case 3, first layer : wrong fitness"
    assert jnp.array_equal(
        repertoire_three.genotypes.action_index[index_three],
        repertoire_zero_deep.genotypes.action_index[index_three],
    ), "Case 3, first layer : wrong genotype"

    # Second layer should be unchanged
    second_layer_index = _change_layer(index_three, 1)
    assert jnp.array_equal(
        repertoire_three.fitnesses[second_layer_index],
        repertoire_zero_deep.fitnesses[second_layer_index],
    ), "Case 3, second layer : wrong fitness"
    assert jnp.array_equal(
        repertoire_three.genotypes.action_index[second_layer_index],
        repertoire_zero_deep.genotypes.action_index[second_layer_index],
    ), "Case 3, second layer : wrong genotype"

    # ___________________________________________
    # Case 3-same) Two individuals have the same genotype (
    """Plan : add genotype_zero twice to the empty repertoire and check that only one is added (= it is repertoire_zero)
    It has been decided not to handle this case. The code will remain present in case it is needed later.
    """

    """genotypes_three_same = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=0),
        genotypes_zero,
        genotypes_zero,
    )
    descriptors_three_same = jnp.concatenate(
        [descriptors_zero, descriptors_zero], axis=0
    )
    index_three_same = index_zero
    fitnesses_three_same = jnp.concatenate([fitnesses_zero, fitnesses_zero], axis=0)

    with jax.disable_jit():
        repertoire_three_same: DiscreteMapElitesRepertoire = add_to_repertoire(
            repertoire=empty_repertoire,
            batch_of_genotypes=genotypes_three_same,
            batch_of_descriptors=descriptors_three_same,
            batch_of_fitnesses=fitnesses_three_same,
            batch_of_extra_scores=extra_scores_two,
        )

    # First layer should be filled with genotype_zero
    assert (
        repertoire_three_same.fitnesses[index_three_same] == fitnesses_zero[0]
    ), "Case 3-same, first layer : wrong fitness"
    assert jnp.array_equal(
        repertoire_three_same.genotypes.sub_ids[index_three_same],
        genotypes_zero.sub_ids[0],
    ), "Case 3-same, first layer : wrong genotype"

    # Second layer should be empty
    second_layer_index = _change_layer(index_three_same, 1)
    assert (
        repertoire_three_same.fitnesses[second_layer_index].item() == -jnp.inf
    ), "Case 3-same, second layer : wrong fitness"
    assert jnp.array_equal(
        repertoire_three_same.genotypes.sub_ids[second_layer_index],
        empty_repertoire.genotypes.sub_ids[second_layer_index],
    ), "Case 3-same, second layer : wrong genotype"
    """


def test_add_to_repertoire_aranged_data():
    """
    Test with aranged data to make it easier to see the movements in the repertoire.
    redundant with the previous test but useful to debug
    """
    # Parameters
    batch_size = 4
    cell_depth = 2
    n_cells_per_dim = (2,)  # possible indices are 0, 1, 2
    jnp.prod(jnp.array(n_cells_per_dim)).item()

    # Create empty repertoire
    nodal_injections_optimized = NodalInjOptimResults(
        pst_taps=jnp.zeros((batch_size, 1)),
    )
    initial_genotype = Genotype(
        action_index=jnp.array([[0], [1], [2], [3]]),
        disconnections=jnp.array([[0], [1], [2], [3]]),
        nodal_injections_optimized=nodal_injections_optimized,
    )
    initial_fitness = -jnp.array([0.0, 1.0, 2.0, 3.0])  # minus sign so the first ones are the best
    initial_descriptors = jnp.array([[0], [1], [2], [3]])
    initial_extra_score = {
        "score": jnp.array([[0.0], [1.0], [2.0], [3.0]]),
    }
    base_repertoire = DiscreteMapElitesRepertoire(
        genotypes=initial_genotype,
        fitnesses=initial_fitness,
        descriptors=initial_descriptors,
        extra_scores=initial_extra_score,
        n_cells_per_dim=n_cells_per_dim,
        cell_depth=cell_depth,
    )

    batch_of_fitnesses = jnp.arange(batch_size)
    nodal_injections_optimized = NodalInjOptimResults(
        pst_taps=(jnp.arange(batch_size) + 4).reshape(-1, 1),
    )
    batch_of_genotypes = Genotype(
        action_index=(jnp.arange(batch_size) + 4).reshape(-1, 1),
        disconnections=(jnp.arange(batch_size) + 4).reshape(-1, 1),
        nodal_injections_optimized=nodal_injections_optimized,
    )
    batch_of_descriptors = jnp.arange(batch_size).reshape(-1, 1)
    batch_of_extra_scores = {
        "score": jnp.arange(batch_size) + 4.0,
    }
    batch_of_extra_scores = None

    with jax.disable_jit():
        repertoire: DiscreteMapElitesRepertoire = add_to_repertoire(
            repertoire=base_repertoire,
            batch_of_genotypes=batch_of_genotypes,
            batch_of_descriptors=batch_of_descriptors,
            batch_of_fitnesses=batch_of_fitnesses,
            batch_of_extra_scores=batch_of_extra_scores,
        )

    assert jnp.array_equal(
        repertoire.fitnesses,
        jnp.array([-0.0, 3.0, 0.0, 2.0]),  # first layer is -0, 3 / second layer is 0, 2
    ), "Wrong fitnesses"

    assert jnp.array_equal(
        repertoire.genotypes.action_index,
        jnp.array([[0], [7], [4], [6]]),
    ), "Wrong action index"

    if batch_of_extra_scores is not None:
        assert jnp.array_equal(
            repertoire.extra_scores["score"],
            jnp.array([[0.0], [7.0], [4.0], [6.0]]),
        ), "Wrong extra_scores"

    return repertoire
