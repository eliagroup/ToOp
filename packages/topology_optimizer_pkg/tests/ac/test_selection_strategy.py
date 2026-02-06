# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session
from toop_engine_topology_optimizer.ac.evolution_functions import (
    default_scorer,
)
from toop_engine_topology_optimizer.ac.select_strategy import (
    filter_metrics_df,
    get_discriminator_df,
    get_discriminator_mask,
    get_dominator_mask,
    get_median_mask,
    get_repertoire_filter_mask,
    select_strategy,
)
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import FilterStrategy, OptimizerType
from toop_engine_topology_optimizer.interfaces.models.base_storage import BaseDBTopology


def test_select_strategy(dc_repertoire: list[BaseDBTopology]) -> None:
    strategy = select_strategy(np.random.default_rng(0), dc_repertoire, default_scorer)
    assert isinstance(strategy, list)
    assert len(strategy)
    assert isinstance(strategy[0], ACOptimTopology)

    # Exactly one strategy should be selected
    assert len(set(t.strategy_hash for t in strategy)) == 1

    # All topologies in the repertoire belonging to this strategy should be included
    for topo in dc_repertoire:
        if topo.strategy_hash == strategy[0].strategy_hash:
            assert topo in strategy

    assert select_strategy(np.random.default_rng(0), [], default_scorer) == []


def test_select_stategy_ac_dc_mix(dc_repertoire: list[BaseDBTopology], session: Session) -> None:
    # Copy the dc strategies to the AC database (as if they were pulled before)
    mixed_topologies = []
    for topology in dc_repertoire:
        metrics = topology.metrics
        metrics["fitness_dc"] = topology.fitness
        new_topo = ACOptimTopology(
            actions=topology.actions,
            disconnections=topology.disconnections,
            pst_setpoints=topology.pst_setpoints,
            unsplit=topology.unsplit,
            timestep=topology.timestep,
            strategy_hash=topology.strategy_hash,
            optimization_id=topology.optimization_id,
            optimizer_type=OptimizerType.AC,
            fitness=topology.fitness,
            metrics=metrics,
        )
        session.add(new_topo)
        session.commit()
        session.refresh(new_topo)
        mixed_topologies.append(new_topo)
        mixed_topologies.append(topology)

    # Select a strategy
    strategy = select_strategy(np.random.default_rng(0), mixed_topologies, default_scorer)
    assert isinstance(strategy, list)
    assert len(strategy)
    assert isinstance(strategy[0], ACOptimTopology)
    assert len(set(t.strategy_hash for t in strategy)) == 1
    assert len(set(t.optimizer_type for t in strategy)) == 1


@pytest.fixture
def sample_metrics_2d_df():
    # Create a simple DataFrame for testing
    data = [
        {"switching_distance": 1, "split_subs": 1, "fitness": -10},
        {"switching_distance": 2, "split_subs": 1, "fitness": -20},
        {"switching_distance": 1, "split_subs": 2, "fitness": -15},
        {"switching_distance": 2, "split_subs": 2, "fitness": -5},
        {"switching_distance": 3, "split_subs": 2, "fitness": -30},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_metrics_3d_df():
    # Create a simple DataFrame for testing
    data = [
        {"switching_distance": 1, "split_subs": 1, "fitness": -10, "disconnections": 5},
        {"switching_distance": 2, "split_subs": 1, "fitness": -20, "disconnections": 0},
        {"switching_distance": 3, "split_subs": 1, "fitness": -10, "disconnections": 0},
        {"switching_distance": 1, "split_subs": 2, "fitness": -15, "disconnections": 3},
        {"switching_distance": 2, "split_subs": 2, "fitness": -5, "disconnections": 5},
        {"switching_distance": 3, "split_subs": 2, "fitness": -30, "disconnections": 1},
        {"switching_distance": 2, "split_subs": 3, "fitness": -9, "disconnections": 0},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def default_filter_strategy() -> FilterStrategy:
    return FilterStrategy(
        filter_dominator_metrics_target=["switching_distance", "split_subs"],
        filter_dominator_metrics_observed=["switching_distance", "split_subs"],
        filter_discriminator_metric_distances={
            "split_subs": {0.0},
            "switching_distance": {-0.9, 0.9},
            "fitness": {-60, 60},
        },
        filter_discriminator_metric_multiplier={"split_subs": 1.0},
        filter_median_metric=["split_subs"],
    )


def test_dominator_filter_basic(sample_metrics_2d_df, default_filter_strategy):
    mask = get_dominator_mask(
        sample_metrics_2d_df,
        target_metrics=default_filter_strategy.filter_dominator_metrics_target,
        observed_metrics=default_filter_strategy.filter_dominator_metrics_observed,
    )
    # Should return a boolean mask of the same length
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert len(mask) == len(sample_metrics_2d_df)
    # At least one row should be True (not dominated)
    assert mask.any()
    mask = get_dominator_mask(
        sample_metrics_2d_df,
        target_metrics=["switching_distance"],
        observed_metrics=["switching_distance"],
        fitness_col="fitness",
    )
    assert mask.any()
    observed = ["switching_distance", "fitness"]
    mask = get_dominator_mask(
        sample_metrics_2d_df, target_metrics=["switching_distance"], fitness_col="fitness", observed_metrics=observed
    )
    assert mask.any()


def test_dominator_filter_all_equal_fitness(default_filter_strategy):
    df = pd.DataFrame(
        {
            "switching_distance": [1, 1, 1],
            "split_subs": [1, 1, 1],
            "fitness": [-10, -10, -10],
        }
    )
    filter_dominator_metrics_target = default_filter_strategy.filter_dominator_metrics_target
    filter_dominator_metrics_observed = default_filter_strategy.filter_dominator_metrics_observed
    mask = get_dominator_mask(
        df, target_metrics=filter_dominator_metrics_target, observed_metrics=filter_dominator_metrics_observed
    )
    # All rows have equal fitness, so none should be dominated
    assert mask.all()


def test_dominator_filter_empty_df(default_filter_strategy):
    filter_dominator_metrics_target = default_filter_strategy.filter_dominator_metrics_target
    filter_dominator_metrics_observed = default_filter_strategy.filter_dominator_metrics_observed
    df = pd.DataFrame(columns=["switching_distance", "split_subs", "fitness"])
    mask = get_dominator_mask(
        df, target_metrics=filter_dominator_metrics_target, observed_metrics=filter_dominator_metrics_observed
    )
    assert isinstance(mask, np.ndarray)
    assert len(mask) == 0


def test_dominator_filter_single_row(default_filter_strategy):
    filter_dominator_metrics_target = default_filter_strategy.filter_dominator_metrics_target
    filter_dominator_metrics_observed = default_filter_strategy.filter_dominator_metrics_observed
    df = pd.DataFrame(
        {
            "switching_distance": [1],
            "split_subs": [1],
            "fitness": [-10],
        }
    )
    mask = get_dominator_mask(
        df, target_metrics=filter_dominator_metrics_target, observed_metrics=filter_dominator_metrics_observed
    )
    assert mask[0] == True


def test_dominator_filter_dominated_rows(default_filter_strategy):
    filter_dominator_metrics_target = default_filter_strategy.filter_dominator_metrics_target
    filter_dominator_metrics_observed = default_filter_strategy.filter_dominator_metrics_observed
    # Row 0 is dominated by row 1 (higher fitness, same split_subs)
    df = pd.DataFrame(
        {
            "switching_distance": [1, 2],
            "split_subs": [1, 1],
            "fitness": [-5, -10],
        }
    )
    mask = get_dominator_mask(
        df, target_metrics=filter_dominator_metrics_target, observed_metrics=filter_dominator_metrics_observed
    )
    # Only the row with higher fitness should remain
    assert mask.tolist() == [True, False]


def test_dominator_filter_numeric_2d(sample_metrics_2d_df, default_filter_strategy: FilterStrategy):
    filter_dominator_metrics_observed = default_filter_strategy.filter_dominator_metrics_observed
    mask = get_dominator_mask(
        sample_metrics_2d_df, target_metrics=["split_subs"], observed_metrics=filter_dominator_metrics_observed
    )

    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10}, # should be kept
    # {'switching_distance': 2, 'split_subs': 1, 'fitness': -20}, # dominated by row 0
    # {'switching_distance': 1, 'split_subs': 2, 'fitness': -15}, # should be kept, not dominated by row 0 as only split_subs is considered
    # {'switching_distance': 2, 'split_subs': 2, 'fitness': -5},  # should be kept
    # {'switching_distance': 3, 'split_subs': 2, 'fitness': -30}] # dominated by row 2 or row 3
    expected_mask = [True, False, True, True, False]
    assert all(mask == expected_mask)

    observed = ["split_subs", "fitness"]
    mask = get_dominator_mask(
        sample_metrics_2d_df, target_metrics=["split_subs"], fitness_col="fitness", observed_metrics=observed
    )

    observed = ["split_subs"]
    mask2 = get_dominator_mask(
        sample_metrics_2d_df, target_metrics=observed, fitness_col="fitness", observed_metrics=observed
    )
    assert all(mask == mask2)
    # only one observed metric and observed == target_metrics -> no effect
    expected_mask = [True, True, True, True, True]
    assert mask.tolist() == expected_mask

    observed = ["switching_distance", "fitness"]
    mask = get_dominator_mask(
        sample_metrics_2d_df, target_metrics=["switching_distance"], fitness_col="fitness", observed_metrics=observed
    )
    # only one observed metric and observed == target_metrics -> no effect
    expected_mask = [True, True, True, True, True]
    assert mask.tolist() == expected_mask

    # test with multiple target metrics

    observed = ["split_subs", "switching_distance"]
    mask = get_dominator_mask(
        sample_metrics_2d_df, target_metrics=observed, fitness_col="fitness", observed_metrics=observed
    )
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10}, # should be kept
    # {'switching_distance': 2, 'split_subs': 1, 'fitness': -20}, # dominated by row 0
    # {'switching_distance': 1, 'split_subs': 2, 'fitness': -15}, # dominated by row 0
    # {'switching_distance': 2, 'split_subs': 2, 'fitness': -5},  # should be kept
    # {'switching_distance': 3, 'split_subs': 2, 'fitness': -30}] # dominated by row 3
    expected_mask = [True, False, False, True, False]
    assert all(mask == expected_mask)

    observed = ["split_subs", "switching_distance"]
    mask = get_dominator_mask(
        sample_metrics_2d_df, target_metrics=["switching_distance"], fitness_col="fitness", observed_metrics=observed
    )

    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10}, # should be kept -> no station below 1
    # {'switching_distance': 2, 'split_subs': 1, 'fitness': -20}, # should be kept -> no station below 1
    # {'switching_distance': 1, 'split_subs': 2, 'fitness': -15}, # dominated by row 0
    # {'switching_distance': 2, 'split_subs': 2, 'fitness': -5},  # should be kept
    # {'switching_distance': 3, 'split_subs': 2, 'fitness': -30}] # should be kept -> switching_distance not present in split 1
    expected_mask = [True, True, False, True, True]
    assert mask.tolist() == expected_mask


def test_dominator_filter_numeric_3d(sample_metrics_3d_df):
    observed = ["switching_distance", "split_subs"]
    mask = get_dominator_mask(
        sample_metrics_3d_df, target_metrics=observed, fitness_col="fitness", observed_metrics=observed
    )
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 2, 'split_subs': 1, 'fitness': -20, 'disconnections': 0}, # dominated by row 0
    #  {'switching_distance': 3, 'split_subs': 1, 'fitness': -10, 'disconnections': 0}, # should be kep, as it has the same fitness as row 0
    #  {'switching_distance': 1, 'split_subs': 2, 'fitness': -15, 'disconnections': 3}, # dominated by row 0
    #   {'switching_distance': 2, 'split_subs': 2, 'fitness': -5, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 3, 'split_subs': 2, 'fitness': -30, 'disconnections': 1}, # dominated by row 4
    #  {'switching_distance': 2, 'split_subs': 3, 'fitness': -9, 'disconnections': 0}] # dominated by row 4
    expected_mask = [True, False, True, False, True, False, False]
    assert all(mask == expected_mask)

    # now it becomes a bit wired to follow in this array representation, as it is not sorted anymore in the expected way

    observed = ["switching_distance", "split_subs", "disconnections"]
    mask = get_dominator_mask(
        sample_metrics_3d_df, target_metrics=observed, fitness_col="fitness", observed_metrics=observed
    )
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 2, 'split_subs': 1, 'fitness': -20, 'disconnections': 0}, # dominated by row 0
    #  {'switching_distance': 3, 'split_subs': 1, 'fitness': -10, 'disconnections': 0}, # should be kept and not be dominated by row 6, as this has been eliminated by row 4
    #  {'switching_distance': 1, 'split_subs': 2, 'fitness': -15, 'disconnections': 3}, # dominated by row 0
    #   {'switching_distance': 2, 'split_subs': 2, 'fitness': -5, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 3, 'split_subs': 2, 'fitness': -30, 'disconnections': 1}, # dominated by row 4
    #  {'switching_distance': 2, 'split_subs': 3, 'fitness': -9, 'disconnections': 0}] # dominated by row 4
    expected_mask = [True, False, True, False, True, False, False]
    assert all(mask == expected_mask)


def test_median_filter(sample_metrics_3d_df, default_filter_strategy: FilterStrategy):
    # Test median filter with a DataFrame that has multiple metrics
    mask = get_median_mask(sample_metrics_3d_df, default_filter_strategy.filter_median_metric)
    # Note: target_metrics=["split_subs"]
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 2, 'split_subs': 1, 'fitness': -20, 'disconnections': 0}, # below median
    #  {'switching_distance': 3, 'split_subs': 1, 'fitness': -10, 'disconnections': 0}, # should be kept
    #  {'switching_distance': 1, 'split_subs': 2, 'fitness': -15, 'disconnections': 3}, # should be kept
    #   {'switching_distance': 2, 'split_subs': 2, 'fitness': -5, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 3, 'split_subs': 2, 'fitness': -30, 'disconnections': 1}, # below median
    #  {'switching_distance': 2, 'split_subs': 3, 'fitness': -9, 'disconnections': 0}] # should be kept
    expected_mask = [True, False, True, True, True, False, True]
    assert all(mask == expected_mask)

    mask = get_median_mask(sample_metrics_3d_df, target_metrics=["split_subs", "switching_distance", "disconnections"])
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10, 'disconnections': 5}, # below median - disconnections
    #  {'switching_distance': 2, 'split_subs': 1, 'fitness': -20, 'disconnections': 0}, # below median - split_subs & switching_distance
    #  {'switching_distance': 3, 'split_subs': 1, 'fitness': -10, 'disconnections': 0}, # should be kept
    #  {'switching_distance': 1, 'split_subs': 2, 'fitness': -15, 'disconnections': 3}, # below median - switching_distance
    #   {'switching_distance': 2, 'split_subs': 2, 'fitness': -5, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 3, 'split_subs': 2, 'fitness': -30, 'disconnections': 1}, # below median - split_subs & switching_distance
    #  {'switching_distance': 2, 'split_subs': 3, 'fitness': -9, 'disconnections': 0}] # should be kept
    expected_mask = [False, False, True, False, True, False, True]
    assert all(mask == expected_mask)

    sample_metrics_3d_df["disconnections"] = -sample_metrics_3d_df["disconnections"]  # make it an artificial fitness metric
    mask = get_median_mask(sample_metrics_3d_df, target_metrics=["split_subs"], fitness_col="disconnections")
    sample_metrics_3d_df["disconnections"] = -sample_metrics_3d_df["disconnections"]  # restore original values
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10, 'disconnections': 5}, # below median - disconnections fitness
    #  {'switching_distance': 2, 'split_subs': 1, 'fitness': -20, 'disconnections': 0}, # should be kept
    #  {'switching_distance': 3, 'split_subs': 1, 'fitness': -10, 'disconnections': 0}, # should be kept
    #  {'switching_distance': 1, 'split_subs': 2, 'fitness': -15, 'disconnections': 3}, # should be kept
    #   {'switching_distance': 2, 'split_subs': 2, 'fitness': -5, 'disconnections': 5}, # below median - disconnections fitness
    #  {'switching_distance': 3, 'split_subs': 2, 'fitness': -30, 'disconnections': 1}, # should be kept
    #  {'switching_distance': 2, 'split_subs': 3, 'fitness': -9, 'disconnections': 0}] # should be kept
    expected_mask = [False, True, True, True, False, True, True]
    assert all(mask == expected_mask)


def test_discriminator_filter_2d(sample_metrics_2d_df):
    observed = ["split_subs", "switching_distance"]
    mask = get_dominator_mask(
        sample_metrics_2d_df, target_metrics=observed, fitness_col="fitness", observed_metrics=observed
    )
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10}, # should be kept
    # {'switching_distance': 2, 'split_subs': 1, 'fitness': -20}, # dominated by row 0
    # {'switching_distance': 1, 'split_subs': 2, 'fitness': -15}, # dominated by row 0
    # {'switching_distance': 2, 'split_subs': 2, 'fitness': -5},  # should be kept
    # {'switching_distance': 3, 'split_subs': 2, 'fitness': -30}] # dominated by row 3
    expected_mask = [True, False, False, True, False]
    assert all(mask == expected_mask)

    discriminator_df = sample_metrics_2d_df[mask]
    metric_distances = {
        "split_subs": {0},
        "switching_distance": {-0.9, 0.9},
        "fitness": {-0.1, 0.1},
    }

    assert all(mask == expected_mask)
    mask = get_discriminator_mask(
        sample_metrics_2d_df,
        discriminator_df=discriminator_df,
        metric_distances=metric_distances,
    )
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10}, # discriminated by discriminator_df
    # {'switching_distance': 2, 'split_subs': 1, 'fitness': -20}, # should be kept
    # {'switching_distance': 1, 'split_subs': 2, 'fitness': -15}, # should be kept
    # {'switching_distance': 2, 'split_subs': 2, 'fitness': -5},  # discriminated by discriminator_df
    # {'switching_distance': 3, 'split_subs': 2, 'fitness': -30}] # should be kept
    expected_mask = [False, True, True, False, True]
    assert all(mask == expected_mask)

    metric_distances = {
        "split_subs": {0},
        "switching_distance": {-0.9, 0.9},
        "fitness": {-1.0, 1.0},  # Note, it is 100%
    }
    mask = get_discriminator_mask(
        sample_metrics_2d_df,
        discriminator_df=discriminator_df,
        metric_distances=metric_distances,
        metric_multiplier={"split_subs": 1.0},
    )
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10}, # discriminated by discriminator_df
    # {'switching_distance': 2, 'split_subs': 1, 'fitness': -20}, # should be kept
    # {'switching_distance': 1, 'split_subs': 2, 'fitness': -15}, # discriminated by split_subs = 2 -> switching_distance > 1 & fitness <= -15
    # {'switching_distance': 2, 'split_subs': 2, 'fitness': -5},  # discriminated by discriminator_df
    # {'switching_distance': 3, 'split_subs': 2, 'fitness': -30}] # should be kept
    expected_mask = [False, True, False, False, True]
    assert all(mask == expected_mask)

    discriminator_df = sample_metrics_2d_df.copy()
    discriminator_df.loc[:, "fitness"] = discriminator_df["fitness"] - 40
    metric_distances = {
        "split_subs": {0},
        "switching_distance": {-1, 1},
        "fitness": {-0.1, 0.1},
    }
    mask = get_discriminator_mask(
        sample_metrics_2d_df,
        discriminator_df=discriminator_df,
        metric_distances=metric_distances,
    )
    # all should be kept, as the discriminator_df fitness is lower than the original fitness considering the metric_distances
    expected_mask = [True, True, True, True, True]
    assert all(mask == expected_mask)

    metric_distances = {
        "split_subs": {0},
        "switching_distance": {-1, 1},
        "fitness": {-0.2, 0.2},
    }
    mask = get_discriminator_mask(
        sample_metrics_2d_df,
        discriminator_df=discriminator_df,
        metric_distances=metric_distances,
        metric_multiplier={"split_subs": 1.0},
    )
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10}, # should be kept
    # {'switching_distance': 2, 'split_subs': 1, 'fitness': -20}, # should be kept
    # {'switching_distance': 1, 'split_subs': 2, 'fitness': -15}, # should be kept
    # {'switching_distance': 2, 'split_subs': 2, 'fitness': -5},  # should be kept
    # {'switching_distance': 3, 'split_subs': 2, 'fitness': -30}] # get discriminated by split_subs = 2 -> switching_distance > 2 & fitness -5 -40 < -30 - 10*2
    expected_mask = [True, True, True, True, False]
    assert all(mask == expected_mask)

    # test raise ValueError if not all metric distances are present in the discriminator DataFrame columns
    with pytest.raises(ValueError):
        mask = get_discriminator_mask(
            sample_metrics_2d_df,
            discriminator_df=discriminator_df,
            metric_distances={"split_subs": {0}, "switching_distance": {-1, 1}, "fitness": {-10, 10}, "unknown_metric": {0}},
        )


def test_repertoire_selection_mask(sample_metrics_3d_df, default_filter_strategy: FilterStrategy):
    # Test repertoire selection mask with a DataFrame that has multiple metrics
    target_metrics = ["switching_distance", "split_subs", "disconnections"]
    default_filter_strategy.filter_dominator_metrics_target = target_metrics
    discriminator_df = pd.DataFrame()
    mask = get_repertoire_filter_mask(
        sample_metrics_3d_df, discriminator_df=discriminator_df, filter_strategy=default_filter_strategy
    )
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 2, 'split_subs': 1, 'fitness': -20, 'disconnections': 0}, # median filter
    #  {'switching_distance': 3, 'split_subs': 1, 'fitness': -10, 'disconnections': 0}, # should be kept
    #  {'switching_distance': 1, 'split_subs': 2, 'fitness': -15, 'disconnections': 3}, # dominated by row 0
    #   {'switching_distance': 2, 'split_subs': 2, 'fitness': -5, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 3, 'split_subs': 2, 'fitness': -30, 'disconnections': 1}, # median filter
    #  {'switching_distance': 2, 'split_subs': 3, 'fitness': -9, 'disconnections': 0}] # dominated by row 4
    expected_mask = [True, False, True, False, True, False, False]
    assert all(mask == expected_mask)

    discriminator_df = pd.DataFrame([{"switching_distance": 1, "split_subs": 1, "fitness": -10, "disconnections": 5}])
    mask = get_repertoire_filter_mask(
        sample_metrics_3d_df, discriminator_df=discriminator_df, filter_strategy=default_filter_strategy
    )
    # [{'switching_distance': 1, 'split_subs': 1, 'fitness': -10, 'disconnections': 5}, # discriminated by discriminator_df
    #  {'switching_distance': 2, 'split_subs': 1, 'fitness': -20, 'disconnections': 0}, # median filter (applied as the first filter)
    #  {'switching_distance': 3, 'split_subs': 1, 'fitness': -10, 'disconnections': 0}, # should be kept
    #  {'switching_distance': 1, 'split_subs': 2, 'fitness': -15, 'disconnections': 3}, # should be kept, as not dominated by row 0 anymore
    #   {'switching_distance': 2, 'split_subs': 2, 'fitness': -5, 'disconnections': 5}, # should be kept
    #  {'switching_distance': 3, 'split_subs': 2, 'fitness': -30, 'disconnections': 1}, # median filter
    #  {'switching_distance': 2, 'split_subs': 3, 'fitness': -9, 'disconnections': 0}] # dominated by row 4
    expected_mask = [False, False, True, True, True, False, False]
    assert all(mask == expected_mask)


def test_subtract_repertoire_selection_fitness(sample_metrics_3d_df: pd.DataFrame, default_filter_strategy: FilterStrategy):
    # Test subtract_repertoire_selection_fitness with a DataFrame that has multiple metrics
    target_metrics = ["switching_distance", "split_subs", "disconnections"]
    discriminator_df = pd.DataFrame()
    metrics_df_substract = filter_metrics_df(
        metrics_df=sample_metrics_3d_df,
        discriminator_df=discriminator_df,
        filter_strategy=default_filter_strategy,
    )
    assert all(metrics_df_substract.columns == sample_metrics_3d_df.columns)

    # at least one value should be less than the subtract_value
    assert len(sample_metrics_3d_df) > len(metrics_df_substract)


def test_select_stategy_ac_dc_mix_filter_applied(
    dc_repertoire: list[BaseDBTopology], session: Session, default_filter_strategy: FilterStrategy
) -> None:
    # Copy the dc strategies to the AC database (as if they were pulled before)
    mixed_topologies = []
    for topology in dc_repertoire:
        metrics = topology.metrics
        metrics["fitness_dc"] = topology.fitness
        new_topo = ACOptimTopology(
            actions=topology.actions,
            disconnections=topology.disconnections,
            pst_setpoints=topology.pst_setpoints,
            unsplit=topology.unsplit,
            timestep=topology.timestep,
            strategy_hash=topology.strategy_hash,
            optimization_id=topology.optimization_id,
            optimizer_type=OptimizerType.AC,
            fitness=topology.fitness,
            metrics=topology.metrics,
        )
        session.add(new_topo)
        session.commit()
        session.refresh(new_topo)
        mixed_topologies.append(new_topo)
        mixed_topologies.append(topology)

    assert len(dc_repertoire) * 2 == len(mixed_topologies), "Not all DC topologies were copied to AC storage"

    # Select a strategy
    strategy = select_strategy(
        np.random.default_rng(0), mixed_topologies, default_scorer, filter_strategy=default_filter_strategy
    )
    assert isinstance(strategy, list)
    assert len(strategy)
    assert isinstance(strategy[0], ACOptimTopology)
    assert len(set(t.strategy_hash for t in strategy)) == 1
    assert len(set(t.optimizer_type for t in strategy)) == 1


def test_get_discriminator_df_basic():
    # Create a sample metrics_df
    data = {
        "optimizer_type": [OptimizerType.AC.value, OptimizerType.AC.value, OptimizerType.AC.value],
        "fitness_dc": [0.1, 0.2, 0.3],
        "metric1": [10, 20, 30],
        "metric2": [100, 200, 300],
    }
    metrics_df = pd.DataFrame(data)
    target_metrics = ["metric1", "metric2"]

    result = get_discriminator_df(metrics_df, target_metrics)

    # Only rows with optimizer_type == AC should be present
    assert len(result) == 3
    # Columns should be target_metrics + "fitness"
    assert set(result.columns) == set(target_metrics + ["fitness"])
    # Values should match original AC rows
    assert result["metric1"].tolist() == [10, 20, 30]
    assert result["metric2"].tolist() == [100, 200, 300]
    assert result["fitness"].tolist() == [0.1, 0.2, 0.3]


def test_get_discriminator_df_empty_metrics_df():
    metrics_df = pd.DataFrame(columns=["optimizer_type", "fitness_dc", "metric1"])
    target_metrics = ["metric1"]
    result = get_discriminator_df(metrics_df, target_metrics)
    assert result.empty

    metrics_df = pd.DataFrame()
    result = get_discriminator_df(metrics_df, target_metrics)
    assert result.empty
