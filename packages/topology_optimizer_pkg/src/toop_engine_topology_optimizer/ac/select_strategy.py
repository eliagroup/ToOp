# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Selection strategy for AC optimization topologies."""

import logbook
import numpy as np
import pandas as pd
from beartype.typing import Callable, Optional, Tuple, Union
from numpy.random import Generator as Rng
from toop_engine_interfaces.types import MetricType
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import FilterStrategy, Fitness, OptimizerType
from toop_engine_topology_optimizer.interfaces.models.base_storage import (
    BaseDBTopology,
    metrics_dataframe,
)

logger = logbook.Logger(__name__)


def select_strategy(
    rng: Rng,
    repertoire: list[BaseDBTopology],
    candidates: list[BaseDBTopology],  # noqa: ARG001
    interest_scorer: Callable[[pd.DataFrame], pd.Series],
    filter_strategy: Optional[FilterStrategy] = None,
) -> Union[list[ACOptimTopology], Tuple[list[ACOptimTopology], list[ACOptimTopology]]]:
    """Select a promising strategy from the repertoire

    Make sure the repertoire only contains topologies with the right optimizer type and optimization id
    as select_topology will not filter for this.

    Parameters
    ----------
    rng : Rng
        The random number generator to use
    repertoire : list[BaseDBTopology]
        The filtered repertoire with all individuals of the optimization in all optimizer types
    candidates : list[BaseDBTopology]
        Candidates which have not yet been evaluated. For a pull operation this will only include DC candidates without an
        AC parent.
    interest_scorer : Callable[[pd.DataFrame], pd.Series]
        The function to score the topologies in the repertoire. The higher the score, the more
        interesting the topology is. Eventually, the topology will be selected with a probability
        proportional to its score.
    filter_strategy : Optional[FilterStrategy], optional
        Whether to filter the repertoire based on discriminator, median or dominator filter.

    Returns
    -------
    Union[list[ACOptimTopology], Tuple[list[ACOptimTopology], list[ACOptimTopology]]]
        The selected strategy which is represented as a list of topologies with similar strategy_hash and
        optimizer type..
        If two is True, a tuple of two lists is returned with two different strategy_hashes.
        If no strategy could be selected because the repertoire wasn't containing enough strategies,
        return an empty list
    """
    if len(repertoire) == 0:
        return []
    # Extract only the metrics in a nice format
    metrics = metrics_dataframe(repertoire)

    if filter_strategy is not None:
        if filter_strategy.filter_dominator_metrics_target is not None:
            discriminator_df = get_discriminator_df(
                metrics[metrics["optimizer_type"] == OptimizerType.AC.value], filter_strategy.filter_dominator_metrics_target
            )
        else:
            discriminator_df = pd.DataFrame()
        metrics = filter_metrics_df(
            metrics_df=metrics[metrics["optimizer_type"] == OptimizerType.DC.value],
            discriminator_df=discriminator_df,
            filter_strategy=filter_strategy,
        )
    else:
        metrics = metrics[metrics["optimizer_type"] == OptimizerType.DC.value]

    # Score them according to some function
    metrics["score"] = interest_scorer(metrics)
    group = metrics.groupby(["strategy_hash", "optimizer_type"])
    if len(group.size()) == 0:
        return []

    strategies = group.sum("score")
    sum_scores = strategies.score.sum()
    if not np.isclose(sum_scores, 0):
        strategies.score /= sum_scores
    else:
        strategies.score = 1 / len(strategies)

    # Select a strategy with probability proportional to its score
    idx = rng.choice(len(strategies), p=strategies.score)
    hash_, optim_type_ = strategies.index[idx]
    return [t for t in repertoire if t.strategy_hash == hash_ and t.optimizer_type.value == optim_type_]


def filter_metrics_df(
    metrics_df: pd.DataFrame,
    discriminator_df: pd.DataFrame,
    filter_strategy: FilterStrategy,
) -> np.ndarray:
    """Get a mask for the metrics DataFrame that filters out rows based on discriminator and median masks.

    This function applies a discriminator, median and dominator mask.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The DataFrame containing the metrics to filter. This is typically the DC repertoire from which new results shall
        be pulled.
    discriminator_df : pd.DataFrame
        The DataFrame containing the discriminator metrics. These are topologies that have previously been AC validated
    filter_strategy : FilterStrategy
        The filter strategy to use for the optimization,
        used to filter out strategies based on the discriminator, median or dominator filter.

    Returns
    -------
    pd.DataFrame
        The filtered metrics DataFrame with similar topologies removed.
        If all topologies are filtered out, the original metrics DataFrame is returned.
    """
    repertoire_mask = get_repertoire_filter_mask(
        metrics_df=metrics_df,
        discriminator_df=discriminator_df,
        filter_strategy=filter_strategy,
    )
    # make sure that the metrics_df is not empty after filtering
    if len(metrics_df[repertoire_mask]) != 0:
        metrics_df = metrics_df[repertoire_mask]
    return metrics_df


def get_repertoire_filter_mask(
    metrics_df: pd.DataFrame,
    discriminator_df: pd.DataFrame,
    filter_strategy: FilterStrategy,
) -> np.ndarray:
    """Get a mask for the metrics DataFrame that filters out rows based on discriminator and median masks.

    This function applies a discriminator, median and dominator mask.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The DataFrame containing the metrics to filter.
    discriminator_df : pd.DataFrame
        The DataFrame containing the discriminator metrics.
    filter_strategy : FilterStrategy
        The filter strategy to use for the optimization,
        used to filter out strategies based on the discriminator, median or dominator filter.

    Returns
    -------
    Bool[np.ndarray, " metrics_df.shape[0]"]
        A boolean mask where True indicates the row is not filtered out.
    """
    # set the dominator metrics observed if not provided
    if (
        filter_strategy.filter_dominator_metrics_observed is None
        and filter_strategy.filter_dominator_metrics_target is not None
    ):
        # set a to target
        filter_strategy.filter_dominator_metrics_observed = filter_strategy.filter_dominator_metrics_target

    # get the discriminator filter mask if the config is provided
    if (
        not discriminator_df.empty
        and filter_strategy.filter_discriminator_metric_distances is not None
        and filter_strategy.filter_discriminator_metric_multiplier is not None
    ):
        discriminator_filter_mask = get_discriminator_mask(
            metrics_df=metrics_df,
            discriminator_df=discriminator_df,
            metric_distances=filter_strategy.filter_discriminator_metric_distances,
            metric_multiplier=filter_strategy.filter_discriminator_metric_multiplier,
        )
    else:
        discriminator_filter_mask = np.ones(len(metrics_df), dtype=bool)

    # get the median filter mask if the config is provided
    if filter_strategy.filter_median_metric is not None:
        median_filter_mask = get_median_mask(
            metrics_df=metrics_df,
            target_metrics=filter_strategy.filter_median_metric,
        )
    else:
        median_filter_mask = np.ones(len(metrics_df), dtype=bool)

    # apply the discriminator and median filter masks
    metrics_df_filtered = metrics_df[discriminator_filter_mask & median_filter_mask]

    # get the dominator filter mask if the config is provided
    if (
        filter_strategy.filter_dominator_metrics_target is not None
        and filter_strategy.filter_dominator_metrics_observed is not None
    ):
        dominator_filter_mask = get_dominator_mask(
            metrics_df=metrics_df_filtered,
            target_metrics=filter_strategy.filter_dominator_metrics_target,
            observed_metrics=filter_strategy.filter_dominator_metrics_observed,
        )
    else:
        dominator_filter_mask = np.ones(len(metrics_df_filtered), dtype=bool)

    # apply the dominator filter mask
    filtered_index = metrics_df_filtered[dominator_filter_mask].index

    return np.isin(metrics_df.index, filtered_index)


def get_median_mask(
    metrics_df: pd.DataFrame, target_metrics: list[MetricType], fitness_col: Optional[Fitness] = "fitness"
) -> np.ndarray:
    """Get a mask for fitness values below the median for each discrete value of the target metrics.

    Note: expects the target metrics to be discrete values, not continuous.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The DataFrame containing the metrics to filter.
    target_metrics : list[MetricType]
        A list of metrics with discrete values to consider for filtering.
        example: ["split_subs"].
    fitness_col : Optional[str], optional
        The column name that contains the fitness values. Defaults to "fitness".

    Returns
    -------
    filter_mask : Bool[np.ndarray, " metrics_df.shape[0]"]
        A boolean mask where True indicates the row is not below the median for any of the target metrics.
    """
    filter_mask = np.zeros(len(metrics_df), dtype=bool)
    for target_metric in target_metrics:
        for discrete_value in metrics_df[target_metric].unique():
            col_mask = (metrics_df[target_metric] == discrete_value).values
            metric_df = metrics_df[col_mask][fitness_col]
            # remove median
            median_fitness = metric_df.median()
            filter_mask |= col_mask & (metrics_df[fitness_col] < median_fitness).values
    # return the filter mask, that removes rows below the median
    filter_mask = ~filter_mask
    return filter_mask


def get_dominator_mask(
    metrics_df: pd.DataFrame,
    target_metrics: list[MetricType],
    observed_metrics: list[MetricType],
    fitness_col: Optional[Fitness] = "fitness",
) -> np.ndarray:
    """Get a mask for rows from a DataFrame that are dominated by other rows based on specified metrics.

    A metric entry if there is any other metric entry with a better fitness,
    in respect to the distance to the original topology.
    The distance in measured by the metric, assuming that lower values are better.

    The target metric is used to fix the discrete value for which the dominance is checked.
    The fitness column is used to determine the fitness of the rows.
    Each observed metric is checked against the minimum fitness of all discrete target values.

    Intended use is target_metrics = ["switching_distance", "split_subs"]
    and observed_metrics = Any additional metric to the target metrics, e.g. "disconnections"


    Parameters
    ----------
    metrics_df : pd.DataFrame
        The DataFrame to filter.
    target_metrics : list[MetricType]
        A list of metrics to consider for dominance.
        A target metric is expected to have discrete values (e.g. not fitness, overload_energy, or max_flow)
        If None, defaults to ["switching_distance", "split_subs"].
    observed_metrics : list[MetricType]
        A list of metrics to observe for dominance.
        If None, defaults to ["switching_distance", "split_subs"].
    fitness_col : Optional[str], optional
        The column name that contains the fitness values. Defaults to "fitness".
        Note: the values are expected to be negative, best fitness converges to zero.

    Returns
    -------
    filter_mask : Bool[np.ndarray, " metrics_df.shape[0]"]
        A boolean mask where True indicates the row is not dominated by another row.

    """
    filter_mask = np.zeros(len(metrics_df), dtype=bool)
    for target_metric in target_metrics:
        for discrete_value in metrics_df[target_metric].unique():
            # get columns mask
            col_mask = (metrics_df[target_metric] == discrete_value).values
            if (col_mask & ~filter_mask).sum() == 0:
                # all the elements have been filtered out already
                # -> no element is dominated by an already eliminated element
                continue
            max_idx = metrics_df[col_mask & ~filter_mask][fitness_col].idxmax()

            # get fitness mask
            fitness_mask = (metrics_df[fitness_col] < metrics_df[fitness_col].loc[max_idx]).values

            # apply dominator condition
            for col in observed_metrics:
                if col == fitness_col:
                    continue
                filter_mask |= (metrics_df[col] > metrics_df[col].loc[max_idx]).values & fitness_mask & col_mask

    # retrun the filter mask, that removes dominated rows
    filter_mask = ~filter_mask

    return filter_mask


def get_discriminator_mask(
    metrics_df: pd.DataFrame,
    discriminator_df: pd.DataFrame,
    metric_distances: dict[str, set[float]],
    metric_multiplier: Optional[dict[str, set[float]]] = None,
) -> np.ndarray:
    """Get a mask for rows in metrics_df that are within a certain distance from the discriminator_df.

    The distance is defined by the metric_distances dictionary, which contains the metrics and their respective distances.
    Use the `use_split_sub_multiplier` flag to apply a multiplier in respect to the split_subs_col.
    e.g. use_split_sub_multiplier=False, the metric_distances is applied directly to the metrics_df.
    If use_split_sub_multiplier=True, the metric_distances are multiplied by the split_subs, leading to a
    larger distance for larger split_subs values.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The DataFrame containing the metrics to filter.
    discriminator_df : pd.DataFrame
        The DataFrame containing the discriminator metrics.
    metric_distances : dict[str, set[float]]
        A dictionary defining the metric distances for filtering.
        The keys are metric names and the values are sets of distances.
        example:
        metric_distances = {
            "split_subs": {0},
            "switching_distance": {-0.9, 0.9},
            "fitness": {-0.1, 0.1},
        }
        Note: the fitness is treated as a percentage.
    metric_multiplier : Optional[dict[str, set[float]]], optional
        A dictionary defining multiplier for the metric distances.
        The keys are metric names and the values are sets of distances.
        If None, defaults to an empty dictionary.
        Multiple values are added by:
        distance_multiplier = (
           `metric_multiplier[metric1]` * `discriminator_df[metric1]` +
           `metric_multiplier[metric2]` * `discriminator_df[metric2]` + ...
        )
        example: {"split_subs": 0.5}
                 The discriminator_df will be multiplied by the split_subs value.
                 In the case of the metric_distances values will be multiplied by the
                 split_subs value and the metric_multiplier.
                 -> metric_distances["switching_distance"] * 0.5 * split_subs_col (e.g. 4 splits)
                 -> the metric distance is increased by this metric multiplier.


    Returns
    -------
    filter_mask : Bool[np.ndarray, " metrics_df.shape[0]"]
        A boolean mask where True indicates the row is not within the distance defined by the discriminator_df
        and metric_distances.

    Raises
    ------
    ValueError
        If not all metric distances are present in the discriminator DataFrame columns.
        If the metric_distances is None, it will use default values based on the use_split_sub_multiplier flag.
    """
    if metric_multiplier is None:
        metric_multiplier = {}

    if not set(metric_distances.keys()).issubset(discriminator_df.columns):
        raise ValueError(
            f"Not all metric distances {metric_distances.keys()} are "
            f"present in the discriminator DataFrame {discriminator_df.columns}. "
        )
    if not set(metric_multiplier.keys()).issubset(discriminator_df.columns):
        raise ValueError(
            f"Not all metric multipliers {metric_multiplier.keys()} are "
            f"present in the discriminator DataFrame {discriminator_df.columns}. "
        )

    discriminator_df["fitness_save"] = discriminator_df["fitness"].copy()  # save original fitness for later use
    discriminator_df["fitness"] = discriminator_df["fitness"].abs()  # ensure fitness is
    metrics_df["fitness_save"] = metrics_df["fitness"].copy()  # save original fitness for later use
    metrics_df["fitness"] = metrics_df["fitness"].abs()  # ensure fitness is positive for the discriminator

    # filter mask for discriminator
    filter_mask = np.zeros(len(metrics_df), dtype=bool)
    for _idx, row in discriminator_df.iterrows():
        mask_metrics = np.ones(len(metrics_df), dtype=bool)
        for metric, distance in metric_distances.items():
            # get distance multiplier
            distance_multiplier = 0
            for metric_multiplier_key, metric_multiplier_value in metric_multiplier.items():
                distance_multiplier += metric_multiplier_value * row[metric_multiplier_key]
            if distance_multiplier == 0:
                # if no multiplier is defined, fall back to 1
                distance_multiplier = 1

            # apply the distance to the metrics_df
            if metric != "fitness":
                min_condition = row[metric] + min(distance) * distance_multiplier
                max_condition = row[metric] + max(distance) * distance_multiplier

            else:
                min_condition = row[metric] * (1 + min(distance) * distance_multiplier)
                max_condition = row[metric] * (1 + max(distance) * distance_multiplier)

            mask_metrics = mask_metrics & (metrics_df[metric] >= min_condition) & (metrics_df[metric] <= max_condition)

        filter_mask += mask_metrics

    # restore original fitness
    metrics_df["fitness"] = metrics_df["fitness_save"]
    metrics_df.drop(columns=["fitness_save"], inplace=True)

    # return the filter mask, that removes discriminated rows
    filter_mask = ~filter_mask

    return filter_mask


def get_discriminator_df(metrics_df: pd.DataFrame, target_metrics: list[str]) -> pd.DataFrame:
    """Get a discriminator DataFrame from the metrics DataFrame.

    The discriminator DataFrame is a subset of the metrics DataFrame that contains only the target metrics.
    It is used to filter out similar topologies from the metrics DataFrame.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The DataFrame containing the metrics to filter.
        Note: expects the metrics_df to contain only AC topologies
        that have as a metric the "fitness_dc" column.
    target_metrics : list[str]
        A list of metrics to consider for filtering.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the target metrics.
    """
    if metrics_df.empty:
        return pd.DataFrame()
    discriminator_df = metrics_df[[*target_metrics, "fitness_dc"]]
    discriminator_df.rename(columns={"fitness_dc": "fitness"}, inplace=True)
    return discriminator_df
