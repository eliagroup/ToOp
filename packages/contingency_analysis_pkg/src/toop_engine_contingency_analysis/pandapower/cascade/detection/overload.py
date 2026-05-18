# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Detect cascade triggers caused by current overloads."""

import logging
from typing import Optional

import pandas as pd
import pandera.typing as pat
from toop_engine_interfaces.loadflow_results import BranchResultSchema

_logger = logging.getLogger(__name__)


def prepare_branch_results_for_overload(current_res: Optional[pat.DataFrame[BranchResultSchema]]) -> pd.DataFrame:
    """Prepare branch results for current-overload checks.

    Parameters
    ----------
    current_res : pat.DataFrame[BranchResultSchema] or None
        Branch result table, or None when no results are available.

    Returns
    -------
    pd.DataFrame
        Branch result table with index values available as normal columns.
    """
    if current_res is None:
        return pd.DataFrame()
    return current_res.reset_index()


def evaluate_overload_triggers(
    current_res: Optional[pat.DataFrame[BranchResultSchema]],
    threshold: float,
) -> pd.DataFrame:
    """Find branches whose loading is above the configured threshold.

    Parameters
    ----------
    current_res : pat.DataFrame[BranchResultSchema] or None
        Branch result table with a loading column.
    threshold : float
        Loading value above which a branch is treated as overloaded.

    Returns
    -------
    pd.DataFrame
        Rows from current_res whose loading is greater than threshold.
    """
    if current_res is None or current_res.empty:
        return pd.DataFrame()

    max_loading = float(current_res["loading"].max())
    _logger.info("Maximum calculated loading: %s", max_loading)

    overloaded = current_res[current_res["loading"] > threshold]
    if overloaded.empty:
        _logger.info("cascading: No cascade records found with %s threshold", threshold)
    else:
        _logger.info("cascading: %s cascade records found", len(overloaded))
    return overloaded


def pick_highest_loading_row(df: pd.DataFrame) -> pd.Series:
    """Pick the most heavily loaded row from a branch result table.

    Parameters
    ----------
    df : pd.DataFrame
        Table with a loading column.

    Returns
    -------
    pd.Series
        Row with the largest loading value.
    """
    return df.loc[df["loading"].idxmax()]
