# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Detect cascade triggers caused by current overloads."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandera.typing as pat
from toop_engine_contingency_analysis.pandapower.cascade.configuration import CascadeConfig
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import TRANSFORMER_TABLES
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_interfaces.loadflow_results import BranchResultSchema

_logger = logging.getLogger(__name__)

#: Contingency id used for the N-0 results.
BASECASE_CONTINGENCY_ID = "BASECASE"


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


def resolve_loading_thresholds(
    current_res: pd.DataFrame,
    cascade_configuration: CascadeConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Resolve the loading threshold that applies to each branch result row.

    The threshold of a row follows from the element type (line, transformer, or
    anything else) encoded in its globally unique ``element`` id, and from whether
    the row belongs to the base case or to a contingency.

    Parameters
    ----------
    current_res : pd.DataFrame
        Branch result table with ``element`` and ``contingency`` columns, as
        produced by :func:`prepare_branch_results_for_overload`.
    cascade_configuration : CascadeConfig
        Cascade settings holding the scalar and the per-type/per-case thresholds.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        Per-row thresholds aligned to ``current_res``, and the same thresholds keyed
        by ``"<element type>/<case>"`` for logging.
    """
    # Globally unique ids are ``f"{table_id}{SEPARATOR}{table}"``; ids without a separator
    # (or missing ones) resolve to no known table and therefore to the scalar threshold.
    element_tables = current_res["element"].astype(str).str.rsplit(SEPARATOR, n=1).str[-1].to_numpy()
    is_basecase = (current_res["contingency"] == BASECASE_CONTINGENCY_ID).to_numpy()

    is_line = element_tables == "line"
    # Both transformer tables resolve to the same threshold, so "trafo" stands for both.
    is_transformer = np.isin(element_tables, list(TRANSFORMER_TABLES))

    line_basecase = cascade_configuration.overload.threshold("line", basecase=True)
    line_contingency = cascade_configuration.overload.threshold("line", basecase=False)
    transformer_basecase = cascade_configuration.overload.threshold("trafo", basecase=True)
    transformer_contingency = cascade_configuration.overload.threshold("trafo", basecase=False)
    other = cascade_configuration.overload.current_loading_threshold

    # Every row starts at the scalar threshold; lines and transformers overwrite it.
    thresholds = np.full(len(current_res), other, dtype=float)
    thresholds[is_line & is_basecase] = line_basecase
    thresholds[is_line & ~is_basecase] = line_contingency
    thresholds[is_transformer & is_basecase] = transformer_basecase
    thresholds[is_transformer & ~is_basecase] = transformer_contingency

    thresholds_by_category = {
        "line/basecase": line_basecase,
        "line/contingency": line_contingency,
        "transformer/basecase": transformer_basecase,
        "transformer/contingency": transformer_contingency,
        "other": other,
    }
    return thresholds, thresholds_by_category


def evaluate_overload_triggers(
    current_res: Optional[pat.DataFrame[BranchResultSchema]],
    cascade_configuration: CascadeConfig,
) -> pd.DataFrame:
    """Find branches whose loading is above the threshold configured for them.

    Each row is compared against the threshold resolved for its element type and
    case by :func:`resolve_loading_thresholds`. The comparison is strict, so a
    branch loaded exactly at its threshold does not trigger a cascade.

    Parameters
    ----------
    current_res : pat.DataFrame[BranchResultSchema] or None
        Branch result table with a loading column, where ``loading`` is the
        per-unit ratio ``i / i_max`` rather than a percentage.
    cascade_configuration : CascadeConfig
        Cascade settings holding the loading thresholds.

    Returns
    -------
    pd.DataFrame
        Rows from current_res whose loading is greater than their threshold.
    """
    if current_res is None or current_res.empty:
        return pd.DataFrame()

    max_loading = float(current_res["loading"].max())
    _logger.info("Maximum calculated loading: %s", max_loading)

    thresholds, applied_thresholds = resolve_loading_thresholds(current_res, cascade_configuration)
    overloaded = current_res[current_res["loading"].to_numpy() > thresholds]
    if overloaded.empty:
        _logger.info("cascading: No cascade records found with thresholds %s", applied_thresholds)
    else:
        _logger.info("cascading: %s cascade records found with thresholds %s", len(overloaded), applied_thresholds)
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
