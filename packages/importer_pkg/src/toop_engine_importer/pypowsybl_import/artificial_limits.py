# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Create the artificial optimization limits of the monitored branches and measure the unsplit grid against them.

The limits are derived from the worst currents of a security analysis, the branch role (optimized or non-worsening)
and the double limits. See docs/importer/pypowsybl/artificial_limits.md.
"""

from copy import deepcopy

import pandas as pd
import polars as pl
import pypowsybl
from beartype.typing import Optional
from pypowsybl.network.impl.network import Network
from toop_engine_contingency_analysis.compute_metrics import compute_metrics
from toop_engine_contingency_analysis.pypowsybl import get_monitored_branch_limits_polars, run_contingency_analysis_powsybl
from toop_engine_importer.pypowsybl_import.loadflow_based_current_limits import (
    create_current_limits_df,
    get_branches_including_limits_and_dangling_lines,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_worst_case_branch_results_polars
from toop_engine_interfaces.loadflow_results import BranchSide
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.preprocess.preprocess_commands import DoubleLimits
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.types import LimitCase, MetricType, OperationalLimit

ARTIFICIAL_LIMITS = (OperationalLimit.ARTIFICIAL_N_0, OperationalLimit.ARTIFICIAL_N_1)
ARTIFICIAL_DC_LIMITS = (OperationalLimit.ARTIFICIAL_DC_N_0, OperationalLimit.ARTIFICIAL_DC_N_1)


def compute_artificial_limits(
    flow: pd.Series,
    limit: pd.Series,
    optimized: pd.Series,
    non_worsening: pd.Series,
    double_limits: DoubleLimits,
) -> pd.Series:
    """Compute the optimization limit of every branch and side from its worst flow.

    - Branches that are optimized and non-worsening ignore the lower and upper limit.
      They get the physical limit if overloaded, otherwise their flow plus ``non_worsening_margin`` (capped at
      the physical limit), at least ``minimum_lower_physical_non_worsening_limit`` times the physical limit.
    - Optimized branches above the upper limit get the upper limit.
    - Non-worsening branches get their flow plus ``non_worsening_margin``, at least
      ``minimum_lower_physical_non_worsening_limit`` times the physical limit.
    - Other optimized branches get the larger of the lower limit and their flow.

    Parameters
    ----------
    flow: pd.Series
        The worst absolute flow of the case.
    limit: pd.Series
        The physical limit of the case, in the unit of the flow.
    optimized: pd.Series
        Whether the branch is optimized.
    non_worsening: pd.Series
        Whether the branch is non-worsening.
    double_limits: DoubleLimits
        The lower and upper limit and the minimum non-worsening limit, relative to the physical limit.

    Returns
    -------
    pd.Series
        The optimization limits. NaN for branches that are neither optimized nor non-worsening and for branches
        without a flow or a physical limit.
    """
    upper_limit = limit * double_limits.upper
    non_worsening_limit = (flow * (1 + double_limits.non_worsening_margin)).clip(
        lower=limit * double_limits.minimum_lower_physical_non_worsening_limit
    )
    optimized_limit = flow.clip(lower=limit * double_limits.lower)
    optimized_limit = optimized_limit.where(flow <= upper_limit, upper_limit)
    both_limit = non_worsening_limit.clip(upper=limit).where(flow <= limit, limit)

    limits = non_worsening_limit.where(non_worsening)
    limits = limits.mask(optimized, optimized_limit)
    limits = limits.mask(optimized & non_worsening, both_limit)

    return limits.where(flow.notna() & limit.notna())


def _get_worst_flows(branches_df: pd.DataFrame, worst_case: pd.DataFrame, side: BranchSide, case: LimitCase) -> pd.Series:
    """Get the worst absolute current of every branch on one side for one case.

    Branches without results keep the current of the loadflow.

    Parameters
    ----------
    branches_df: pd.DataFrame
        The branches dataframe with the columns i1 and i2.
    worst_case: pd.DataFrame
        The worst absolute currents with the columns element, side, n0 and n1.
    side: BranchSide
        The side of the branches.
    case: LimitCase
        The case being looked at (N-0 or N-1).

    Returns
    -------
    pd.Series
        The worst current per branch of the dataframe.
    """
    worst = worst_case[worst_case["side"] == side.value].set_index("element").reindex(branches_df.index)
    n_0_flow = worst["n0"].fillna(branches_df[f"i{side.value}"].abs())
    if case == "n_0":
        return n_0_flow
    return worst["n1"].fillna(n_0_flow)


def get_artificial_limits_for_case(
    branches_df: pd.DataFrame,
    worst_case: pd.DataFrame,
    double_limits: DoubleLimits,
    limit: OperationalLimit,
) -> list[pd.DataFrame]:
    """Get the optimization limits of the monitored branches for one limit.

    Parameters
    ----------
    branches_df: pd.DataFrame
        The branches dataframe with the columns type, i1, i2, optimized, non_worsening, the limits and group names
        of the case and the boundary lines of the tie lines.
    worst_case: pd.DataFrame
        The worst absolute currents with the columns element, side, n0 and n1.
    double_limits: DoubleLimits
        The lower and upper limit, relative to the physical limit.
    limit: OperationalLimit
        The limit to create, e.g. ARTIFICIAL_N_0 or ARTIFICIAL_DC_N_1. It defines the case, name and acceptable duration.

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits.
    """
    case = limit.case
    sides = (BranchSide.ONE, BranchSide.TWO)
    flows = {side: _get_worst_flows(branches_df, worst_case, side, case) for side in sides}

    artificial_limits = []
    for element_type in ("LINE", "TWO_WINDINGS_TRANSFORMER"):
        elements_df = branches_df[branches_df["type"] == element_type]
        if elements_df.empty:
            continue
        for side in sides:
            new_limit = compute_artificial_limits(
                flow=flows[side][elements_df.index],
                limit=elements_df[f"{case}_i{side.value}_max"],
                optimized=elements_df["optimized"],
                non_worsening=elements_df["non_worsening"],
                double_limits=double_limits,
            )
            artificial_limits.append(
                create_current_limits_df(
                    new_limit,
                    element_type=element_type,
                    side=side,
                    limit=limit,
                    group_names=elements_df[f"{case}_group_name_{side.value}"],
                )
            )

    tie_lines_df = branches_df[branches_df["type"] == "TIE_LINE"]
    if not tie_lines_df.empty:
        tie_lines_df = tie_lines_df.assign(
            new_limit=compute_artificial_limits(
                flow=pd.concat(flows, axis=1).max(axis=1)[tie_lines_df.index],
                limit=tie_lines_df[[f"{case}_i1_max", f"{case}_i2_max"]].min(axis=1),
                optimized=tie_lines_df["optimized"],
                non_worsening=tie_lines_df["non_worsening"],
                double_limits=double_limits,
            )
        )
        for side_value, dangling_line_col in zip([1, 2], ["boundary_line1_id", "boundary_line2_id"], strict=True):
            dangling_df = tie_lines_df.set_index(dangling_line_col)
            artificial_limits.append(
                create_current_limits_df(
                    dangling_df["new_limit"],
                    element_type="BOUNDARY_LINE",
                    side=BranchSide.NONE,
                    limit=limit,
                    group_names=dangling_df[f"{case}_group_name_{side_value}"],
                )
            )
    return artificial_limits


def add_default_permanent_limits(network: Network, monitored_branch_ids: list[str], default_permanent: float) -> None:
    """Add a permanent limit to the monitored branches that have None on a branch SIDE.

    The limits of tie lines are set on their boundary lines.

    Parameters
    ----------
    network: Network
        The network to add the limits to. It is modified in place.
    monitored_branch_ids: list[str]
        The ids of the monitored branches.
    default_permanent: float
        The permanent limit in A of the added limits.
    """
    existing_limits = network.get_operational_limits().reset_index()
    has_permanent = set(
        existing_limits.loc[existing_limits["name"] == OperationalLimit.PERMANENT, ["element_id", "side"]].itertuples(
            index=False
        )
    )
    branches = network.get_branches(attributes=["type"])
    branches = branches[branches.index.isin(monitored_branch_ids)]
    tie_lines = network.get_tie_lines(attributes=["boundary_line1_id", "boundary_line2_id"])
    missing = []
    for element_id, element_type in branches["type"].items():
        for side in (BranchSide.ONE, BranchSide.TWO):
            if (element_id, side.name) in has_permanent:
                continue
            if element_type == "TIE_LINE":
                missing.append((tie_lines.at[element_id, f"boundary_line{side.value}_id"], BranchSide.NONE.name))
            elif element_type in ("LINE", "TWO_WINDINGS_TRANSFORMER"):
                missing.append((element_id, side.name))
    if not missing:
        return
    # The dataframe form silently creates nothing for elements without any limit, the keyword form does not
    network.create_loading_limits(
        element_id=[element_id for element_id, _ in missing],
        side=[side for _, side in missing],
        name=[OperationalLimit.DEFAULT_PERMANENT.value] * len(missing),
        type=["CURRENT"] * len(missing),
        value=[default_permanent] * len(missing),
        acceptable_duration=[-1] * len(missing),
    )


def _shift_occupied_durations(new_limits_df: pd.DataFrame, occupied_limits_df: pd.DataFrame) -> pd.DataFrame:
    """Raise the acceptable duration of new limits until it is free on their element, side, type and group.

    Powsybl keys temporary limits by these and the duration, not by name, so a taken duration would overwrite a limit.

    Parameters
    ----------
    new_limits_df: pd.DataFrame
        The new limits, indexed by element_id, side, type, acceptable_duration and group_name.
    occupied_limits_df: pd.DataFrame
        The limits that stay in the network, same index.

    Returns
    -------
    pd.DataFrame
        The new limits with free durations.
    """
    key_levels = ["element_id", "side", "type", "group_name"]
    new_df = new_limits_df.reset_index()
    taken = set(occupied_limits_df.reset_index()[[*key_levels, "acceptable_duration"]].itertuples(index=False, name=None))
    durations = []
    for row in new_df[[*key_levels, "acceptable_duration"]].itertuples(index=False, name=None):
        *key, duration = row
        while (*key, duration) in taken:
            duration += 1
        taken.add((*key, duration))
        durations.append(duration)
    new_df["acceptable_duration"] = durations
    return new_df.set_index(new_limits_df.index.names)


def create_artificial_limits(
    network: Network,
    nminus1_definition: Nminus1Definition,
    worst_case_currents: pl.DataFrame,
    double_limits: DoubleLimits,
    limits: tuple[OperationalLimit, ...] = ARTIFICIAL_LIMITS,
) -> pd.DataFrame:
    """Create the optimization limits of the monitored branches in the network.

    The limits replace the limits of the same element, side and name, see compute_artificial_limits.

    Parameters
    ----------
    network: Network
        The network to create the limits for. The loadflow calculation needs to have happened
    nminus1_definition: Nminus1Definition
        The N-1 definition with the optimized and non-worsening monitored branches
    worst_case_currents: pl.DataFrame
        The worst absolute currents with the columns element, side, n0 and n1,
        see extract_worst_case_branch_results_polars
    double_limits: DoubleLimits
        The lower and upper limit, relative to the physical limit.
    limits: tuple[OperationalLimit, ...]
        The limits to create, ARTIFICIAL_LIMITS by default. Use ARTIFICIAL_DC_LIMITS for DC results.

    Returns
    -------
    pd.DataFrame
        The new limits for the network including the already existing ones
    """
    monitored_branches = [element for element in nminus1_definition.monitored_elements if element.kind == "branch"]
    existing_limits = network.get_operational_limits()
    branches_df = get_branches_including_limits_and_dangling_lines(
        network.get_branches(attributes=["type", "i1", "i2"]),
        existing_limits,
        network.get_tie_lines(attributes=["boundary_line1_id", "boundary_line2_id"]),
    )
    monitored_elements_df = pd.DataFrame(
        {
            "optimized": [element.optimized for element in monitored_branches],
            "non_worsening": [element.non_worsening for element in monitored_branches],
        },
        index=[element.id for element in monitored_branches],
        dtype=bool,
    )
    branches_df[["optimized", "non_worsening"]] = monitored_elements_df.reindex(branches_df.index, fill_value=False)
    worst_case = pd.DataFrame(worst_case_currents.to_dict(as_series=False))

    new_limits = []
    for limit in limits:
        new_limits += get_artificial_limits_for_case(branches_df, worst_case, double_limits, limit)
    if not new_limits:
        return existing_limits
    new_limits_df = pd.concat(new_limits)

    # Exclude tie lines, since they cant be directly updated
    old_limits_df = existing_limits[existing_limits.element_type != "TIE_LINE"]
    key_columns = ["element_id", "side", "name"]
    replaced = pd.MultiIndex.from_frame(old_limits_df.reset_index()[key_columns]).isin(
        pd.MultiIndex.from_frame(new_limits_df.reset_index()[key_columns])
    )
    kept_limits_df = old_limits_df[~replaced]
    updated_limits_df = pd.concat([kept_limits_df, _shift_occupied_durations(new_limits_df, kept_limits_df)])
    updated_limits_df = updated_limits_df.drop(columns=["element_type"])
    updated_limits_df = updated_limits_df[
        updated_limits_df.index.get_level_values("element_id").isin(existing_limits.index.get_level_values("element_id"))
    ]
    network.create_operational_limits(updated_limits_df.reset_index("acceptable_duration"))
    return updated_limits_df


def attach_artificial_limits_to_results(
    network: Network,
    nminus1_definition: Nminus1Definition,
    security_analysis_results: LoadflowResultsPolars,
    double_limits: DoubleLimits,
    limits: tuple[OperationalLimit, ...] = ARTIFICIAL_LIMITS,
) -> LoadflowResultsPolars:
    """Create the artificial limits from the worst currents and attach the new limits to the results.

    Works for AC and DC results alike, the limits tell them apart.

    Currents do not depend on limits, so the returned results equal those of an analysis run after the limits were
    created, without running one.

    Parameters
    ----------
    network: Network
        The network to create the artificial limits on. It is modified in place.
    nminus1_definition: Nminus1Definition
        The N-1 definition with the optimized and non-worsening monitored branches.
    security_analysis_results: LoadflowResultsPolars
        The results of the AC security analysis of the network.
    double_limits: DoubleLimits
        The lower and upper limit, relative to the physical limit.
    limits: tuple[OperationalLimit, ...]
        The limits to create. Use ARTIFICIAL_DC_LIMITS for DC results.

    Returns
    -------
    LoadflowResultsPolars
        The security analysis results with the branch limits of the monitored branches, including the artificial ones.
    """
    worst_case_currents = extract_worst_case_branch_results_polars(security_analysis_results, nminus1_definition, timestep=0)
    create_artificial_limits(
        network,
        nminus1_definition,
        worst_case_currents,
        double_limits,
        limits=limits,
    )
    branch_limits = get_monitored_branch_limits_polars(network, nminus1_definition, timestep=0)
    return security_analysis_results.model_copy(update={"branch_limits": branch_limits})


def compute_initial_metrics(
    loadflow_results: LoadflowResultsPolars, nminus1_definition: Nminus1Definition
) -> dict[MetricType, float]:
    """Compute the metrics of the unsplit grid.

    Parameters
    ----------
    loadflow_results: LoadflowResultsPolars
        The AC N-1 results of the unsplit grid, with the branch limits to measure against.
    nminus1_definition: Nminus1Definition
        The N-1 definition, which holds the base case.

    Returns
    -------
    dict[MetricType, float]
        The metrics. Metrics of limits that no branch has are left out.
    """
    base_case = nminus1_definition.base_case
    metrics = compute_metrics(
        loadflow_results,
        base_case_id=base_case.id if base_case is not None else None,
        displayed_branch_ids=nminus1_definition.displayed_branch_ids,
    )
    return {name: value for name, value in metrics.items() if value is not None}


def compute_dc_artificial_limit_metrics(
    network: Network,
    nminus1_definition: Nminus1Definition,
    double_limits: DoubleLimits,
    lf_params: pypowsybl.loadflow.Parameters,
    job_id: str,
    n_processes: int,
    batch_size: Optional[int],
) -> dict[MetricType, float]:
    """Run a DC security analysis, create the artificial DC limits from it and measure the unsplit grid against them.

    The DC solver reads these limits instead of the AC based ones, as they match its DC flows.

    Parameters
    ----------
    network: Network
        The network to create the artificial DC limits on. Only its limits are modified.
    nminus1_definition: Nminus1Definition
        The N-1 definition with the optimized and non-worsening monitored branches.
    double_limits: DoubleLimits
        The lower and upper limit, relative to the physical limit.
    lf_params: pypowsybl.loadflow.Parameters
        The loadflow parameters of the import.
    job_id: str
        The job id stored with the results.
    n_processes: int
        The number of threads of the security analysis.
    batch_size: Optional[int]
        Maximum number of outage contingencies per sequential analysis.

    Returns
    -------
    dict[MetricType, float]
        The DC metrics of the unsplit grid against the artificial DC limits, named like the artificial limit metrics.
        Metrics of limits that no branch has are left out.
    """
    # A copy, since the DC loadflow also overwrites the boundary voltages of dangling lines, which are not variant-aware
    # TODO fix deep copy -> run_contingency_analysis_powsybl() modfies network
    dc_results = run_contingency_analysis_powsybl(
        net=deepcopy(network),
        n_minus_1_definition=nminus1_definition,
        job_id=job_id,
        timestep=0,
        method="dc",
        polars=True,
        lf_params=lf_params,
        n_processes=n_processes,
        batch_size=batch_size,
    )
    assert isinstance(dc_results, LoadflowResultsPolars), "Security analysis did not return LoadflowResultsPolars"
    dc_loadflow = attach_artificial_limits_to_results(
        network=network,
        nminus1_definition=nminus1_definition,
        security_analysis_results=dc_results,
        double_limits=double_limits,
        limits=ARTIFICIAL_DC_LIMITS,
    )
    # The metrics measure against the artificial limits, so expose the DC limits under their names
    dc_limit_names = {
        dc_limit.value: limit.value for dc_limit, limit in zip(ARTIFICIAL_DC_LIMITS, ARTIFICIAL_LIMITS, strict=True)
    }
    branch_limits = dc_loadflow.branch_limits.filter(
        ~pl.col("limit_name").is_in(list(dc_limit_names.values()))
    ).with_columns(pl.col("limit_name").replace(dc_limit_names))
    return compute_initial_metrics(dc_loadflow.model_copy(update={"branch_limits": branch_limits}), nminus1_definition)
