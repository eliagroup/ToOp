# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests that a result filter policy reaches the pandapower result frames it is meant to shrink.

The predicates themselves are covered in ``tests/test_result_filter.py``; what is at stake here is the plumbing - that the
policy arrives, that it is applied at a point where the columns it reads are already correct, and that N-0 survives.

Note that case14 rates every branch and every bus, so this fixture has no unknown ``loading`` or ``vm_loading`` values at
all. Asserting unknown-is-kept here would pass vacuously; that rule is covered against real data by the powsybl wiring
test, whose fixture leaves about half its branches unrated.
"""

import pandapower as pp
import polars as pl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from toop_engine_contingency_analysis.pandapower import contingency_analysis_pandapower
from toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower import (
    run_contingency_analysis_pandapower,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    ContingencyAnalysisConfig,
    ParallelConfig,
)
from toop_engine_contingency_analysis.result_filter import apply_result_filter
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.loadflow_result_filter import (
    BranchLoadflowResultFilter,
    LoadflowResultFilter,
    NodeLoadflowResultFilter,
)
from toop_engine_interfaces.loadflow_result_helpers import convert_polars_loadflow_results_to_pandas
from toop_engine_interfaces.loadflow_result_helpers_polars import (
    load_loadflow_results_polars,
    save_loadflow_results_polars,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
)

BASECASE = "BASECASE"
THRESHOLD = 0.7

ACTIVE_POLICY = LoadflowResultFilter(
    branch_filters=BranchLoadflowResultFilter(loading_above=THRESHOLD),
    node_filters=NodeLoadflowResultFilter(vm_loading_above=THRESHOLD),
)


@pytest.fixture
def nminus1_definition() -> Nminus1Definition:
    """An N-1 definition over every line and bus of case14, including an explicit N-0 case.

    Returns
    -------
    Nminus1Definition
        Monitored lines and buses, one contingency per line, plus the basecase.
    """
    net = pp.networks.case14()
    return Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=get_globally_unique_id(index, "line"), name=str(index), kind="branch", type="line")
            for index in net.line.index
        ]
        + [
            MonitoredElement(id=get_globally_unique_id(index, "bus"), name=str(index), kind="bus", type="bus")
            for index in net.bus.index
        ],
        contingencies=[Contingency(id=BASECASE, elements=[])]
        + [
            Contingency(
                id=str(index),
                elements=[
                    GridElement(id=get_globally_unique_id(index, "line"), name=str(index), kind="branch", type="line")
                ],
            )
            for index in net.line.index
        ],
    )


def _run(nminus1_definition: Nminus1Definition, result_filter: LoadflowResultFilter) -> LoadflowResultsPolars:
    """Run the analysis under a filter policy.

    Parameters
    ----------
    nminus1_definition : Nminus1Definition
        The N-1 definition to compute.
    result_filter : LoadflowResultFilter
        The policy to apply.

    Returns
    -------
    LoadflowResultsPolars
        The results of the run.
    """
    return run_contingency_analysis_pandapower(
        net=pp.networks.case14(),
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
        cfg=ContingencyAnalysisConfig(method="ac", polars=True, result_filter=result_filter),
    )


def test_filter_reaches_the_pandapower_results(nminus1_definition):
    """The policy arrives, shrinks both tables, drops only what it should, and leaves N-0 whole."""
    unfiltered = _run(nminus1_definition, LoadflowResultFilter())
    filtered = _run(nminus1_definition, ACTIVE_POLICY)

    unfiltered_branches = unfiltered.branch_results.collect()
    unfiltered_nodes = unfiltered.node_results.collect()
    filtered_branches = filtered.branch_results.collect()
    filtered_nodes = filtered.node_results.collect()

    default_cfg = _run(nminus1_definition, ContingencyAnalysisConfig().result_filter)
    assert unfiltered_branches.equals(default_cfg.branch_results.collect()), (
        "the default config must produce exactly what it produced before the filter existed"
    )
    assert unfiltered_nodes.equals(default_cfg.node_results.collect()), (
        "the default config must produce exactly what it produced before the filter existed"
    )

    assert filtered_branches.height < unfiltered_branches.height, "case14 should have quiet branches to drop"
    assert filtered_nodes.height < unfiltered_nodes.height, "case14 sits near nominal, so most N-1 node rows should go"

    dropped_branches = unfiltered_branches.join(
        filtered_branches.select(["contingency", "element", "side"]), on=["contingency", "element", "side"], how="anti"
    )
    assert (dropped_branches["contingency"] != BASECASE).all(), "N-0 branch rows are exempt"
    assert dropped_branches["loading"].fill_nan(None).drop_nulls().lt(THRESHOLD).all(), (
        "every dropped branch row must have been below the threshold"
    )

    dropped_nodes = unfiltered_nodes.join(
        filtered_nodes.select(["contingency", "element"]), on=["contingency", "element"], how="anti"
    )
    assert (dropped_nodes["contingency"] != BASECASE).all(), "N-0 node rows are exempt"
    assert dropped_nodes["vm_loading"].fill_nan(None).drop_nulls().abs().lt(THRESHOLD).all(), (
        "every dropped node row must have been inside the band"
    )

    assert filtered_branches.filter(pl.col("contingency") == BASECASE).equals(
        unfiltered_branches.filter(pl.col("contingency") == BASECASE)
    ), "the N-0 branch rows come through whole"
    assert filtered_nodes.filter(pl.col("contingency") == BASECASE).equals(
        unfiltered_nodes.filter(pl.col("contingency") == BASECASE)
    ), "the N-0 node rows come through whole"

    # Filtering per outage inside the lib has to mean exactly the same thing as filtering the finished result. This is
    # what pins the application point: applied any earlier, the columns the predicate reads are not final yet.
    expected = apply_result_filter(unfiltered, ACTIVE_POLICY, BASECASE)
    assert filtered_branches.equals(expected.branch_results.collect()), (
        "per-outage filtering must match filtering the concatenated result"
    )
    assert filtered_nodes.equals(expected.node_results.collect()), (
        "per-outage filtering must match filtering the concatenated result"
    )


def test_the_policy_is_stored_with_the_results(nminus1_definition, tmp_path):
    """Filtered results are indistinguishable from quiet ones unless they say what filtered them."""
    unfiltered = _run(nminus1_definition, LoadflowResultFilter())
    filtered = _run(nminus1_definition, ACTIVE_POLICY)

    assert unfiltered.result_filter is None, "an unfiltered result must not claim a policy"
    assert filtered.result_filter == ACTIVE_POLICY, "the results must carry the policy they were produced under"

    fs = DirFileSystem(path=str(tmp_path), fs=LocalFileSystem())
    reference = save_loadflow_results_polars(fs, "filtered", filtered)
    assert load_loadflow_results_polars(fs, reference).result_filter == ACTIVE_POLICY, (
        "the policy must survive a save/load round trip, or a stored result cannot be read safely"
    )

    reference = save_loadflow_results_polars(fs, "unfiltered", unfiltered)
    assert load_loadflow_results_polars(fs, reference).result_filter is None, "an unfiltered result stays unfiltered"

    as_pandas = convert_polars_loadflow_results_to_pandas(filtered)
    assert as_pandas.result_filter == ACTIVE_POLICY, "conversion to pandas must not lose the policy"


def test_the_parallel_path_is_told_which_contingency_is_the_basecase(nminus1_definition, monkeypatch):
    """The parallel path empties each batch's contingency list, so the basecase id has to be resolved upstream of it.

    Without this, the N-0 exemption would silently stop working as soon as n_processes > 1.
    """
    captured = {}

    class _Captured(Exception):
        """Raised to stop the run once the parallel context has been observed."""

    def _capture(net, n_minus_1_definition, ctx, **kwargs):  # noqa: ANN001, ANN003, ANN202, ARG001
        captured["ctx"] = ctx
        raise _Captured

    monkeypatch.setattr(contingency_analysis_pandapower, "run_contingency_analysis_parallel", _capture)

    with pytest.raises(_Captured):
        run_contingency_analysis_pandapower(
            net=pp.networks.case14(),
            n_minus_1_definition=nminus1_definition,
            job_id="test_job",
            timestep=0,
            cfg=ContingencyAnalysisConfig(
                method="ac",
                polars=True,
                parallel=ParallelConfig(n_processes=2),
                result_filter=ACTIVE_POLICY,
            ),
        )

    assert captured["ctx"].basecase_contingency_id == BASECASE, "the resolved N-0 id must reach the parallel workers"
    assert captured["ctx"].result_filter == ACTIVE_POLICY, "the policy must reach the parallel workers"
