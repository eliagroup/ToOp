# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Base-case screening: a violating N-0 state short-circuits cascade simulation."""

import pandapower as pp
import pandas as pd
import pytest
from test_busbar_cascade import _build_nminus1_definition, create_net
from toop_engine_contingency_analysis.pandapower import run_contingency_analysis_pandapower
from toop_engine_contingency_analysis.pandapower.cascade.basecase import (
    build_basecase_cascade_results,
    screen_basecase_for_cascade,
)
from toop_engine_contingency_analysis.pandapower.cascade.configuration import DistanceProtectionSeverity
from toop_engine_contingency_analysis.pandapower.cascade.detection.basecase_screen import (
    _resolve_branch_element,
)
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeEvent, CascadeReasonType
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import translate_nminus1_for_pandapower
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    CascadeConfig,
    ContingencyAnalysisConfig,
    DistanceProtectionConfig,
    DistanceProtectionFactors,
    OverloadConfig,
    ParallelConfig,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR, get_globally_unique_id
from toop_engine_interfaces.loadflow_results import ConvergenceStatus, LoadflowResults

#: Base-case violations reuse the ordinary cascade reasons, so they are identified by their
#: step number: simulated cascade events always start at 1, leaving step 0 to the base case.
BASECASE_CASCADE_NUMBER = 0


def _basecase_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["cascade_number"] == BASECASE_CASCADE_NUMBER]


def _overloaded_net() -> pp.pandapowerNet:
    """Build the busbar net driven far past its line ratings already in the base case."""
    net = create_net()
    net.load.loc[net.load["origin_id"].eq("load_at_b1"), ["p_mw", "q_mvar"]] = [2000.0, 280.0]
    net.sgen.loc[net.sgen.index, ["p_mw", "q_mvar"]] = [1000.0, 110.0]
    net.line["r_ohm_per_km"] *= 0.2
    net.line["x_ohm_per_km"] *= 0.2

    for table in ("line", "bus", "switch"):
        net[table]["global_id"] = net[table].index.map(lambda idx, t=table: get_globally_unique_id(idx, t))
    return net


def _healthy_net() -> pp.pandapowerNet:
    """Build the same net left at its original, comfortably loaded base case."""
    net = create_net()
    for table in ("line", "bus", "switch"):
        net[table]["global_id"] = net[table].index.map(lambda idx, t=table: get_globally_unique_id(idx, t))
    return net


def _cascade_config(*, stop_on_basecase: bool) -> CascadeConfig:
    return CascadeConfig(
        depth_limit=3,
        overload=OverloadConfig(current_loading_threshold=1.5),
        min_island_size=2,
        cascade_log_elements=["line", "switch"],
        distance_protection=DistanceProtectionConfig(
            alarm=DistanceProtectionFactors(
                basecase_line=1.0,
                basecase_transformer=1.0,
                basecase_bus_coupler=1.0,
                contingency_line=1.0,
                contingency_transformer=1.0,
                contingency_bus_coupler=1.0,
            ),
            warning=DistanceProtectionFactors(
                basecase_line=1.5,
                basecase_transformer=1.5,
                basecase_bus_coupler=1.5,
                contingency_line=1.5,
                contingency_transformer=1.5,
                contingency_bus_coupler=1.5,
            ),
        ),
        stop_cascade_on_basecase_violation=stop_on_basecase,
    )


def _run(net: pp.pandapowerNet, cascade_cfg: CascadeConfig) -> LoadflowResults:
    return run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=_build_nminus1_definition(net),
        job_id="test",
        timestep=0,
        cfg=ContingencyAnalysisConfig(
            method="ac",
            min_island_size=2,
            cascade=cascade_cfg,
            parallel=ParallelConfig(n_processes=1, batch_size=None),
            runpp_kwargs={"lightsim2grid": False, "enforce_q_lims": True},
        ),
    )


def _monitored_elements(net: pp.pandapowerNet) -> pd.DataFrame:
    """The translated monitored-element table the screen expects."""
    definition = translate_nminus1_for_pandapower(_build_nminus1_definition(net), net)
    return definition.monitored_elements


def _cascade_rows(lf_results: LoadflowResults) -> list[dict]:
    return lf_results.cascade_results.reset_index().to_dict(orient="records")


def test_violating_basecase_reports_itself_and_skips_every_contingency_cascade() -> None:
    """The run reports the base-case violation instead of simulating any contingency cascade."""
    rows = _cascade_rows(_run(_overloaded_net(), _cascade_config(stop_on_basecase=True)))

    assert rows, "the base-case violation should still be reported"
    # Every row belongs to the base case, not to a contingency, and sits before step 1.
    assert {row["contingency"] for row in rows} == {"BASECASE"}
    assert {row["cascade_number"] for row in rows} == {BASECASE_CASCADE_NUMBER}
    # The ordinary cascade reasons are reused; no base-case-specific reason exists.
    assert {row["cascade_reason"] for row in rows} <= {
        CascadeReasonType.CASCADE_REASON_CURRENT.value,
        CascadeReasonType.CASCADE_REASON_DISTANCE.value,
    }


def test_basecase_overload_reports_the_loading_that_violated() -> None:
    rows = _basecase_rows(_cascade_rows(_run(_overloaded_net(), _cascade_config(stop_on_basecase=True))))
    overload_rows = [row for row in rows if row["cascade_reason"] == CascadeReasonType.CASCADE_REASON_CURRENT.value]

    assert overload_rows
    for row in overload_rows:
        assert row["loading"] > 1.5
        assert row["element_mrid"] is not None
        assert row["element_name"] is not None


def test_basecase_distance_protection_reports_r_and_x() -> None:
    """A base-case relay trip carries the R/X it was measured at, like a simulated one does."""
    rows = _basecase_rows(_cascade_rows(_run(_overloaded_net(), _cascade_config(stop_on_basecase=True))))
    relay_rows = [row for row in rows if row["cascade_reason"] == CascadeReasonType.CASCADE_REASON_DISTANCE.value]

    assert relay_rows
    for row in relay_rows:
        assert row["r_ohm"] is not None
        assert row["x_ohm"] is not None
        assert row["distance_protection_severity"] in {
            DistanceProtectionSeverity.DANGER.value,
            DistanceProtectionSeverity.ALARM.value,
            DistanceProtectionSeverity.WARNING.value,
        }


def test_skipping_the_cascade_keeps_the_n1_loadflow_results() -> None:
    """Only cascade simulation is skipped; the contingency results themselves still come back."""
    lf_results = _run(_overloaded_net(), _cascade_config(stop_on_basecase=True))

    contingencies = lf_results.branch_results.reset_index()["contingency"].unique()
    assert len(contingencies) > 1, "contingency load flows must still run"
    assert any(warning.startswith("Base case already violates") for warning in lf_results.warnings)


def test_flag_off_simulates_the_cascade_from_the_violating_basecase() -> None:
    """With screening disabled the old behaviour is restored: real cascade events, no base-case rows."""
    rows = _cascade_rows(_run(_overloaded_net(), _cascade_config(stop_on_basecase=False)))

    assert rows
    assert not _basecase_rows(rows)
    assert max(row["cascade_number"] for row in rows) >= 1


def test_healthy_basecase_is_not_flagged() -> None:
    """A base case within its limits must not short-circuit anything."""
    rows = _cascade_rows(_run(_healthy_net(), _cascade_config(stop_on_basecase=True)))

    assert not _basecase_rows(rows)


@pytest.mark.parametrize("stop_on_basecase", [True, False])
def test_healthy_basecase_gives_the_same_result_either_way(stop_on_basecase: bool) -> None:
    """Screening is inert on a clean base case, so the flag cannot change the outcome there."""
    rows = _cascade_rows(_run(_healthy_net(), _cascade_config(stop_on_basecase=stop_on_basecase)))

    assert not _basecase_rows(rows)


def test_screening_is_skipped_when_the_basecase_did_not_converge() -> None:
    """A failed base case leaves stale ``res_*`` tables, so there is nothing sound to screen."""
    net = _healthy_net()

    events = screen_basecase_for_cascade(
        net,
        cascade_configuration=_cascade_config(stop_on_basecase=True),
        monitored_elements=_monitored_elements(net),
        switch_element_mapping=pd.DataFrame(),
        bus_couplers_mrids=set(),
        timestep=0,
        basecase_status=ConvergenceStatus.FAILED,
    )

    assert events == []


def test_screening_is_skipped_when_cascade_is_disabled() -> None:
    """No cascade configuration means nothing to screen for."""
    net = _healthy_net()

    events = screen_basecase_for_cascade(
        net,
        cascade_configuration=None,
        monitored_elements=_monitored_elements(net),
        switch_element_mapping=pd.DataFrame(),
        bus_couplers_mrids=set(),
        timestep=0,
        basecase_status=ConvergenceStatus.CONVERGED,
    )

    assert events == []


def test_no_events_builds_an_empty_result_table() -> None:
    """A clean base case still has to produce a frame with the right schema, not None."""
    frame = build_basecase_cascade_results([], timestep=0)

    assert frame.is_empty()
    for column in ("timestep", "contingency", "cascade_number", "cascade_reason", "r_ohm", "x_ohm"):
        assert column in frame.columns


def test_events_are_reported_under_the_basecase_contingency() -> None:
    """Every base-case row belongs to BASECASE at step 0, whatever the event carried."""
    frame = build_basecase_cascade_results(
        [
            CascadeEvent(
                element_mrid="line-1",
                element_id=f"0{SEPARATOR}line",
                element_name="L1",
                cascade_number=BASECASE_CASCADE_NUMBER,
                cascade_reason=CascadeReasonType.CASCADE_REASON_CURRENT,
                loading=1.7,
            )
        ],
        timestep=3,
    )

    row = frame.to_dicts()[0]
    assert row["timestep"] == 3
    assert row["contingency"] == "BASECASE"
    assert row["cascade_number"] == BASECASE_CASCADE_NUMBER
    assert row["cascade_reason"] == CascadeReasonType.CASCADE_REASON_CURRENT.value
    assert row["loading"] == 1.7
    assert row["r_ohm"] is None


@pytest.mark.parametrize(
    "element_id",
    [
        "no-separator-at-all",
        f"{SEPARATOR}line",
        f"0{SEPARATOR}not_a_table",
        f"999999{SEPARATOR}line",
        f"not_an_int{SEPARATOR}line",
    ],
)
def test_unresolvable_element_ids_report_no_name(element_id: str) -> None:
    """An id that names no row must degrade to ``(None, None)`` rather than raise.

    The reporting path runs after a violation has already been found, so a malformed id must
    not lose the whole report.
    """
    assert _resolve_branch_element(_healthy_net(), element_id) == (None, None)


def test_resolvable_element_id_reports_mrid_and_name() -> None:
    """The counterpart: a well-formed id resolves to the element's origin id and name."""
    net = _healthy_net()
    first = net.line.index[0]

    mrid, name = _resolve_branch_element(net, f"{first}{SEPARATOR}line")

    assert mrid == net.line.loc[first, "origin_id"]
    assert name == net.line.loc[first, "name"]
