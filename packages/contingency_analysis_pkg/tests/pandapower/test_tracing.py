# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""OpenTelemetry spans of the pandapower contingency analysis.

The engine only depends on ``opentelemetry-api``; these tests plug in the SDK with an in-memory
exporter to look at the spans. A tracer provider can be set once per process, so one provider
with a shared exporter is installed for the whole module and cleared between tests.
"""

import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pandapower as pp
import pytest
import ray
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from toop_engine_contingency_analysis.pandapower import get_full_nminus1_definition_pandapower
from toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower import (
    run_contingency_analysis_pandapower,
    run_contingency_analysis_sequential,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import translate_nminus1_for_pandapower
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    ContingencyAnalysisConfig,
    PandapowerContingencyGroup,
    ParallelConfig,
    SequentialContingencyAnalysisContext,
    SlackAllocationConfig,
)
from toop_engine_contingency_analysis.tracing import (
    TracingConfig,
    attached_trace_carrier,
    current_trace_carrier,
    tracer,
)
from toop_engine_grid_helpers.pandapower.example_grids import example_multivoltage_cross_coupler
from toop_engine_interfaces.nminus1_definition import Nminus1Definition

_EXPORTER = InMemorySpanExporter()


def _install_provider() -> None:
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))


_install_provider()


@pytest.fixture()
def spans() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER


def _by_name(exporter: InMemorySpanExporter) -> dict[str, list[ReadableSpan]]:
    grouped: dict[str, list[ReadableSpan]] = {}
    for span in exporter.get_finished_spans():
        grouped.setdefault(span.name, []).append(span)
    return grouped


def _children_of(exporter: InMemorySpanExporter, parent: ReadableSpan) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.parent is not None and s.parent.span_id == parent.context.span_id]


def _definition(net: pp.pandapowerNet, contingency_limit: int | None = 6) -> Nminus1Definition:
    full = get_full_nminus1_definition_pandapower(net)
    basecase = [c for c in full.contingencies if c.is_basecase()]
    others = [c for c in full.contingencies if not c.is_basecase()]
    return Nminus1Definition(
        monitored_elements=full.monitored_elements,
        contingencies=basecase + others[:contingency_limit],
        id_type=full.id_type,
    )


def _config(detail: str = "outage", n_processes: int = 1, apply_outage_grouping: bool = False) -> ContingencyAnalysisConfig:
    # Grouping stays off on the bus-branch oberrhein net: there every contingency shares one
    # component and the grouped outage takes out the slack bus.
    return ContingencyAnalysisConfig(
        method="dc",
        apply_outage_grouping=apply_outage_grouping,
        parallel=ParallelConfig(n_processes=n_processes),
        tracing=TracingConfig(detail=detail),
    )


def _run(net: pp.pandapowerNet, cfg: ContingencyAnalysisConfig, on_progress=None) -> None:
    run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=_definition(net),
        job_id="trace_job",
        timestep=0,
        cfg=cfg,
        on_progress=on_progress,
    )


def test_sequential_run_produces_the_phase_span_tree(pandapower_net: pp.pandapowerNet, spans: InMemorySpanExporter) -> None:
    _run(pandapower_net, _config())

    by_name = _by_name(spans)
    (run,) = by_name["toop.ca.run"]
    assert run.parent is None
    assert run.attributes["toop.job_id"] == "trace_job"
    assert run.attributes["toop.method"] == "dc"
    assert run.attributes["toop.outage_grouping"] is False
    assert run.attributes["toop.n_groups"] == len(by_name["toop.ca.outage"])

    phase_children = {s.name for s in _children_of(spans, run)}
    assert phase_children == {
        "toop.ca.translate_nminus1",
        "toop.ca.basecase_loadflow",
        "toop.ca.switch_element_mapping",
        "toop.ca.basecase_cascade_screen",
        "toop.ca.copy_basecase_net",
        "toop.ca.sequential",
        "toop.ca.concatenate_results",
        "toop.ca.to_pandas",
    }
    assert run.attributes["toop.n_groups"] == 7

    (basecase,) = by_name["toop.ca.basecase_loadflow"]
    assert basecase.attributes["toop.status"] == "CONVERGED"
    assert {s.name for s in _children_of(spans, basecase)} == {"toop.ca.slack_allocation"}

    (sequential,) = by_name["toop.ca.sequential"]
    sequential_children = Counter(s.name for s in _children_of(spans, sequential))
    assert sequential_children["toop.ca.result_constants"] == 1
    assert sequential_children["toop.ca.outage"] == run.attributes["toop.n_groups"]
    # No cascade config, so the run constants phase is skipped.
    assert "toop.ca.cascade_run_constants" not in by_name


def test_outage_grouping_is_a_phase_with_the_component_build_inside(spans: InMemorySpanExporter) -> None:
    net = example_multivoltage_cross_coupler()
    run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=_definition(net, contingency_limit=None),
        job_id="trace_job",
        timestep=0,
        cfg=_config(apply_outage_grouping=True),
    )

    by_name = _by_name(spans)
    (run,) = by_name["toop.ca.run"]
    (grouping,) = by_name["toop.ca.outage_grouping"]
    assert grouping.parent.span_id == run.context.span_id
    assert grouping.attributes["toop.n_groups_out"] == run.attributes["toop.n_groups"]
    assert grouping.attributes["toop.n_groups_out"] <= grouping.attributes["toop.n_contingencies_in"]
    (components,) = _children_of(spans, grouping)
    assert components.name == "toop.ca.connected_components"
    assert components.attributes["toop.n_components"] > 0
    assert len(by_name["toop.ca.outage"]) == run.attributes["toop.n_groups"]


def test_outage_spans_carry_status_and_step_durations(pandapower_net: pp.pandapowerNet, spans: InMemorySpanExporter) -> None:
    _run(pandapower_net, _config(detail="outage"))

    outages = _by_name(spans)["toop.ca.outage"]
    assert outages
    for outage in outages:
        assert outage.attributes["toop.status"] in {"CONVERGED", "FAILED", "NO_CALCULATION"}
        assert outage.attributes["toop.n_contingencies"] >= 1
        assert outage.attributes["toop.dur.copy_net_ms"] >= 0
        assert outage.attributes["toop.dur.topology_ms"] >= 0
        assert outage.attributes["toop.dur.results_ms"] >= 0
        # Steps are attributes, not child spans, at this detail level.
        assert _children_of(spans, outage) == []

    converged = [o for o in outages if o.attributes["toop.status"] == "CONVERGED"]
    assert converged
    assert all(o.attributes["toop.dur.power_flow_ms"] >= 0 for o in converged)
    assert all(o.attributes["toop.lf.converged"] is True for o in converged)
    assert all(o.attributes["toop.dur.results.branch_ms"] >= 0 for o in converged)


def test_outage_steps_detail_turns_steps_into_child_spans(
    pandapower_net: pp.pandapowerNet, spans: InMemorySpanExporter
) -> None:
    _run(pandapower_net, _config(detail="outage_steps"))

    outages = _by_name(spans)["toop.ca.outage"]
    converged = [o for o in outages if o.attributes["toop.status"] == "CONVERGED"]
    assert converged
    step_names = {s.name for s in _children_of(spans, converged[0])}
    assert {
        "toop.ca.outage.copy_net",
        "toop.ca.outage.topology",
        "toop.ca.outage.power_flow",
        "toop.ca.outage.results",
        "toop.ca.outage.cascade",
    } <= step_names
    assert not any(key.startswith("toop.dur.") for key in converged[0].attributes)

    (results_step,) = [s for s in _children_of(spans, converged[0]) if s.name == "toop.ca.outage.results"]
    assert {"toop.ca.outage.results.branch", "toop.ca.outage.results.node", "toop.ca.outage.results.switch"} <= {
        s.name for s in _children_of(spans, results_step)
    }


def test_phase_detail_has_no_outage_spans(pandapower_net: pp.pandapowerNet, spans: InMemorySpanExporter) -> None:
    _run(pandapower_net, _config(detail="phase"))

    by_name = _by_name(spans)
    assert "toop.ca.run" in by_name
    assert "toop.ca.sequential" in by_name
    assert "toop.ca.outage" not in by_name
    assert not any(name.startswith("toop.ca.outage.") for name in by_name)
    # Per-outage attributes must not leak onto the enclosing span when there is no outage span.
    (sequential,) = by_name["toop.ca.sequential"]
    assert "toop.status" not in sequential.attributes


def test_progress_reports_are_span_events(pandapower_net: pp.pandapowerNet, spans: InMemorySpanExporter) -> None:
    _run(pandapower_net, _config(), on_progress=lambda done, total: None)

    events = sorted(
        (event for span in spans.get_finished_spans() for event in span.events if event.name == "progress"),
        key=lambda event: event.timestamp,
    )
    assert events
    assert events[0].attributes["toop.progress.done"] == 0
    total = events[0].attributes["toop.progress.total"]
    assert events[-1].attributes == {"toop.progress.done": total, "toop.progress.total": total}


def test_a_carrier_makes_the_batch_span_part_of_the_callers_trace(
    pandapower_net: pp.pandapowerNet, spans: InMemorySpanExporter
) -> None:
    """The worker-side entry point continues the trace it is handed, in-process here."""
    pp_definition = translate_nminus1_for_pandapower(_definition(pandapower_net), pandapower_net)
    pp_definition.grouped_contingencies = [
        PandapowerContingencyGroup(contingencies=[c], elements=c.elements, outage_group_id=c.unique_id)
        for c in pp_definition.contingencies
    ]
    pp.rundcpp(pandapower_net)
    ctx = SequentialContingencyAnalysisContext(
        job_id="trace_job",
        timestep=0,
        slack_allocation_config=SlackAllocationConfig(),
        method="dc",
        basecase_net=deepcopy(pandapower_net),
        switch_element_mapping=_empty_switch_mapping(),
        spps_conditions=pp_definition.spps_conditions,
        spps_actions=pp_definition.spps_actions,
    )

    with tracer.start_as_current_span("driver") as driver:
        carrier = current_trace_carrier()
    assert "traceparent" in carrier

    run_contingency_analysis_sequential(
        net=pandapower_net,
        n_minus_1_definition=pp_definition,
        ctx=ctx,
        trace_carrier=carrier,
        batch_index=3,
        submitted_at=0.0,
    )

    (batch,) = _by_name(spans)["toop.ca.batch"]
    assert batch.context.trace_id == driver.context.trace_id
    assert batch.parent.span_id == driver.context.span_id
    assert batch.attributes["toop.batch_index"] == 3
    assert batch.attributes["toop.n_groups"] == len(pp_definition.grouped_contingencies)
    assert batch.attributes["toop.submit_to_start_ms"] > 0
    assert batch.attributes["process.pid"] == os.getpid()
    assert "toop.ca.sequential" not in _by_name(spans)


def _empty_switch_mapping():
    from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
    from toop_engine_interfaces.loadflow_results import SwitchElementMappingSchema

    return get_empty_dataframe_from_model(SwitchElementMappingSchema)


class _JsonLinesExporter(SpanExporter):
    """One JSON object per line, appended to a file per process; readable from another process."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def export(self, spans) -> SpanExportResult:
        with self._path.open("a") as handle:
            for span in spans:
                handle.write(json.dumps(json.loads(span.to_json())) + "\n")
        return SpanExportResult.SUCCESS


def _worker_tracing_hook() -> None:
    """What a host application installs on Ray workers (``runtime_env["worker_process_setup_hook"]``)."""
    span_dir = Path(os.environ["TOOP_TEST_SPAN_DIR"])
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_JsonLinesExporter(span_dir / f"{os.getpid()}.jsonl")))
    trace.set_tracer_provider(provider)


# Test modules are not importable on Ray workers, so ship the hook (and the exporter class it uses) by value.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])


@pytest.fixture()
def ray_with_worker_tracing(tmp_path: Path, worker_id: str):
    if ray.is_initialized():
        ray.shutdown()
    span_dir = tmp_path / "spans"
    span_dir.mkdir()
    ray.init(
        _temp_dir=str(Path("/tmp") / f"tr-{worker_id}"),
        include_dashboard=False,
        namespace=f"pytest-tracing-{os.environ.get('PYTEST_XDIST_WORKER', 'local')}",
        runtime_env={
            "env_vars": {"TOOP_TEST_SPAN_DIR": str(span_dir)},
            "worker_process_setup_hook": _worker_tracing_hook,
        },
    )
    yield span_dir
    ray.shutdown()


def _worker_spans(span_dir: Path) -> list[dict]:
    return [json.loads(line) for file in span_dir.glob("*.jsonl") for line in file.read_text().splitlines() if line]


@pytest.mark.xdist_group("ray")
def test_the_carrier_survives_the_trip_into_a_ray_task(ray_with_worker_tracing: Path) -> None:
    @ray.remote
    def trace_id_in_worker(carrier: dict[str, str]) -> int:
        with attached_trace_carrier(carrier):
            return trace.get_current_span().get_span_context().trace_id

    with tracer.start_as_current_span("driver") as driver:
        carrier = current_trace_carrier()
        assert ray.get(trace_id_in_worker.remote(carrier)) == driver.context.trace_id


@pytest.mark.xdist_group("ray")
def test_parallel_run_links_worker_batch_spans_to_the_driver_trace(
    pandapower_net: pp.pandapowerNet, spans: InMemorySpanExporter, ray_with_worker_tracing: Path
) -> None:
    _run(pandapower_net, _config(n_processes=2))

    by_name = _by_name(spans)
    (run,) = by_name["toop.ca.run"]
    (parallel,) = by_name["toop.ca.parallel"]
    assert parallel.attributes["toop.n_batches"] == 2
    assert parallel.attributes["toop.first_result_after_ms"] > 0
    assert {s.name for s in _children_of(spans, parallel)} == {
        "toop.ca.ray_init",
        "toop.ca.put_inputs",
        "toop.ca.submit",
        "toop.ca.wait",
    }
    (ray_init,) = by_name["toop.ca.ray_init"]
    assert ray_init.attributes["ray.was_initialized"] is True
    submitted = [e for e in by_name["toop.ca.submit"][0].events if e.name == "batch_submitted"]
    assert [e.attributes["toop.batch_index"] for e in submitted] == [0, 1]
    finished = [
        e for s in (by_name["toop.ca.submit"] + by_name["toop.ca.wait"]) for e in s.events if e.name == "batch_finished"
    ]
    assert sorted(e.attributes["toop.batch_index"] for e in finished) == [0, 1]
    # The driver process never sees the worker spans.
    assert "toop.ca.batch" not in by_name

    worker_spans = _worker_spans(ray_with_worker_tracing)
    batches = [s for s in worker_spans if s["name"] == "toop.ca.batch"]
    assert sorted(b["attributes"]["toop.batch_index"] for b in batches) == [0, 1]
    driver_trace_id = f"0x{run.context.trace_id:032x}"
    assert {b["context"]["trace_id"] for b in batches} == {driver_trace_id}
    assert {b["parent_id"] for b in batches} == {f"0x{parallel.context.span_id:016x}"}
    for batch in batches:
        assert batch["attributes"]["toop.submit_to_start_ms"] > 0
        assert "ray.task_id" in batch["attributes"]
        assert batch["attributes"]["process.pid"] != os.getpid()

    outages = [s for s in worker_spans if s["name"] == "toop.ca.outage"]
    assert len(outages) == run.attributes["toop.n_groups"]
    assert {o["context"]["trace_id"] for o in outages} == {driver_trace_id}
    assert {o["parent_id"] for o in outages} == {b["context"]["span_id"] for b in batches}
