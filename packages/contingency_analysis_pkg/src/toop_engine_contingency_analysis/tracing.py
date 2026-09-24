# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""OpenTelemetry tracing for the contingency-analysis engine.

Only ``opentelemetry-api`` is required. The engine never configures an SDK or exporter: without a
tracer provider set up by the host application every span here is a no-op. Span names are
``toop.ca.*``, attribute names ``toop.*``.

Ray workers are separate processes, so the OpenTelemetry context does not travel with a task on
its own. The driver serialises it with :func:`current_trace_carrier` and the worker restores it
with :func:`attached_trace_carrier`; the host application is responsible for giving worker
processes a tracer provider (for example through Ray's ``worker_process_setup_hook``).
"""

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.metadata import PackageNotFoundError, version

import ray
from beartype.typing import Iterator, Literal, Optional, Protocol, Sized
from opentelemetry import context, propagate, trace
from opentelemetry.trace import Span
from pydantic import BaseModel

logger = logging.getLogger(__name__)

TraceDetail = Literal["phase", "outage", "outage_steps"]

#: Wall-clock reference for :func:`process_uptime_s`; a fresh Ray worker imports this module right
#: before its first task, so a small uptime marks a cold worker.
_IMPORTED_AT_MONOTONIC = time.monotonic()


def _package_version() -> str:
    try:
        return version("toop_engine_contingency_analysis")
    except PackageNotFoundError:
        return "unknown"


tracer = trace.get_tracer("toop_engine_contingency_analysis", _package_version())


class TracingConfig(BaseModel):
    """How much of one N-1 run becomes spans.

    - ``"phase"``: only the run-level phases (translation, grouping, base case, batches, ...).
    - ``"outage"``: additionally one ``toop.ca.outage`` span per outage group, carrying the
      durations of its steps as ``toop.dur.<step>_ms`` attributes. The default.
    - ``"outage_steps"``: the steps of every outage become child spans (``toop.ca.outage.<step>``).
      Several spans per outage; meant for deep dives, not for production.
    """

    detail: TraceDetail = "outage"


_detail: ContextVar[TraceDetail] = ContextVar("toop_trace_detail", default="outage")
_step_durations: ContextVar[Optional[dict[str, float]]] = ContextVar("toop_step_durations", default=None)
_outage: ContextVar[Span] = ContextVar("toop_outage_span", default=trace.INVALID_SPAN)


@contextmanager
def trace_detail(detail: TraceDetail) -> Iterator[None]:
    """Set the trace detail for the enclosed run (a context variable, so it needs re-setting inside a Ray task)."""
    token = _detail.set(detail)
    try:
        yield
    finally:
        _detail.reset(token)


def set_attrs(span: Span, **attrs: object) -> None:
    """Set span attributes, dropping ``None`` values and stringifying anything OpenTelemetry cannot carry."""
    for key, value in attrs.items():
        if value is None:
            continue
        span.set_attribute(key, value if isinstance(value, (bool, int, float, str)) else str(value))


def add_event(name: str, **attrs: object) -> None:
    """Add an event to the current span (no-op without a recording span)."""
    trace.get_current_span().add_event(name, {k: v for k, v in attrs.items() if v is not None})


@contextmanager
def span(name: str, **attrs: object) -> Iterator[Span]:
    """Start ``name`` as the current span with *attrs* set up front."""
    with tracer.start_as_current_span(name) as current:
        set_attrs(current, **attrs)
        yield current


@contextmanager
def outage_span(**attrs: object) -> Iterator[Span]:
    """One ``toop.ca.outage`` span per outage group.

    Collects the durations recorded by :func:`step` inside it and writes them as
    ``toop.dur.<step>_ms`` attributes when the span ends. Yields a non-recording span for
    detail ``"phase"``.
    """
    if _detail.get() == "phase":
        yield trace.INVALID_SPAN
        return

    durations: dict[str, float] = {}
    durations_token = _step_durations.set(durations)
    try:
        with tracer.start_as_current_span("toop.ca.outage") as current:
            outage_token = _outage.set(current)
            try:
                set_attrs(current, **attrs)
                yield current
            finally:
                _outage.reset(outage_token)
                for step_name, milliseconds in durations.items():
                    current.set_attribute(f"toop.dur.{step_name}_ms", round(milliseconds, 3))
    finally:
        _step_durations.reset(durations_token)


def current_outage_span() -> Span:
    """Return the enclosing ``toop.ca.outage`` span; non-recording outside one (including detail ``"phase"``)."""
    return _outage.get()


@contextmanager
def step(name: str, **attrs: object) -> Iterator[Span]:
    """One step of an outage (``copy_net``, ``power_flow``, ``results.branch``, ...).

    With detail ``"outage_steps"`` this is a child span ``toop.ca.outage.<name>``. Otherwise the
    duration is accumulated onto the enclosing :func:`outage_span` (repeated steps such as SpPS
    power flows add up) and the outage span itself is yielded, so attributes set by the caller
    land there.
    """
    if _detail.get() == "outage_steps":
        with tracer.start_as_current_span(f"toop.ca.outage.{name}") as current:
            set_attrs(current, **attrs)
            yield current
        return

    durations = _step_durations.get()
    if durations is None:
        yield trace.INVALID_SPAN
        return

    started = time.perf_counter()
    try:
        yield trace.get_current_span()
    finally:
        durations[name] = durations.get(name, 0.0) + (time.perf_counter() - started) * 1000.0


def current_trace_carrier() -> dict[str, str]:
    """Serialise the current trace context (W3C ``traceparent``) so a Ray task can continue the trace."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


@contextmanager
def attached_trace_carrier(carrier: Optional[dict[str, str]]) -> Iterator[None]:
    """Make the trace context from :func:`current_trace_carrier` current for the enclosed block."""
    if not carrier:
        yield
        return
    token = context.attach(propagate.extract(carrier))
    try:
        yield
    finally:
        context.detach(token)


def flush_tracer_provider() -> None:
    """Export pending spans now. Needed at the end of a Ray task: the worker process may be reused or killed."""
    force_flush = getattr(trace.get_tracer_provider(), "force_flush", None)
    if callable(force_flush):
        force_flush()


def process_uptime_s() -> float:
    """Seconds since this module was imported into the current process."""
    return time.monotonic() - _IMPORTED_AT_MONOTONIC


def ray_runtime_attrs() -> dict[str, object]:
    """``ray.*`` and ``process.pid`` attributes of the current task; empty outside a Ray worker."""
    attrs: dict[str, object] = {"process.pid": os.getpid()}
    if not ray.is_initialized():
        return attrs
    try:
        runtime = ray.get_runtime_context()
        attrs["ray.node_id"] = runtime.get_node_id()
        attrs["ray.worker_id"] = runtime.get_worker_id()
        task_id = runtime.get_task_id()
        if task_id is not None:
            attrs["ray.task_id"] = task_id
    except Exception:
        logger.debug("Ray runtime attributes unavailable", exc_info=True)
    return attrs


def loadflow_attrs(net: object) -> dict[str, object]:
    """Solver facts pandapower leaves on the net after a power flow."""
    ppc = getattr(net, "_ppc", None) or {}
    options = getattr(net, "_options", None) or {}
    return {
        "toop.lf.converged": getattr(net, "converged", None),
        "toop.lf.iterations": ppc.get("iterations"),
        "toop.lf.init": options.get("init_vm_pu"),
        "toop.lf.algorithm": options.get("algorithm"),
        "toop.lf.lightsim2grid": options.get("lightsim2grid"),
    }


class _NetTables(Protocol):
    bus: Sized
    line: Sized
    trafo: Sized
    switch: Sized


def net_size_attrs(net: _NetTables) -> dict[str, object]:
    """Cheap size proxy for spans that copy or serialise a net."""
    return {
        "toop.net.n_bus": len(net.bus),
        "toop.net.n_line": len(net.line),
        "toop.net.n_trafo": len(net.trafo),
        "toop.net.n_switch": len(net.switch),
    }
