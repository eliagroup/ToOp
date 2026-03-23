# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for the OTel-structured logger."""

import json
import time

import pytest
import structlog.testing
from toop_engine_grid_helpers.logging.config import _SEVERITY_MAP, configure
from toop_engine_grid_helpers.logging.logger import get_logger


class TestGetLogger:
    def test_returns_bound_logger(self) -> None:
        logger = get_logger("test.module")
        assert logger is not None

    def test_logger_name_in_event_dict(self) -> None:
        # configure() must run before capture_logs() so it doesn't override the capture setup
        configure()
        logger = get_logger("my.module")
        with structlog.testing.capture_logs() as cap_logs:
            logger.info("hello")
        assert cap_logs[0]["logger.name"] == "my.module"

    def test_extra_context_passed_through(self) -> None:
        configure()
        logger = get_logger("ctx.test")
        with structlog.testing.capture_logs() as cap_logs:
            logger.info("message", operation="optimize", n_buses=42)
        assert cap_logs[0]["operation"] == "optimize"
        assert cap_logs[0]["n_buses"] == 42


class TestOtelJsonShape:
    """Integration tests that validate the full JSON output shape."""

    def _log_and_parse(self, capsys: pytest.CaptureFixture[str], level: str = "info", msg: str = "hello") -> dict:  # type: ignore[type-arg]
        logger = get_logger("otel.test")
        getattr(logger, level)(msg)
        return json.loads(capsys.readouterr().out.strip())

    def test_top_level_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._log_and_parse(capsys)
        assert set(out.keys()) == {"Timestamp", "SeverityText", "Body", "TraceId", "SpanId", "Attributes"}

    def test_trace_and_span_id_are_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._log_and_parse(capsys)
        assert out["TraceId"] == ""
        assert out["SpanId"] == ""

    def test_body_contains_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._log_and_parse(capsys, msg="grid loaded")
        assert out["Body"] == "grid loaded"

    def test_timestamp_is_nanoseconds(self, capsys: pytest.CaptureFixture[str]) -> None:
        before = time.time_ns()
        out = self._log_and_parse(capsys)
        after = time.time_ns()
        assert before <= out["Timestamp"] <= after

    def test_logger_name_in_attributes(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._log_and_parse(capsys)
        assert out["Attributes"]["logger.name"] == "otel.test"

    def test_thread_name_in_attributes(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._log_and_parse(capsys)
        assert "thread.name" in out["Attributes"]


class TestSeverityMapping:
    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            ("info", "INFO"),
            ("warning", "WARN"),
            ("error", "ERROR"),
            ("critical", "FATAL"),
            # debug is below the default INFO filter — tested separately via capture_logs
        ],
    )
    def test_severity_text(self, capsys: pytest.CaptureFixture[str], level: str, expected: str) -> None:
        logger = get_logger("severity.test")
        getattr(logger, level)("msg")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["SeverityText"] == expected

    def test_debug_severity_mapping(self) -> None:
        """DEBUG is filtered at runtime (below INFO threshold) but maps correctly in the processor."""
        configure()
        logger = get_logger("dbg.test")
        with structlog.testing.capture_logs() as cap_logs:
            logger.debug("low-level detail")
        # capture_logs bypasses level filtering; verify the mapping entry exists
        assert _SEVERITY_MAP["debug"] == "DEBUG"
        # level-filtered: nothing emitted at runtime (correct — default threshold is INFO)
        assert len(cap_logs) == 0

    def test_warning_is_not_spelled_out(self, capsys: pytest.CaptureFixture[str]) -> None:
        get_logger("warn.test").warning("check")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["SeverityText"] == "WARN"
        assert out["SeverityText"] != "WARNING"


class TestServiceAttributes:
    def test_service_name_from_env(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_SERVICE_NAME", "toop-grid-helpers")
        get_logger("svc.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["service.name"] == "toop-grid-helpers"

    def test_service_namespace_from_env(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_SERVICE_NAMESPACE", "toop")
        get_logger("svc.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["service.namespace"] == "toop"

    def test_deployment_environment_from_env(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEPLOYMENT_ENV", "prd")
        get_logger("env.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["deployment.environment"] == "prd"

    def test_zone_name_from_env(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZONE_NAME", "Business")
        get_logger("zone.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["zone.name"] == "Business"

    def test_optional_k8s_pod_name(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("K8S_POD_NAME", "toop-pod-abc")
        get_logger("k8s.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["k8s.pod.name"] == "toop-pod-abc"

    def test_optional_k8s_absent_when_not_set(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("K8S_POD_NAME", raising=False)
        monkeypatch.delenv("K8S_NODE_NAME", raising=False)
        get_logger("k8s.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert "k8s.pod.name" not in out["Attributes"]
        assert "k8s.node.name" not in out["Attributes"]

    def test_optional_service_version(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_SERVICE_VERSION", "2.1.0")
        get_logger("ver.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["service.version"] == "2.1.0"


class TestExceptionAttributes:
    def test_exception_fields_on_logger_exception(self, capsys: pytest.CaptureFixture[str]) -> None:
        logger = get_logger("exc.test")
        try:
            raise RuntimeError("something broke")
        except RuntimeError:
            logger.exception("caught it")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["SeverityText"] == "ERROR"
        assert out["Attributes"]["exception.type"] == "builtins.RuntimeError"
        assert out["Attributes"]["exception.message"] == "something broke"
        assert "Traceback" in out["Attributes"]["exception.stacktrace"]

    def test_exc_info_kwarg(self, capsys: pytest.CaptureFixture[str]) -> None:
        logger = get_logger("exc.test")
        try:
            raise ValueError("bad value")
        except ValueError as exc:
            logger.error("failed", exc_info=exc)
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["exception.type"] == "builtins.ValueError"

    def test_no_exception_fields_on_normal_log(self, capsys: pytest.CaptureFixture[str]) -> None:
        get_logger("clean.test").info("all good")
        out = json.loads(capsys.readouterr().out.strip())
        assert "exception.type" not in out["Attributes"]
        assert "exception.message" not in out["Attributes"]
