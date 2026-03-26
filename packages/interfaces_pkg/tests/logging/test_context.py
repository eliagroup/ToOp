# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for the job-context mechanism and Kafka header helpers."""

import json
import threading
from typing import Any

import pytest
from toop_engine_grid_helpers.logging import bind_context, clear_context, context_to_headers, get_context, headers_to_context
from toop_engine_interfaces.logging.logger import get_logger


class TestBindContext:
    def test_key_appears_in_attributes(self, capsys: pytest.CaptureFixture[str]) -> None:
        bind_context(optimization_id="opt-abc")
        get_logger("ctx.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["optimization_id"] == "opt-abc"

    def test_multiple_calls_merge(self, capsys: pytest.CaptureFixture[str]) -> None:
        bind_context(preprocess_id="pre-xyz")
        bind_context(optimization_id="opt-abc")
        get_logger("ctx.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["preprocess_id"] == "pre-xyz"
        assert out["Attributes"]["optimization_id"] == "opt-abc"

    def test_later_call_overwrites_same_key(self, capsys: pytest.CaptureFixture[str]) -> None:
        bind_context(optimization_id="old")
        bind_context(optimization_id="new")
        get_logger("ctx.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["optimization_id"] == "new"

    def test_get_context_reflects_bindings(self) -> None:
        bind_context(optimization_id="opt-abc", preprocess_id="pre-xyz")
        ctx = get_context()
        assert ctx == {"optimization_id": "opt-abc", "preprocess_id": "pre-xyz"}

    def test_get_context_returns_copy(self) -> None:
        bind_context(optimization_id="opt-abc")
        ctx = get_context()
        ctx["injected"] = "should-not-leak"
        assert "injected" not in get_context()


class TestClearContext:
    def test_clear_removes_all_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        bind_context(optimization_id="opt-abc")
        clear_context()
        get_logger("ctx.test").info("hi")
        out = json.loads(capsys.readouterr().out.strip())
        assert "optimization_id" not in out["Attributes"]

    def test_context_empty_after_clear(self) -> None:
        bind_context(optimization_id="opt-abc")
        clear_context()
        assert get_context() == {}


class TestThreadIsolation:
    def test_context_isolated_per_thread(self) -> None:
        """Context set in one thread must not be visible in another."""
        bind_context(optimization_id="main-thread")

        results: dict[str, Any] = {}

        def worker() -> None:
            results["ctx"] = get_context()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # Main thread still has its context
        assert get_context()["optimization_id"] == "main-thread"
        # Spawned thread starts with an empty context
        assert results["ctx"] == {}


class TestKafkaHeaders:
    def test_round_trip(self) -> None:
        bind_context(optimization_id="opt-abc", preprocess_id="pre-xyz")
        headers = context_to_headers()
        decoded = headers_to_context(headers)
        assert decoded == {"optimization_id": "opt-abc", "preprocess_id": "pre-xyz"}

    def test_empty_context_produces_no_headers(self) -> None:
        assert context_to_headers() == []

    def test_none_headers_returns_empty_dict(self) -> None:
        assert headers_to_context(None) == {}

    def test_empty_headers_returns_empty_dict(self) -> None:
        assert headers_to_context([]) == {}

    def test_non_prefixed_headers_are_ignored(self) -> None:
        headers = [("x-toop-optimization_id", b"opt-abc"), ("x-other-key", b"ignored")]
        decoded = headers_to_context(headers)
        assert decoded == {"optimization_id": "opt-abc"}
        assert "x-other-key" not in decoded

    def test_string_values_decoded_as_str(self) -> None:
        headers = [("x-toop-preprocess_id", b"pre-xyz")]
        decoded = headers_to_context(headers)
        assert isinstance(decoded["preprocess_id"], str)

    def test_bind_context_from_headers(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Simulate the consumer side: extract headers → bind_context → log."""
        incoming_headers = [("x-toop-preprocess_id", b"pre-abc"), ("x-toop-optimization_id", b"opt-xyz")]
        bind_context(**headers_to_context(incoming_headers))
        get_logger("kafka.test").info("received command")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["Attributes"]["preprocess_id"] == "pre-abc"
        assert out["Attributes"]["optimization_id"] == "opt-xyz"
