# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for structured logging configuration."""

import json
import logging

import pytest
import structlog
from toop_engine_interfaces.structlog_config import configure_structured_logging


def _read_log(capsys: pytest.CaptureFixture[str]) -> str:
    """Helper to read captured log output."""
    return capsys.readouterr().out.strip()


def _read_log_json(capsys: pytest.CaptureFixture[str]) -> dict:
    """Helper to read captured log output and parse as JSON."""
    return json.loads(
        s=_read_log(capsys),
    )


def test_processors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_structured_logging()

    structlog.get_logger().info("hello")

    log = _read_log_json(capsys)

    assert set(log.keys()) == {
        "event",
        "level",
        "logger",
        "timestamp",
        "thread.name",
    }


def test_log_level(
    capsys: pytest.CaptureFixture[str],
) -> None:

    configure_structured_logging(log_level=logging.INFO)

    logger = structlog.get_logger()

    msg = "This is a test message"  # Should not appear in output, config set to INFO

    logger.debug(msg)

    assert _read_log(capsys) == ""

    logger.info(msg)

    assert msg in _read_log(capsys)


def test_exc(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_structured_logging()

    logger = structlog.get_logger()

    msg = "caught exception"
    try:
        raise ValueError("bad value")
    except ValueError:
        structlog.get_logger().exception(msg)

    log = _read_log_json(capsys)

    assert (
        log["exception.type"] == "builtins.ValueError"
        and log["exception.message"] == "bad value"
        and "Traceback" in log["exception.stacktrace"]
        and log["event"] == msg
    )


def test_no_exception_fields_on_normal_log(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_structured_logging()

    structlog.get_logger().info("all good")

    log = _read_log_json(capsys)

    exception_fields = {"exception.type", "exception.message", "exception.stacktrace"}

    assert exception_fields - set(log.keys()) == exception_fields


def test_key_appears_in_attributes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_structured_logging()

    key = "optimization_id"
    value = "opt-abc"

    with structlog.contextvars.bound_contextvars(**{key: value}):
        structlog.get_logger().info("hi")
        log = _read_log_json(capsys)

    assert log[key] == value
