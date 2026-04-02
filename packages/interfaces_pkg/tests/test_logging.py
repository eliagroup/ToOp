import json
import logging

import pytest
import structlog
from toop_engine_interfaces.logging import configure_structure_logging


def test_processors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_structure_logging()
    logger = structlog.get_logger()

    logger.info("hello")

    out = json.loads(capsys.readouterr().out.strip())

    assert set(out.keys()) == {
        "event",
        "level",
        "logger",
        "timestamp",
        "thread.name",
    }


def test_log_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_structure_logging(log_level=logging.INFO)

    msg = "This is a test message"

    logger = structlog.get_logger()

    logger.debug(msg)

    assert capsys.readouterr().out == ""

    logger.info(msg)

    assert json.loads(capsys.readouterr().out.strip())["event"] == msg


def test_exc(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_structure_logging()
    logger = structlog.get_logger()
    msg = "caught exception"
    try:
        raise ValueError("bad value")
    except ValueError:
        logger.exception(msg)

    out = json.loads(capsys.readouterr().out.strip())

    assert (
        out["exception.type"] == "builtins.ValueError"
        and out["exception.message"] == "bad value"
        and "Traceback" in out["exception.stacktrace"]
        and out["event"] == msg
    )


def test_no_exception_fields_on_normal_log(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_structure_logging()
    logger = structlog.get_logger()
    logger.info("all good")
    out = json.loads(capsys.readouterr().out.strip())

    exception_fields = {"exception.type", "exception.message", "exception.stacktrace"}

    assert exception_fields - set(out.keys()) == exception_fields


def test_key_appears_in_attributes(capsys: pytest.CaptureFixture[str]) -> None:
    configure_structure_logging()

    key = "optimization_id"
    value = "opt-abc"

    with structlog.contextvars.bound_contextvars(**{key: value}):
        structlog.get_logger().info("hi")
        out = json.loads(capsys.readouterr().out.strip())

    assert out[key] == value
