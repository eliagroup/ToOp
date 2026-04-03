# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for structured logging configuration."""

import json
import logging
import textwrap
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import docker
import pytest
import structlog
from opentelemetry.instrumentation.auto_instrumentation import initialize
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


def test_otel_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate logs are exported through OTEL auto-instrumentation into collector output.

    This explicitly tests are logs emitted with structlog and our configuration are properly picked up by the OTEL logging instrumentation and exported to the collector.

    In particular, it verifies that the OTEL log record includes:
        -  context variables and extra variables as attributes
        -  the log message as the body
    """

    def _wait_for_collector_port(
        container: docker.models.containers.Container,
        container_port: str,
        timeout_s: float = 20.0,
    ) -> int:
        """Wait until the collector exposes a given container port and return host port."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            container.reload()
            port_binding = container.attrs.get("NetworkSettings", {}).get("Ports", {}).get(container_port)
            if port_binding and port_binding[0].get("HostPort"):
                return int(port_binding[0]["HostPort"])
            time.sleep(0.2)
        raise TimeoutError(f"OTEL collector did not expose port {container_port} in time")

    def _wait_for_collector_ready(
        port: int,
        timeout_s: float = 60.0,
    ) -> None:
        """Wait until collector health endpoint reports ready."""
        deadline = time.monotonic() + timeout_s
        url = f"http://127.0.0.1:{port}/"
        while time.monotonic() < deadline:
            try:
                with urlopen(url, timeout=1) as response:  # noqa: S310
                    if response.status == 200:
                        return
            except (URLError, OSError):
                # Collector can transiently reset/close sockets while booting.
                pass
            time.sleep(0.2)
        raise TimeoutError("OTEL collector health endpoint did not become ready in time")

    def _get_otel_log(
        log_file: Path,
        marker: str,
        timeout_s: float = 60.0,
    ) -> dict:
        """Wait until the collector output file contains a specific marker string."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if log_file.exists():
                content = log_file.read_text(encoding="utf-8")
                if marker in content:
                    return json.loads(content)
            time.sleep(0.2)
        raise TimeoutError(f"Collector output did not contain marker: {marker}")

    def _get_otel_scoped_log(
        scope_logs: list[dict],
        scope: str,
    ) -> dict:
        for log in scope_logs:
            if log["scope"]["name"] == scope:
                return log
        raise ValueError(f"No log found for scope: {scope}")

    def _assert_otel_attribute_value(
        attributes: list[dict],
        key: str,
        value: str,
    ) -> None:
        for attr in attributes:
            if attr["key"] == key:
                assert attr["value"]["stringValue"] == value, (
                    f"Expected value '{value}' for attribute '{key}', but found '{attr['value']['stringValue']}'"
                )
                return
        raise ValueError(f"Attribute with key '{key}' not found")

    container_name = "test_otel_collector"

    docker_client = docker.from_env()

    # Kill and remove any existing container with the same name to avoid conflicts
    for container in docker_client.containers.list(all=True):
        if container.name in container_name:
            container = docker_client.containers.get(container.id)
            container.remove(v=True, force=True)

    # Configure OTEL collector with OTLP receiver and file exporter
    collector_config = tmp_path / "otel-collector.yaml"
    collector_output = tmp_path / "collector-logs.json"

    health_port = 13133
    otlp_port = 4318

    collector_config.write_text(
        textwrap.dedent(
            f"""
                        receivers:
                            otlp:
                                protocols:
                                    http:
                                        endpoint: 0.0.0.0:{otlp_port}

                        extensions:
                            health_check:
                                endpoint: 0.0.0.0:{health_port}

                        processors:
                            batch: {{}}

                        exporters:
                            file:
                                path: /tmp/otel/collector-logs.json

                        service:
                            extensions: [health_check]
                            pipelines:
                                logs:
                                    receivers: [otlp]
                                    processors: [batch]
                                    exporters: [file]
                    """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        # Start OTEL collector in a docker container with the config and wait until it's ready to receive logs
        container = docker_client.containers.run(
            image="otel/opentelemetry-collector-contrib:0.149.0",
            command=["--config=/etc/otelcol/config.yaml"],
            name=container_name,
            detach=True,
            ports={
                "4318/tcp": None,  # Map to random host port, we will discover it later
                "13133/tcp": None,  # health check endpoint
            },
            volumes={
                str(collector_config): {
                    "bind": "/etc/otelcol/config.yaml",
                    "mode": "ro",
                },
                str(tmp_path): {"bind": "/tmp/otel", "mode": "rw"},
            },
        )

        # Wait for the collector to expose the ports and get the host ports mapped to the container ports

        host_otlp_port, host_health_port = (
            _wait_for_collector_port(
                container=container,
                container_port=port,
            )
            for port in (f"{otlp_port}/tcp", f"{health_port}/tcp")
        )

        # Poll the health endpoint until the collector is ready to receive logs
        _wait_for_collector_ready(
            port=host_health_port,
        )

        # Configure environment variables for OTEL auto-instrumentation to send logs to our collector
        for key, value in {
            "OTEL_SERVICE_NAME": "interfaces-otel-e2e",
            "OTEL_TRACES_EXPORTER": "none",
            "OTEL_METRICS_EXPORTER": "none",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED": "true",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{host_otlp_port}",
        }.items():
            monkeypatch.setenv(
                name=key,
                value=value,
            )

        # Set testing variables
        logger_name = "otel-test"
        msg = "otel-integration-message"

        # Variables to be included in the log record context and as extra attributes
        context_vars = {"optimization_id": "opt-otel-e2e"}

        # Extra variables passed directly to the log call, should also appear as attributes
        extra_vars = {"extra_key": "extra-value"}

        # Initialize OTEL auto-instrumentation -> We normally call the application with `otel-instrument`,
        # but since we are running the test directly, we need to initialize it manually here
        initialize()

        # Configure structured logging with our custom configuration
        configure_structured_logging()

        # Log a message with structlog, including context variables and extra variables
        with structlog.contextvars.bound_contextvars(
            **context_vars,
        ):
            structlog.get_logger(logger_name).info(
                msg,
                **extra_vars,
            )

        # Wait for the collector to export the logs and read the output
        otel_log = _get_otel_log(
            log_file=collector_output,
            marker=msg,
        )

        # Get the resource log for our service
        # We expect only one
        assert len(otel_log["resourceLogs"]) == 1, "Expected exactly one resource log in collector output"

        resource_log = otel_log["resourceLogs"].pop()

        # Assert there is scope log with our logger name, which indicates the log record is from our logger and not some internal OTEL logs
        assert logger_name in (log["scope"]["name"] for log in resource_log["scopeLogs"])

        # Get scoped log
        scoped_log = _get_otel_scoped_log(
            scope_logs=resource_log["scopeLogs"],
            scope=logger_name,
        )

        # We expect only one log record from our logger
        assert len(scoped_log["logRecords"]) == 1
        log = scoped_log["logRecords"][0]

        # Validate log body
        assert log["body"]["stringValue"] == msg

        # Validate attribute keys
        expected_attributes = context_vars | extra_vars

        # Get the keys of the attributes in the log record
        attribute_keys = {attr["key"] for attr in log["attributes"]}

        # Assert our custom attribute is present
        for key in expected_attributes.keys():
            assert key in attribute_keys, f"Expected attribute '{key}' not found in log record attributes"

        # Validate attribute values

        for key, value in expected_attributes.items():
            _assert_otel_attribute_value(
                attributes=log["attributes"],
                key=key,
                value=value,
            )
    finally:
        if container is not None:
            container.remove(
                v=True,
                force=True,
            )
