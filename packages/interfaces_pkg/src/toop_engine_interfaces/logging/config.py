# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Structlog configuration for the OTel Logs Data Model.

The logging level is set to 20 (INFO) by default, but can be overridden by setting the LOG_LEVEL env var.
Valid level names or numbers can be found in: https://docs.python.org/3/library/logging.html#levels.
"""

import functools
import os
import sys
import threading
import time
import traceback
from collections.abc import MutableMapping
from typing import Any

import structlog
from toop_engine_interfaces.logging.context import _add_job_context

_SEVERITY_MAP: dict[str, str] = {
    "trace": "TRACE",
    "debug": "DEBUG",
    "info": "INFO",
    "warning": "WARN",
    "warn": "WARN",
    "error": "ERROR",
    "critical": "FATAL",
    "fatal": "FATAL",
}

_configured = False

# Type alias matching structlog's EventDict
_EventDict = MutableMapping[str, Any]


# --- processors ---


def _add_timestamp(_logger: Any, _method: str, event_dict: _EventDict) -> _EventDict:  # noqa: ANN401
    """Inject nanosecond Unix epoch timestamp."""
    event_dict["_ts"] = time.time_ns()
    return event_dict


@functools.cache
def _service_attrs() -> dict[str, str]:
    """Read OTel resource attributes from env vars once per process."""
    attrs: dict[str, str] = {
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "unknown"),
        "service.namespace": os.environ.get("OTEL_SERVICE_NAMESPACE", "unknown"),
        "deployment.environment": os.environ.get("DEPLOYMENT_ENV", "unknown"),
        "zone.name": os.environ.get("ZONE_NAME", "unknown"),
    }
    for key, env in [
        ("service.version", "OTEL_SERVICE_VERSION"),
        ("k8s.pod.name", "K8S_POD_NAME"),
        ("k8s.node.name", "K8S_NODE_NAME"),
    ]:
        if value := os.environ.get(env):
            attrs[key] = value
    return attrs


def _add_service_attributes(_logger: Any, _method: str, event_dict: _EventDict) -> _EventDict:  # noqa: ANN401
    """Inject service resource attributes. Caller-bound keys take precedence."""
    for key, value in _service_attrs().items():
        event_dict.setdefault(key, value)
    return event_dict


def _add_thread_name(_logger: Any, _method: str, event_dict: _EventDict) -> _EventDict:  # noqa: ANN401
    """Inject the current thread name."""
    event_dict.setdefault("thread.name", threading.current_thread().name)
    return event_dict


def _format_exception(_logger: Any, _method: str, event_dict: _EventDict) -> _EventDict:  # noqa: ANN401
    """Convert exc_info into OTel exception.* attributes."""
    exc_info = event_dict.pop("exc_info", None)
    if exc_info is None:
        return event_dict
    if exc_info is True:
        exc_info = sys.exc_info()
    elif isinstance(exc_info, BaseException):
        exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
    exc_type, exc_value, exc_tb = exc_info
    if exc_type is not None and exc_value is not None:
        event_dict["exception.type"] = f"{exc_type.__module__}.{exc_type.__qualname__}"
        event_dict["exception.message"] = str(exc_value)
        event_dict["exception.stacktrace"] = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    return event_dict


def _to_otel_shape(_logger: Any, _method: str, event_dict: _EventDict) -> dict[str, Any]:  # noqa: ANN401
    """Restructure event dict into the OTel Logs Data Model. Must be last before JSONRenderer."""
    body = event_dict.pop("event", "")
    timestamp = event_dict.pop("_ts", 0)
    severity = _SEVERITY_MAP.get(str(event_dict.pop("level", "info")).lower(), "INFO")
    return {
        "Timestamp": timestamp,
        "SeverityText": severity,
        "Body": body,
        "TraceId": "",
        "SpanId": "",
        "Attributes": dict(event_dict),
    }


# --- configuration ---


def configure(log_level: int = 20) -> None:
    """Configure structlog with the OTel processor chain. Idempotent.

    Find common log levels here: https://docs.python.org/3/library/logging.html#levels

    Args:
        log_level: The minimum log level to emit, as an int or valid level name string. Default is 20 (INFO).

    """
    global _configured  # noqa: PLW0603
    if _configured:
        return
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            _add_timestamp,
            _add_service_attributes,
            _add_thread_name,
            _format_exception,
            _add_job_context,
            _to_otel_shape,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True
