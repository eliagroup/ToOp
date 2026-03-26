# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Job-scoped logging context backed by a Python ContextVar."""

from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

_job_context: ContextVar[dict[str, str] | None] = ContextVar("_job_context", default=None)

_EventDict = MutableMapping[str, Any]


def bind_context(**kwargs: str) -> None:
    """Merge keyword arguments into the current logging context.

    Keys already present are overwritten; unrelated keys are preserved.
    The context is stored in a ContextVar so it is isolated per thread and async task.

    Parameters
    ----------
    **kwargs : str
        Arbitrary key-value pairs to add or update, e.g. ``optimization_id="opt-abc"``.
    """
    ctx = _job_context.get()
    if ctx is None:
        ctx = {}
    _job_context.set({**ctx, **kwargs})


def clear_context() -> None:
    """Remove all keys from the current logging context."""
    _job_context.set({})


def get_context() -> dict[str, str]:
    """Return a snapshot of the current logging context.

    Returns
    -------
    dict[str, str]
        A copy of the current context dict.
    """
    ctx = _job_context.get()
    if ctx is None:
        return {}
    return dict(ctx)


def _add_job_context(_logger: Any, _method: str, event_dict: _EventDict) -> _EventDict:  # noqa: ANN401
    """Structlog processor: inject the current job context into every log record.

    Keys already set by the caller take precedence (via ``setdefault``).
    """
    ctx = _job_context.get()
    if ctx is not None:
        for key, value in ctx.items():
            event_dict.setdefault(key, value)
    return event_dict
