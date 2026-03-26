# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for propagating logging context across Kafka message boundaries via headers."""

from toop_engine_grid_helpers.logging.context import _job_context

_PREFIX = "x-toop-"


def context_to_headers() -> list[tuple[str, str | bytes]]:
    """Encode the current logging context as a list of Kafka message headers.

    Returns
    -------
    list[tuple[str, str | bytes]]
        One ``(header-name, value)`` pair per context key.  The header name is
        ``x-toop-<key>``; the value is UTF-8-encoded.  Returns an empty list
        when the context is empty.
    """
    headers: list[tuple[str, str | bytes]] = [(_PREFIX + k, v.encode()) for k, v in _job_context.get().items()]
    return headers


def headers_to_context(headers: list[tuple[str, bytes]] | None) -> dict[str, str]:
    """Decode ``x-toop-*`` Kafka headers into a context dict.

    Parameters
    ----------
    headers : list[tuple[str, bytes]] | None
        Raw headers as returned by ``confluent_kafka.Message.headers()``.

    Returns
    -------
    dict[str, str]
        Context keys without the ``x-toop-`` prefix, values decoded from UTF-8.
        Returns an empty dict when *headers* is ``None`` or empty.
    """
    if not headers:
        return {}
    return {k[len(_PREFIX) :]: (v.decode() if isinstance(v, bytes) else v) for k, v in headers if k.startswith(_PREFIX)}
