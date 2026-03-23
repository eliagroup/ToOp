# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Factory for creating OTel-structured loggers."""

from typing import Any

import structlog
from toop_engine_grid_helpers.logging.config import configure


def get_logger(name: str) -> Any:  # noqa: ANN401
    """Return a structlog BoundLogger emitting OTel-structured JSON.

    Configures structlog on first call (idempotent). The logger.name attribute
    is pre-bound so every emitted record identifies its origin.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
        A bound structlog logger ready for use.

    Example::

        from toop_engine_grid_helpers.logging.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Grid loaded", n_buses=42)
    """
    configure()
    return structlog.get_logger(name).bind(**{"logger.name": name})
