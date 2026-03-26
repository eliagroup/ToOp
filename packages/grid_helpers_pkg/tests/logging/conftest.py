# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Pytest fixtures for logging tests."""

from typing import Iterator

import pytest
import structlog
import toop_engine_grid_helpers.logging.config as log_config
from toop_engine_grid_helpers.logging.context import clear_context


@pytest.fixture(autouse=True)
def reset_structlog() -> Iterator[None]:
    """Reset structlog, the configuration flag, and job context between tests."""
    structlog.reset_defaults()
    log_config._configured = False
    log_config._service_attrs.cache_clear()
    clear_context()
    yield
    structlog.reset_defaults()
    log_config._configured = False
    log_config._service_attrs.cache_clear()
    clear_context()
