# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Testing fixtures."""

import pandera
import pytest
import structlog

# Tests always validate, regardless of PANDERA_VALIDATION_ENABLED
pandera.config.reset_config_context(
    pandera.config.PanderaConfig(validation_enabled=True, validation_depth=pandera.config.ValidationDepth.SCHEMA_AND_DATA)
)


@pytest.fixture(autouse=True)
def reset_structlog() -> None:
    """Reset structlog between tests."""
    structlog.reset_defaults()
