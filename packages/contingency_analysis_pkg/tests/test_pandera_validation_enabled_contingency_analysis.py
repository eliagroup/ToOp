# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandas as pd
import pandera.pandas as pa
import pandera.typing as pat
import pytest


class _IntColumnSchema(pa.DataFrameModel):
    value: pat.Series[int]


def test_pandera_validation_is_enabled_in_tests() -> None:
    """Guard that pandera validation is active during the test run.

    The test suite must validate schemas, even if production deactivates validation via
    PANDERA_VALIDATION_ENABLED=False. If someone disables validation in the test configuration (conftest.py), this
    test fails because the invalid frame below is no longer rejected.
    """
    invalid_df = pd.DataFrame({"value": ["not an int"]})
    with pytest.raises(pa.errors.SchemaError):
        _IntColumnSchema.validate(invalid_df)
