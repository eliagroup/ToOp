# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The pandapower helpers are usable without the optional ``pypowsybl`` extra installed."""

import subprocess
import sys
import textwrap

import pytest

# Blocking ``pypowsybl`` in ``sys.modules`` makes any ``import pypowsybl...`` raise
# ``ImportError`` even when the package is installed, so this also holds in a
# development environment that has both extras.
_IMPORT_WITHOUT_PYPOWSYBL = textwrap.dedent(
    """
    import sys
    sys.modules["pypowsybl"] = None
    import {module}
    """
)


@pytest.mark.parametrize(
    "module",
    [
        "toop_engine_grid_helpers.pandapower",
        "toop_engine_grid_helpers.pandapower.slack_allocation",
        "toop_engine_grid_helpers.pandapower.outage_group",
        "toop_engine_grid_helpers.asset_topology_helpers",
    ],
)
def test_pandapower_modules_import_without_pypowsybl(module: str) -> None:
    result = subprocess.run(  # noqa: S603 - fixed interpreter and literal module names
        [sys.executable, "-c", _IMPORT_WITHOUT_PYPOWSYBL.format(module=module)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
