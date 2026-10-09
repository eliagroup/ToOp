# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The pandapower N-1 engine is importable without the optional ``pypowsybl`` extra installed."""

import subprocess
import sys
import textwrap

# Blocking ``pypowsybl`` in ``sys.modules`` makes any ``import pypowsybl...`` raise
# ``ImportError`` even when the package is installed, so this also holds in a
# development environment that has both extras.
_IMPORT_WITHOUT_PYPOWSYBL = textwrap.dedent(
    """
    import sys
    sys.modules["pypowsybl"] = None
    import toop_engine_contingency_analysis.pandapower
    """
)


def test_pandapower_engine_imports_without_pypowsybl() -> None:
    result = subprocess.run(  # noqa: S603 - fixed interpreter and literal code
        [sys.executable, "-c", _IMPORT_WITHOUT_PYPOWSYBL],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
