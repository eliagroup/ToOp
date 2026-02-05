# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines an abstract runner class.

An abstract runner class encapsulates a loadflow solver and a postprocessing pipeline.
There are currently implementations for pandapower and powsybl as runners.
This module helps with postprocessing of optimizer results.
"""

from beartype.claw import beartype_this_package

# Make sure beartype_this_package is the only imported module
beartype_only_non_dunder_import = all(d.startswith("_") or d == "beartype_this_package" for d in dir())
if beartype_only_non_dunder_import:
    beartype_this_package()  # Leave this at the top. Otherwise the modules imported before wont be beartyped
else:
    raise ImportError(
        "Please make sure that beartype_this_package is the only imported module before calling beartype_this_package"
        "Please check the import statements."
    )

from .abstract_runner import (
    AbstractLoadflowRunner,
)
from .postprocess_pandapower import (
    PandapowerRunner,
)
from .postprocess_powsybl import (
    PowsyblRunner,
)

__all__ = ["AbstractLoadflowRunner", "PandapowerRunner", "PowsyblRunner"]
