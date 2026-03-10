# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
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
