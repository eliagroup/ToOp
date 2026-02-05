# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import sys

from beartype.claw import beartype_this_package
from pandera import Int

beartype_this_package()

if sys.platform == "win32":
    Int.check = lambda self, pandera_dtype, data_container=None: isinstance(pandera_dtype, Int)  # noqa: ARG005
