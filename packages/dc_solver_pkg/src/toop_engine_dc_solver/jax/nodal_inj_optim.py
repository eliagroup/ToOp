# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for nodal injection optimization start options."""

import jax.numpy as jnp
from toop_engine_dc_solver.jax.types import NodalInjOptimResults, NodalInjStartOptions


def make_start_options(
    old_res: NodalInjOptimResults | None,
) -> NodalInjStartOptions | None:
    """Create start options for nodal injection optimization from previous results.

    Returns None if old_res is None, indicating no previous results to use as starting point.
    """
    if old_res is None:
        return None
    return NodalInjStartOptions(
        previous_results=old_res,
        precision_percent=jnp.array(1.0),  # TODO
    )
