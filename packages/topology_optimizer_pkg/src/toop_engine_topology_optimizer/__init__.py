# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
import os

if os.getenv("DEBUG"):
    from beartype.claw import beartype_this_package
    from jaxtyping import install_import_hook

    install_import_hook(["toop_engine_dc_solver.jax", "toop_engine_topology_optimizer.dc"], "beartype.beartype")
    beartype_this_package()
