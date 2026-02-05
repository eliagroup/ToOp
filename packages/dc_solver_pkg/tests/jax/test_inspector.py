# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from jax import numpy as jnp
from toop_engine_dc_solver.jax.inspector import inspect_topology
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
)


def test_inspect_topology(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    computations, candidates, static_information = jax_inputs

    topo_id = 12
    bsdf_res, disconnection_res, lodf_res = inspect_topology(
        computations.topologies[topo_id],
        computations.sub_ids[topo_id],
        None,
        static_information,
    )

    assert bsdf_res.shape == (computations.topologies.shape[1],)
    assert disconnection_res is None
    assert lodf_res.shape == (static_information.dynamic_information.branches_to_fail.shape[0],)

    assert jnp.all(bsdf_res)
    assert jnp.all(lodf_res)
