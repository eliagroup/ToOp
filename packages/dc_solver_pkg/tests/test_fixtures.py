# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax.numpy as jnp
from toop_engine_dc_solver.jax.topology_computations import (
    convert_topo_to_action_set_index,
    convert_topo_to_action_set_index_jittable,
    is_in_action_set,
)
from toop_engine_dc_solver.jax.types import InjectionComputations, StaticInformation, TopoVectBranchComputations


def test_jax_inputs_in_action_set(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, injections, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information

    assert is_in_action_set(topologies, dynamic_information.action_set).all()

    action_topos, act_set = convert_topo_to_action_set_index(
        topologies, dynamic_information.action_set, extend_action_set=True
    )
    assert act_set == dynamic_information.action_set

    action_topos_2 = convert_topo_to_action_set_index_jittable(topologies, dynamic_information.action_set)

    assert jnp.array_equal(action_topos.action, action_topos_2.action)
