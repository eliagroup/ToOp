# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os
import pickle

import jax
import jax.numpy as jnp
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.inputs import (
    deserialize_static_information,
    load_static_information,
    save_static_information,
    serialize_static_information,
    validate_static_information,
)
from toop_engine_dc_solver.jax.types import (
    ActionSet,
    BranchLimits,
    InjectionComputations,
    NonRelBBOutageData,
    StaticInformation,
    TopoVectBranchComputations,
)
from toop_engine_dc_solver.jax.utils import HashableArrayWrapper


def assert_static_information(a: StaticInformation, b: StaticInformation) -> None:
    assert jax.tree_util.tree_map(
        lambda a, b: jnp.array_equal(a, b, equal_nan=True) if isinstance(a, jnp.ndarray) else a == b,
        a.dynamic_information,
        b.dynamic_information,
    )

    for key in a.dynamic_information.__dataclass_fields__.keys():
        if key == "branch_limits":
            assert isinstance(getattr(b.dynamic_information, key), BranchLimits)
            assert isinstance(getattr(a.dynamic_information, key), BranchLimits)
        elif key == "action_set":
            assert isinstance(getattr(b.dynamic_information, key), ActionSet)
            assert isinstance(getattr(a.dynamic_information, key), ActionSet)
            assert a.dynamic_information.action_set == b.dynamic_information.action_set
        elif key == "non_rel_bb_outage_data":
            if a.dynamic_information.non_rel_bb_outage_data is not None:
                assert isinstance(getattr(b.dynamic_information, key), NonRelBBOutageData)
                assert isinstance(getattr(a.dynamic_information, key), NonRelBBOutageData)
                assert a.dynamic_information.non_rel_bb_outage_data == b.dynamic_information.non_rel_bb_outage_data
        elif key == "nodal_injection_information":
            if a.dynamic_information.nodal_injection_information is not None:
                assert getattr(b.dynamic_information, key) is not None
                assert getattr(a.dynamic_information, key) is not None
                assert a.dynamic_information.nodal_injection_information == b.dynamic_information.nodal_injection_information
        else:
            assert jnp.array_equal(
                getattr(a.dynamic_information, key),
                getattr(b.dynamic_information, key),
                equal_nan=True,
            )

    for key in a.solver_config.__dataclass_fields__.keys():
        assert isinstance(getattr(b.solver_config, key), type(getattr(a.solver_config, key)))
        if isinstance(getattr(a.solver_config, key), HashableArrayWrapper):
            assert jnp.array_equal(
                getattr(a.solver_config, key).val,
                getattr(b.solver_config, key).val,
                equal_nan=True,
            )
        else:
            assert getattr(a.solver_config, key) == getattr(b.solver_config, key)


def test_load_save(
    tmp_path: str,
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    save_static_information(os.path.join(tmp_path, "test.hdf5"), static_information)
    loaded = load_static_information(os.path.join(tmp_path, "test.hdf5"))
    validate_static_information(loaded)

    assert_static_information(static_information, loaded)

    # Test with multi-outages and branch actions
    static_information = replace(
        static_information,
        dynamic_information=replace(
            static_information.dynamic_information,
            multi_outage_branches=[jnp.array([[0, 8, 12]], dtype=int)],
            multi_outage_nodes=[jnp.zeros((1, 0), dtype=int)],
        ),
    )
    save_static_information(os.path.join(tmp_path, "test2.hdf5"), static_information)
    loaded = load_static_information(os.path.join(tmp_path, "test2.hdf5"))
    validate_static_information(loaded)
    assert jnp.array_equal(
        static_information.dynamic_information.multi_outage_branches[0],
        loaded.dynamic_information.multi_outage_branches[0],
    )
    assert jnp.array_equal(
        static_information.dynamic_information.multi_outage_nodes[0],
        loaded.dynamic_information.multi_outage_nodes[0],
    )
    assert jnp.array_equal(
        static_information.dynamic_information.action_set.branch_actions,
        loaded.dynamic_information.action_set.branch_actions,
    )
    assert jnp.array_equal(
        static_information.dynamic_information.action_set.n_actions_per_sub,
        loaded.dynamic_information.action_set.n_actions_per_sub,
    )


def test_pickle_static_information(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    pickled = pickle.dumps(static_information)
    loaded = pickle.loads(pickled)
    validate_static_information(loaded)

    assert_static_information(static_information, loaded)


def test_serialize_static_information(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    serialized = serialize_static_information(static_information)
    loaded = deserialize_static_information(serialized)
    validate_static_information(loaded)

    assert_static_information(static_information, loaded)
