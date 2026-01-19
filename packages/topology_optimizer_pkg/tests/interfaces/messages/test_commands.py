# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json

import pytest
from toop_engine_topology_optimizer.interfaces.messages.commands import (
    Command,
    StartOptimizationCommand,
)
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DescriptorDef,
)
from toop_engine_topology_optimizer.interfaces.messages.heartbeats import Heartbeat, IdleHeartbeat
from toop_engine_topology_optimizer.interfaces.messages.results import OptimizationStoppedResult, Result


def test_infer_missing_observed_metrics():
    # Test case where target metrics are already in observed metrics
    params = BatchedMEParameters(
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=("overload_energy_n_1", "max_flow_n_0"),
    )
    assert "overload_energy_n_1" in params.observed_metrics

    # Test case where target metrics are not in observed metrics
    params = BatchedMEParameters(
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=("max_flow_n_0",),
    )
    assert "overload_energy_n_1" in params.observed_metrics

    # Test case where descriptor metrics are not in observed metrics
    params = BatchedMEParameters(
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=("max_flow_n_0",),
        me_descriptors=(DescriptorDef(metric="underload_energy_n_0", num_cells=1),),
    )
    assert "underload_energy_n_0" in params.observed_metrics

    # Test case where both target and descriptor metrics are not in observed metrics
    params = BatchedMEParameters(
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=("max_flow_n_0",),
        me_descriptors=(DescriptorDef(metric="underload_energy_n_0", num_cells=1),),
    )
    assert "overload_energy_n_1" in params.observed_metrics
    assert "underload_energy_n_0" in params.observed_metrics


def test_deserialization():
    with pytest.raises(ValueError):
        StartOptimizationCommand(
            optimization_id="test",
            grid_files=[GridFile(framework=Framework.PANDAPOWER, grid_folder="test", coupling="tight")],
        )

    command = Command(
        command=StartOptimizationCommand(
            optimization_id="test",
            grid_files=[
                GridFile(
                    framework=Framework.PANDAPOWER,
                    grid_folder="test",
                )
            ],
        )
    )

    serialized = command.model_dump()
    del serialized["command"]["dc_params"]["ga_config"]["substation_split_prob"]
    json_serialized = json.dumps(serialized)

    parsed = Command.model_validate_json(json_serialized)
    assert isinstance(parsed.command, StartOptimizationCommand)

    result = Result(
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        instance_id="joghurt",
        result=OptimizationStoppedResult(reason="error"),
    )
    serialized = result.model_dump_json()
    parsed = Result.model_validate_json(serialized)
    assert isinstance(parsed.result, OptimizationStoppedResult)

    heartbeat = Heartbeat(optimizer_type=OptimizerType.DC, instance_id="joghurt", message=IdleHeartbeat())
    serialized = heartbeat.model_dump_json()
    parsed = Heartbeat.model_validate_json(serialized)
    assert isinstance(parsed.message, IdleHeartbeat)
