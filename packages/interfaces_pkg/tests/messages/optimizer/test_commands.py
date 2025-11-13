import pytest
from toop_engine_interfaces.messages.optimiser_ac_dc_commons_factory import (
    Framework,
    OptimizerType,
)
from toop_engine_interfaces.messages.optimiser_ac_dc_commons_factory import (
    create_descriptor_def as DescriptorDef,
)
from toop_engine_interfaces.messages.optimiser_ac_dc_commons_factory import (
    create_grid_file as GridFile,
)
from toop_engine_interfaces.messages.optimiser_commands_factory import (
    create_command as Command,
)
from toop_engine_interfaces.messages.optimiser_commands_factory import (
    create_start_optimization_command as StartOptimizationCommand,
)
from toop_engine_interfaces.messages.optimiser_dc_params_factory import (
    create_batched_me_parameters as BatchedMEParameters,
)
from toop_engine_interfaces.messages.optimiser_dc_params_factory import (
    create_target_metric as TargetMetric,
)
from toop_engine_interfaces.messages.optimiser_heartbeats_factory import (
    create_heartbeat as Heartbeat,
)
from toop_engine_interfaces.messages.optimiser_heartbeats_factory import (
    create_heartbeat_union as HeartbeatUnion,
)
from toop_engine_interfaces.messages.optimiser_heartbeats_factory import (
    create_idle_heartbeat as IdleHeartbeat,
)
from toop_engine_interfaces.messages.optimiser_results_factory import (
    create_optimization_stopped_result as OptimizationStoppedResult,
)
from toop_engine_interfaces.messages.optimiser_results_factory import (
    create_result as Result,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_commands_pb2 import Command as PbCommand
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_heartbeats_pb2 import Heartbeat as PbHeartbeat
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_results_pb2 import Result as PbResult


def test_infer_missing_observed_metrics():
    # Test case where target metrics are already in observed metrics
    params = BatchedMEParameters(
        target_metrics=(TargetMetric("overload_energy_n_1", 1.0),),
        observed_metrics=("overload_energy_n_1", "max_flow_n_0"),
    )
    assert "overload_energy_n_1" in params.observed_metrics

    # Test case where target metrics are not in observed metrics
    params = BatchedMEParameters(
        target_metrics=(TargetMetric("overload_energy_n_1", 1.0),),
        observed_metrics=("max_flow_n_0",),
    )
    assert "overload_energy_n_1" in params.observed_metrics

    # Test case where descriptor metrics are not in observed metrics
    params = BatchedMEParameters(
        target_metrics=(TargetMetric("overload_energy_n_1", 1.0),),
        observed_metrics=("max_flow_n_0",),
        me_descriptors=(DescriptorDef(metric="underload_energy_n_0", num_cells=1),),
    )
    assert "underload_energy_n_0" in params.observed_metrics

    # Test case where both target and descriptor metrics are not in observed metrics
    params = BatchedMEParameters(
        target_metrics=(TargetMetric("overload_energy_n_1", 1.0),),
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

    serialized_command = command.SerializeToString()
    # del serialized["command"]["dc_params"]["ga_config"]["substation_split_prob"]
    # serialized_data = json.dumps(serialized)

    parsed_command = PbCommand()
    parsed_command.ParseFromString(serialized_command)

    assert isinstance(parsed_command, PbCommand)

    result = Result(
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        instance_id="joghurt",
        result=OptimizationStoppedResult(reason="error"),
    )
    serialized = result.SerializeToString()
    parsed_result = PbResult()
    parsed_result.ParseFromString(serialized)
    assert isinstance(parsed_result, PbResult)

    heartbeat = Heartbeat(
        optimizer_type=OptimizerType.DC, instance_id="joghurt", message=HeartbeatUnion(message=IdleHeartbeat())
    )
    serialized_heartbeat = heartbeat.SerializeToString()
    parsed_heartbeat = PbHeartbeat()
    parsed_heartbeat.ParseFromString(serialized_heartbeat)
    assert isinstance(parsed_heartbeat, PbHeartbeat)
