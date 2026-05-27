# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import logging
import time
from pathlib import Path
from uuid import uuid4

import jax.numpy as jnp
import numpy as np
import pytest
import ray
import structlog
from confluent_kafka import Consumer, Producer
from fsspec import AbstractFileSystem
from fsspec.implementations.dirfs import DirFileSystem
from jax_dataclasses import replace
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_dc_solver.example_grids import (
    complex_grid_battery_hvdc_svc_3w_trafo_data_folder,
    parallel_pst_data_folder,
    three_node_pst_example_folder_powsybl,
)
from toop_engine_dc_solver.jax.aggregate_results import aggregate_to_metric_batched
from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_computations import default_topology
from toop_engine_dc_solver.jax.types import NodalInjOptimResults, NodalInjStartOptions
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.network_data import extract_action_set, extract_nminus1_definition
from toop_engine_grid_helpers.powsybl.loadflow_parameters import SINGLE_SLACK
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message
from toop_engine_topology_optimizer.ac.scoring_functions import compute_metrics_single_timestep
from toop_engine_topology_optimizer.ac.worker import Args as ACArgs
from toop_engine_topology_optimizer.ac.worker import main as ac_main
from toop_engine_topology_optimizer.dc.worker.worker import Args as DCArgs
from toop_engine_topology_optimizer.dc.worker.worker import main as dc_main
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters, ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commands import Command, ShutdownCommand, StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef, Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    LoadflowSolverParameters,
)
from toop_engine_topology_optimizer.interfaces.messages.results import (
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    TopologyPushResult,
)

from .fake_kafka import FakeConsumer, FakeConsumerEmptyException, FakeProducer

logger = structlog.get_logger()
# Ensure that tests using Kafka are not run in parallel with each other
pytestmark = pytest.mark.xdist_group("kafka")


def dc_main_wrapper(args: DCArgs, processed_gridfile_fs: AbstractFileSystem) -> None:
    instance_id = str(uuid4())
    command_consumer = LongRunningKafkaConsumer(
        topic=args.optimizer_command_topic,
        group_id="dc_optimizer",
        bootstrap_servers=args.kafka_broker,
        client_id=instance_id,
    )
    producer = Producer(
        {
            "bootstrap.servers": args.kafka_broker,
            "client.id": instance_id,
            "log_level": 2,
        }
    )

    dc_main(args, processed_gridfile_fs, producer, command_consumer)


def ac_main_wrapper(
    args: ACArgs,
    processed_gridfile_fs: AbstractFileSystem,
    loadflow_result_fs: AbstractFileSystem,
) -> None:
    instance_id = str(uuid4())
    command_consumer = LongRunningKafkaConsumer(
        topic=args.optimizer_command_topic,
        group_id="ac_optimizer",
        bootstrap_servers=args.kafka_broker,
        client_id=instance_id,
    )
    result_consumer = LongRunningKafkaConsumer(
        topic=args.optimizer_results_topic,
        group_id="ac_optimizer_results",
        bootstrap_servers=args.kafka_broker,
        client_id=instance_id,
    )
    producer = Producer(
        {
            "bootstrap.servers": args.kafka_broker,
            "client.id": instance_id,
            "log_level": 2,
        }
    )
    ac_main(args, loadflow_result_fs, processed_gridfile_fs, producer, command_consumer, result_consumer)


@ray.remote
def launch_dc_worker(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    processed_gridfile_fs: AbstractFileSystem,
):
    logging.basicConfig(level=logging.INFO)
    try:
        dc_main_wrapper(
            DCArgs(
                optimizer_command_topic=kafka_command_topic,
                optimizer_heartbeat_topic=kafka_heartbeat_topic,
                optimizer_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
                instance_id="dc_worker",
            ),
            processed_gridfile_fs=processed_gridfile_fs,
        )
    except SystemExit:
        # This is expected when the worker receives a shutdown command
        logger.info("DC worker stopped")
        pass


@ray.remote
def launch_ac_worker(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    processed_gridfile_fs: AbstractFileSystem,
    loadflow_result_fs: AbstractFileSystem,
):
    logging.basicConfig(level=logging.INFO)
    print("Starting AC worker")
    try:
        ac_main_wrapper(
            ACArgs(
                optimizer_command_topic=kafka_command_topic,
                optimizer_heartbeat_topic=kafka_heartbeat_topic,
                optimizer_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
                instance_id="ac_worker",
            ),
            loadflow_result_fs=loadflow_result_fs,
            processed_gridfile_fs=processed_gridfile_fs,
        )
    except SystemExit:
        # This is expected when the worker receives a shutdown command
        logger.info("AC worker stopped")
        pass


# TODO: set to 200, once the xdist_group is run on a dedicated runner
@pytest.mark.skip(reason="This test is currently flaky, should be fixed and re-enabled")
def test_ac_dc_integration(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    grid_folder: Path,
    loadflow_result_folder: Path,
) -> None:
    # Start the ray runtime
    ray.init(num_cpus=4)

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    try:
        ac_future = launch_ac_worker.remote(
            kafka_command_topic=kafka_command_topic,
            kafka_heartbeat_topic=kafka_heartbeat_topic,
            kafka_results_topic=kafka_results_topic,
            kafka_connection_str=kafka_connection_str,
            processed_gridfile_fs=processed_gridfile_fs,
            loadflow_result_fs=loadflow_result_fs,
        )
        dc_future = launch_dc_worker.remote(
            kafka_command_topic=kafka_command_topic,
            kafka_heartbeat_topic=kafka_heartbeat_topic,
            kafka_results_topic=kafka_results_topic,
            kafka_connection_str=kafka_connection_str,
            processed_gridfile_fs=processed_gridfile_fs,
        )

        grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
        ac_parameters = ACOptimizerParameters(
            ga_config=ACGAParameters(
                runtime_seconds=50,
                pull_prob=1.0,
                reconnect_prob=0.0,
                close_coupler_prob=0.0,
                seed=42,
                enable_ac_rejection=False,
            )
        )
        dc_parameters = DCOptimizerParameters(
            ga_config=BatchedMEParameters(iterations_per_epoch=2, runtime_seconds=30),
            loadflow_solver_config=LoadflowSolverParameters(
                batch_size=16,
            ),
        )
        start_command = Command(
            command=StartOptimizationCommand(
                ac_params=ac_parameters,
                dc_params=dc_parameters,
                grid_files=grid_files,
                optimization_id="test",
            )
        )

        producer = Producer({"bootstrap.servers": kafka_connection_str, "log_level": 2})
        producer.produce(kafka_command_topic, value=serialize_message(start_command.model_dump_json()))
        producer.flush()

        # This is the runtime of the AC worker
        time.sleep(50)

        consumer = Consumer(
            {
                "bootstrap.servers": kafka_connection_str,
                "group.id": "integration_test",
                "auto.offset.reset": "earliest",
                "log_level": 2,
            }
        )
        consumer.subscribe([kafka_results_topic])

        ac_converged = False
        dc_converged = False
        ac_topo_push = False
        dc_topo_push = False
        split_topo_push = False

        result_history = []
        while message := consumer.poll(timeout=10.0):
            result = Result.model_validate_json(deserialize_message(message.value()))
            result_history.append(result)
            if isinstance(result.result, OptimizationStoppedResult):
                assert result.result.reason == "converged", f"{result}"
                if result.optimizer_type == OptimizerType.AC:
                    ac_converged = True
                elif result.optimizer_type == OptimizerType.DC:
                    dc_converged = True
            elif isinstance(result.result, TopologyPushResult):
                if result.optimizer_type == OptimizerType.AC:
                    ac_topo_push = True
                elif result.optimizer_type == OptimizerType.DC:
                    dc_topo_push = True
                for strategy in result.result.strategies:
                    if len(strategy.timesteps[0].actions):
                        split_topo_push = True
                        break

            if ac_converged and dc_converged:
                break

        logger.info(f"{[type(result.result) for result in result_history]}")
        assert result_history
        assert ac_converged
        assert dc_converged
        assert dc_topo_push
        assert split_topo_push
        assert ac_topo_push

        shutdown_command = Command(command=ShutdownCommand())
        producer.produce(kafka_command_topic, value=serialize_message(shutdown_command.model_dump_json()))
        producer.flush()

        # Give everyone a chance to shutdown
        ray.get(ac_future)
        ray.get(dc_future)

    finally:
        ray.shutdown()


@pytest.mark.timeout(100)
def test_ac_dc_integration_sequential(grid_folder: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
    ac_parameters = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=20,
            remaining_loadflow_wait_seconds=5,
            seed=42,
            enable_ac_rejection=False,
        )
    )
    dc_parameters = DCOptimizerParameters(
        ga_config=BatchedMEParameters(iterations_per_epoch=2, runtime_seconds=20),
        loadflow_solver_config=LoadflowSolverParameters(
            batch_size=16,
        ),
    )
    start_command = Command(
        command=StartOptimizationCommand(
            ac_params=ac_parameters,
            dc_params=dc_parameters,
            grid_files=grid_files,
            optimization_id="test",
        )
    )

    command_consumer = FakeConsumer(
        messages={"commands": [serialize_message(start_command.model_dump_json())]}, kill_on_empty=True
    )
    producer = FakeProducer()

    with pytest.raises(FakeConsumerEmptyException):
        dc_main(
            DCArgs(
                optimizer_command_topic="commands",
                optimizer_heartbeat_topic="heartbeats",
                optimizer_results_topic="results",
            ),
            processed_gridfile_fs=DirFileSystem(str(grid_folder)),
            producer=producer,
            command_consumer=command_consumer,
        )

    assert "results" in producer.messages
    assert len(producer.messages["results"]) > 0
    # First one should be a OptimizationStartedResult
    first_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][0]))
    assert isinstance(first_msg, Result)
    assert isinstance(first_msg.result, OptimizationStartedResult)

    # There should be at least one TopologyPushResult before the OptimizationStoppedResult
    second_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][1]))
    assert isinstance(second_msg, Result)
    assert isinstance(second_msg.result, TopologyPushResult)
    assert len(second_msg.result.strategies) > 0
    assert len(second_msg.result.strategies[0].timesteps) > 0

    last_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][-1]))
    assert isinstance(last_msg, Result)
    assert isinstance(last_msg.result, OptimizationStoppedResult)
    assert last_msg.result.reason == "converged"

    result_consumer = FakeConsumer(
        messages={
            "results": producer.messages["results"],
        }
    )
    command_consumer = FakeConsumer(
        messages={"commands": [serialize_message(start_command.model_dump_json())]}, kill_on_empty=True
    )

    producer = FakeProducer()

    with pytest.raises(FakeConsumerEmptyException):
        ac_main(
            ACArgs(
                optimizer_command_topic="commands",
                optimizer_heartbeat_topic="heartbeats",
                optimizer_results_topic="results",
            ),
            loadflow_result_fs=DirFileSystem(str(tmp_path_factory.mktemp("loadflow_results"))),
            processed_gridfile_fs=DirFileSystem(str(grid_folder)),
            producer=producer,
            command_consumer=command_consumer,
            result_consumer=result_consumer,
        )

    assert "results" in producer.messages
    assert len(producer.messages["results"]) > 0

    # First one should be a OptimizationStartedResult
    first_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][0]))
    assert isinstance(first_msg, Result)
    assert isinstance(first_msg.result, OptimizationStartedResult)

    # There should be at least one TopologyPushResult before the OptimizationStoppedResult
    second_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][1]))
    assert isinstance(second_msg, Result)
    assert isinstance(second_msg.result, TopologyPushResult)
    assert len(second_msg.result.strategies) > 0
    assert len(second_msg.result.strategies[0].timesteps) > 0

    last_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][-1]))
    assert isinstance(last_msg, Result)
    assert isinstance(last_msg.result, OptimizationStoppedResult)
    assert last_msg.result.reason == "converged"


@pytest.mark.timeout(100)
def test_ac_dc_integration_psts(tmp_path_factory: pytest.TempPathFactory) -> None:
    grid_folder = tmp_path_factory.mktemp("grid_folder")
    (grid_folder / "threenode").mkdir()
    three_node_pst_example_folder_powsybl(grid_folder / "threenode")
    load_grid(
        data_folder_dirfs=DirFileSystem(str(grid_folder / "threenode")),
        parameters=PreprocessParameters(),
    )

    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="threenode")]
    ac_parameters = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=20, pull_prob=1.0, reconnect_prob=0.0, close_coupler_prob=0.0, seed=42, enable_ac_rejection=False
        )
    )
    dc_parameters = DCOptimizerParameters(
        ga_config=BatchedMEParameters(
            iterations_per_epoch=10,
            runtime_seconds=20,
            substation_split_prob=0,
            n_worst_contingencies=2,
            pst_mutation_sigma=3.0,
            enable_nodal_inj_optim=True,
            target_metrics=(("overload_energy_n_1", 1.0),),
            observed_metrics=("overload_energy_n_1", "split_subs"),
            me_descriptors=(DescriptorDef(metric="split_subs", num_cells=2),),
        ),
        loadflow_solver_config=LoadflowSolverParameters(
            batch_size=16,
            max_num_splits=1,
            max_num_disconnections=0,
        ),
    )
    start_command = Command(
        command=StartOptimizationCommand(
            ac_params=ac_parameters,
            dc_params=dc_parameters,
            grid_files=grid_files,
            optimization_id="test",
        )
    )

    command_consumer = FakeConsumer(
        messages={"commands": [serialize_message(start_command.model_dump_json())]}, kill_on_empty=True
    )
    producer = FakeProducer()

    with pytest.raises(FakeConsumerEmptyException):
        dc_main(
            DCArgs(
                optimizer_command_topic="commands",
                optimizer_heartbeat_topic="heartbeats",
                optimizer_results_topic="results",
            ),
            processed_gridfile_fs=DirFileSystem(str(grid_folder)),
            producer=producer,
            command_consumer=command_consumer,
        )

    assert "results" in producer.messages
    assert len(producer.messages["results"]) > 0
    # First one should be a OptimizationStartedResult
    # And we should have overloads in the grid
    first_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][0]))
    assert isinstance(first_msg, Result)
    assert isinstance(first_msg.result, OptimizationStartedResult)
    assert first_msg.result.initial_topology.timesteps[0].pst_setpoints is None
    assert first_msg.result.initial_topology.timesteps[0].metrics.fitness < 0

    # There should be at least one TopologyPushResult before the OptimizationStoppedResult
    second_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][1]))
    assert isinstance(second_msg, Result)
    assert isinstance(second_msg.result, TopologyPushResult)
    assert len(second_msg.result.strategies) > 0
    assert len(second_msg.result.strategies[0].timesteps) > 0
    assert second_msg.result.strategies[0].timesteps[0].pst_setpoints is not None

    last_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][-1]))
    assert isinstance(last_msg, Result)
    assert isinstance(last_msg.result, OptimizationStoppedResult)
    assert last_msg.result.reason == "converged"

    result_consumer = FakeConsumer(
        messages={
            "results": producer.messages["results"],
        }
    )
    command_consumer = FakeConsumer(
        messages={"commands": [serialize_message(start_command.model_dump_json())]}, kill_on_empty=True
    )

    producer = FakeProducer()

    with pytest.raises(FakeConsumerEmptyException):
        ac_main(
            ACArgs(
                optimizer_command_topic="commands",
                optimizer_heartbeat_topic="heartbeats",
                optimizer_results_topic="results",
            ),
            loadflow_result_fs=DirFileSystem(str(tmp_path_factory.mktemp("loadflow_results"))),
            processed_gridfile_fs=DirFileSystem(str(grid_folder)),
            producer=producer,
            command_consumer=command_consumer,
            result_consumer=result_consumer,
        )

    assert "results" in producer.messages
    assert len(producer.messages["results"]) > 0

    # First one should be a OptimizationStartedResult
    first_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][0]))
    assert isinstance(first_msg, Result)
    assert isinstance(first_msg.result, OptimizationStartedResult)

    # There should be at least one TopologyPushResult before the OptimizationStoppedResult
    second_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][1]))
    assert isinstance(second_msg, Result)
    assert isinstance(second_msg.result, TopologyPushResult)
    assert len(second_msg.result.strategies) > 0
    assert len(second_msg.result.strategies[0].timesteps) > 0

    last_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][-1]))
    assert isinstance(last_msg, Result)
    assert isinstance(last_msg.result, OptimizationStoppedResult)
    assert last_msg.result.reason == "converged"


def test_dc_optimizer_fitness_ac_validation_fitness_3pst(tmp_path_factory: pytest.TempPathFactory) -> None:
    fixture_name = "three_node_pst_example_data_folder"

    grid_folder = tmp_path_factory.mktemp("grid_folder")
    fixture_folder = grid_folder / fixture_name
    fixture_folder.mkdir()
    three_node_pst_example_folder_powsybl(fixture_folder)
    _, static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(fixture_folder)),
        parameters=PreprocessParameters(),
    )

    dynamic_information = static_information.dynamic_information
    nodal_injection_information = dynamic_information.nodal_injection_information
    assert nodal_injection_information is not None, "Grid should have controllable PSTs for this test"
    assert dynamic_information.action_set is not None, "Grid should have an action set for metric aggregation"

    solver_config = replace(static_information.solver_config, batch_size_bsdf=1)
    topology_batch = default_topology(solver_config)
    actions: list[int] = []
    disconnections: list[int] = []
    n_random_cases = 10

    runner = PowsyblRunner()
    runner.load_base_grid(fixture_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_action_set(extract_action_set(network_data))
    nminus1_definition = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_definition)
    base_case_id = nminus1_definition.base_case.id if nminus1_definition.base_case is not None else None

    pst_n_taps = np.asarray(nodal_injection_information.pst_n_taps, dtype=int)
    initial_rel_taps = np.asarray(nodal_injection_information.starting_tap_idx, dtype=int)
    grid_model_low_tap = np.asarray(nodal_injection_information.grid_model_low_tap, dtype=int)
    possible_changed_states = int(np.prod(pst_n_taps, dtype=np.int64)) - 1
    assert possible_changed_states >= n_random_cases, "Need at least 10 distinct changed PST states for this test"

    rng = np.random.default_rng(42)
    sampled_rel_taps: list[np.ndarray] = []
    seen_taps: set[tuple[int, ...]] = set()
    while len(sampled_rel_taps) < n_random_cases:
        candidate = rng.integers(low=np.zeros_like(pst_n_taps), high=pst_n_taps)
        candidate_key = tuple(int(value) for value in candidate.tolist())
        if np.array_equal(candidate, initial_rel_taps) or candidate_key in seen_taps:
            continue
        sampled_rel_taps.append(candidate)
        seen_taps.add(candidate_key)

    solver_metrics = []
    runner_metrics = []
    absolute_taps_list = []
    for sample_index, rel_taps in enumerate(sampled_rel_taps):
        solver_results, success_dc = compute_symmetric_batch(
            topology_batch=topology_batch,
            disconnection_batch=None,
            injections=None,
            nodal_inj_start_options=NodalInjStartOptions(
                previous_results=NodalInjOptimResults(pst_tap_idx=jnp.asarray(rel_taps, dtype=int)[None, None, :]),
                precision_percent=jnp.array(0.0),
            ),
            dynamic_information=dynamic_information,
            solver_config=solver_config,
        )
        assert np.all(success_dc), f"DC solver failed for sample {sample_index} with relative taps {rel_taps.tolist()}"

        solver_metric = float(
            np.asarray(
                aggregate_to_metric_batched(
                    lf_res_batch=solver_results,
                    branch_limits=dynamic_information.branch_limits,
                    reassignment_distance=dynamic_information.action_set.reassignment_distance,
                    n_relevant_subs=dynamic_information.n_sub_relevant,
                    metric="overload_energy_n_1",
                    initial_pst_tap_idx=nodal_injection_information.starting_tap_idx,
                )
            )[0]
        )
        solver_metrics.append(solver_metric)

        absolute_taps = (rel_taps + grid_model_low_tap).tolist()
        absolute_taps_list.append(absolute_taps)
        dc_loadflow = runner.run_loadflow_single_timestep(
            actions=actions,
            disconnections=disconnections,
            pst_setpoints=absolute_taps,
            method="dc",
        )
        dc_metrics_validation = compute_metrics_single_timestep(
            actions=actions,
            disconnections=disconnections,
            loadflow=dc_loadflow,
            additional_info=None,
            base_case_id=base_case_id,
        )
        runner_metric = float(dc_metrics_validation.extra_scores["overload_energy_n_1"])
        runner_metrics.append(runner_metric)

    assert np.allclose(solver_metrics, runner_metrics, atol=1e-5, rtol=0.0), (
        f"DC solver versus DC validation failed. Taps: {absolute_taps_list}"
    )


def test_dc_optimizer_fitness_ac_validation_fitness_parallel_pst(tmp_path_factory: pytest.TempPathFactory) -> None:
    fixture_name = "three_node_pst_example_data_folder"

    grid_folder = tmp_path_factory.mktemp("grid_folder")
    fixture_folder = grid_folder / fixture_name
    fixture_folder.mkdir()
    _ = parallel_pst_data_folder(fixture_folder)
    _, static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(fixture_folder)),
        parameters=PreprocessParameters(),
    )

    dynamic_information = static_information.dynamic_information
    nodal_injection_information = dynamic_information.nodal_injection_information
    assert nodal_injection_information is not None, "Grid should have controllable PSTs for this test"
    assert dynamic_information.action_set is not None, "Grid should have an action set for metric aggregation"

    solver_config = replace(static_information.solver_config, batch_size_bsdf=1)
    topology_batch = default_topology(solver_config)
    actions: list[int] = []
    disconnections: list[int] = []
    n_random_cases = 10

    runner = PowsyblRunner()
    runner.load_base_grid(fixture_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_action_set(extract_action_set(network_data))
    nminus1_definition = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_definition)
    base_case_id = nminus1_definition.base_case.id if nminus1_definition.base_case is not None else None

    pst_n_taps = np.asarray(nodal_injection_information.pst_n_taps, dtype=int)
    initial_rel_taps = np.asarray(nodal_injection_information.starting_tap_idx, dtype=int)
    grid_model_low_tap = np.asarray(nodal_injection_information.grid_model_low_tap, dtype=int)
    possible_changed_states = int(np.prod(pst_n_taps, dtype=np.int64)) - 1
    assert possible_changed_states >= n_random_cases, "Need at least 10 distinct changed PST states for this test"

    rng = np.random.default_rng(42)
    sampled_rel_taps: list[np.ndarray] = []
    seen_taps: set[tuple[int, ...]] = set()
    while len(sampled_rel_taps) < n_random_cases:
        candidate = rng.integers(low=np.zeros_like(pst_n_taps), high=pst_n_taps)
        candidate_key = tuple(int(value) for value in candidate.tolist())
        if np.array_equal(candidate, initial_rel_taps) or candidate_key in seen_taps:
            continue
        sampled_rel_taps.append(candidate)
        seen_taps.add(candidate_key)

    solver_metrics = []
    runner_metrics = []
    absolute_taps_list = []
    for sample_index, rel_taps in enumerate(sampled_rel_taps):
        solver_results, success_dc = compute_symmetric_batch(
            topology_batch=topology_batch,
            disconnection_batch=None,
            injections=None,
            nodal_inj_start_options=NodalInjStartOptions(
                previous_results=NodalInjOptimResults(pst_tap_idx=jnp.asarray(rel_taps, dtype=int)[None, None, :]),
                precision_percent=jnp.array(0.0),
            ),
            dynamic_information=dynamic_information,
            solver_config=solver_config,
        )
        assert np.all(success_dc), f"DC solver failed for sample {sample_index} with relative taps {rel_taps.tolist()}"

        solver_metric = float(
            np.asarray(
                aggregate_to_metric_batched(
                    lf_res_batch=solver_results,
                    branch_limits=dynamic_information.branch_limits,
                    reassignment_distance=dynamic_information.action_set.reassignment_distance,
                    n_relevant_subs=dynamic_information.n_sub_relevant,
                    metric="overload_energy_n_1",
                    initial_pst_tap_idx=nodal_injection_information.starting_tap_idx,
                )
            )[0]
        )
        solver_metrics.append(solver_metric)

        absolute_taps = (rel_taps + grid_model_low_tap).tolist()
        absolute_taps_list.append(absolute_taps)
        dc_loadflow = runner.run_loadflow_single_timestep(
            actions=actions,
            disconnections=disconnections,
            pst_setpoints=absolute_taps,
            method="dc",
        )
        dc_metrics_validation = compute_metrics_single_timestep(
            actions=actions,
            disconnections=disconnections,
            loadflow=dc_loadflow,
            additional_info=None,
            base_case_id=base_case_id,
        )
        runner_metric = float(dc_metrics_validation.extra_scores["overload_energy_n_1"])
        runner_metrics.append(runner_metric)

    assert np.allclose(solver_metrics, runner_metrics, atol=1e-5, rtol=0.0), (
        f"DC solver versus DC validation failed. Taps: {absolute_taps_list}"
    )


def test_dc_optimizer_fitness_ac_validation_fitness_complex(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test that the DC solver's fitness metric matches the validation's fitness metric in DC on a more complex grid with PSTs.

    Warning: DC computation of Powsybl runner needs to use `SINGLE_SLACK` to match DC solver results.
    """
    fixture_name = "complex_grid_data_folder"

    grid_folder = tmp_path_factory.mktemp(fixture_name)
    network_data = complex_grid_battery_hvdc_svc_3w_trafo_data_folder(grid_folder, np.array([True, True]))
    static_information = load_static_information(grid_folder / PREPROCESSING_PATHS["static_information_file_path"])

    di = static_information.dynamic_information
    nodal_injection_information = di.nodal_injection_information
    assert nodal_injection_information is not None, "Grid should have controllable PSTs for this test"
    assert di.action_set is not None, "Grid should have an action set for metric aggregation"

    solver_config = replace(static_information.solver_config, batch_size_bsdf=1)
    topology_batch = default_topology(solver_config)
    actions: list[int] = []
    disconnections: list[int] = []
    n_random_cases = 10

    solver_res_no_pst, success_dc_no_pst = compute_symmetric_batch(
        topology_batch=default_topology(solver_config),
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=None,
        dynamic_information=di,
        solver_config=solver_config,
    )
    assert np.all(success_dc_no_pst), "DC solver without PST changes should succeed"

    n_0_no_pst = -solver_res_no_pst.n_0_matrix[0, 0]
    n_1_no_pst = -solver_res_no_pst.n_1_matrix[0, 0]

    solver_n_1 = float(
        np.asarray(
            aggregate_to_metric_batched(
                lf_res_batch=solver_res_no_pst,
                branch_limits=di.branch_limits,
                reassignment_distance=di.action_set.reassignment_distance,
                n_relevant_subs=di.n_sub_relevant,
                metric="overload_energy_n_0",
                initial_pst_tap_idx=None,
            )
        )[0]
    )
    solver_basecase_n_1_metric = float(
        np.asarray(
            aggregate_to_metric_batched(
                lf_res_batch=solver_res_no_pst,
                branch_limits=di.branch_limits,
                reassignment_distance=di.action_set.reassignment_distance,
                n_relevant_subs=di.n_sub_relevant,
                metric="overload_energy_n_1",
                initial_pst_tap_idx=None,
            )
        )[0]
    )

    # Runner for validation - needs to use SINGLE_SLACK to match DC solver computations
    runner = PowsyblRunner(lf_params=SINGLE_SLACK)  # Required to match DC solver results.
    runner.load_base_grid(grid_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_action_set(extract_action_set(network_data))
    nminus1_definition = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_definition)
    base_case_id = nminus1_definition.base_case.id if nminus1_definition.base_case is not None else None

    runner_res_no_pst = runner.run_dc_loadflow(
        actions=actions,
        disconnections=disconnections,
        pst_setpoints=None,
    )
    # Matrices
    n_0_runner_pst, n_1_runner_pst, success_ref = extract_solver_matrices_polars(runner_res_no_pst, nminus1_definition, 0)
    assert np.all(success_ref), "Pypowsybl runner without PST changes should succeed"
    assert np.allclose(n_0_no_pst, n_0_runner_pst, atol=1e-5, rtol=0.0), (
        f"N-0 matrix mismatch between DC solver and runner. Solver: {n_0_no_pst}, Runner: {n_0_runner_pst}"
    )
    assert np.allclose(n_1_no_pst, n_1_runner_pst, atol=1e-5, rtol=0.0), (
        f"N-1 matrix mismatch between DC solver and runner. Solver: {n_1_no_pst}, Runner: {n_1_runner_pst}"
    )

    runner_basecase_n_1_metrics = compute_metrics_single_timestep(
        actions=actions,
        disconnections=disconnections,
        loadflow=runner_res_no_pst,
        additional_info=None,
        base_case_id=base_case_id,
    )
    runner_basecase_n_1_metric = float(runner_basecase_n_1_metrics.extra_scores["overload_energy_n_1"])

    assert np.allclose(solver_basecase_n_1_metric, runner_basecase_n_1_metric, atol=1e-5, rtol=0.0), (
        f"DC solver versus DC validation failed on initial state. Solver: {solver_basecase_n_1_metric}, Runner: {runner_basecase_n_1_metric}"
    )

    pst_n_taps = np.asarray(nodal_injection_information.pst_n_taps, dtype=int)
    initial_rel_taps = np.asarray(nodal_injection_information.starting_tap_idx, dtype=int)
    grid_model_low_tap = np.asarray(nodal_injection_information.grid_model_low_tap, dtype=int)
    possible_changed_states = int(np.prod(pst_n_taps, dtype=np.int64)) - 1
    assert possible_changed_states >= n_random_cases, "Need at least 10 distinct changed PST states for this test"

    rng = np.random.default_rng(42)
    sampled_rel_taps: list[np.ndarray] = []
    seen_taps: set[tuple[int, ...]] = set()
    while len(sampled_rel_taps) < n_random_cases:
        candidate = rng.integers(low=np.zeros_like(pst_n_taps), high=pst_n_taps)
        candidate_key = tuple(int(value) for value in candidate.tolist())
        if np.array_equal(candidate, initial_rel_taps) or candidate_key in seen_taps:
            continue
        sampled_rel_taps.append(candidate)
        seen_taps.add(candidate_key)

    solver_metrics = []
    runner_metrics = []
    absolute_taps_list = []
    for sample_index, rel_taps in enumerate(sampled_rel_taps):
        solver_results, success_dc = compute_symmetric_batch(
            topology_batch=topology_batch,
            disconnection_batch=None,
            injections=None,
            nodal_inj_start_options=NodalInjStartOptions(
                previous_results=NodalInjOptimResults(pst_tap_idx=jnp.asarray(rel_taps, dtype=int)[None, None, :]),
                precision_percent=jnp.array(0.0),
            ),
            dynamic_information=di,
            solver_config=solver_config,
        )
        assert np.all(success_dc), f"DC solver failed for sample {sample_index} with relative taps {rel_taps.tolist()}"

        solver_metric = float(
            np.asarray(
                aggregate_to_metric_batched(
                    lf_res_batch=solver_results,
                    branch_limits=di.branch_limits,
                    reassignment_distance=di.action_set.reassignment_distance,
                    n_relevant_subs=di.n_sub_relevant,
                    metric="overload_energy_n_1",
                    initial_pst_tap_idx=nodal_injection_information.starting_tap_idx,
                )
            )[0]
        )
        solver_metrics.append(solver_metric)

        absolute_taps = (rel_taps + grid_model_low_tap).tolist()
        absolute_taps_list.append(absolute_taps)
        dc_loadflow = runner.run_loadflow_single_timestep(
            actions=actions,
            disconnections=disconnections,
            pst_setpoints=absolute_taps,
            method="dc",
        )
        dc_metrics_validation = compute_metrics_single_timestep(
            actions=actions,
            disconnections=disconnections,
            loadflow=dc_loadflow,
            additional_info=None,
            base_case_id=base_case_id,
        )
        runner_metric = float(dc_metrics_validation.extra_scores["overload_energy_n_1"])
        runner_metrics.append(runner_metric)

    assert np.allclose(solver_metrics, runner_metrics, atol=1e-5, rtol=0.0), (
        f"DC solver versus DC validation failed. Taps: {absolute_taps_list}"
    )
