# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The AC worker that listens to the kafka topics, organizes optimization runs, etc."""

import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from uuid import uuid4

import structlog
from beartype.typing import Callable
from confluent_kafka import Producer
from fsspec import AbstractFileSystem
from sqlmodel import Session
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message
from toop_engine_topology_optimizer.ac.listener import poll_results_topic
from toop_engine_topology_optimizer.ac.optimizer import (
    AcNotConvergedError,
    evaluate_remaining_contingencies,
    initialize_optimization,
    process_fast_failing_results,
    run_fast_failing_epoch,
    wait_for_first_dc_results,
)
from toop_engine_topology_optimizer.ac.storage import create_session, scrub_db
from toop_engine_topology_optimizer.ac.summary import write_summary
from toop_engine_topology_optimizer.ac.types import OptimizerData
from toop_engine_topology_optimizer.dc.worker.worker import Args as DCArgs
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commands import Command, ShutdownCommand, StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.commons import GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.heartbeats import (
    Heartbeat,
    HeartbeatUnion,
    IdleHeartbeat,
    OptimizationStartedHeartbeat,
    OptimizationStatsHeartbeat,
    StartupHeartbeat,
)
from toop_engine_topology_optimizer.interfaces.messages.results import (
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    ResultUnion,
    Strategy,
)

logger = structlog.get_logger(__name__)


class Args(DCArgs):
    """Command line arguments for the AC worker.

    Mostly the same as the DC worker except for an additional loadflow results folder
    """


@dataclass
class WorkerData:
    """Data that is stored across optimization runs"""

    command_consumer: LongRunningKafkaConsumer
    """A kafka consumer listening in for optimization commands"""

    result_consumer: LongRunningKafkaConsumer
    """A kafka consumer listening on the results topic, constantly writing results to the database. This is polled
    both during the optimization and idle loop to keep the database up to date."""

    producer: Producer
    """A kafka producer to send heartbeats and results"""

    db: Session
    """An initialized database session to an in-memory sqlite database."""


def initialize_optimization_run(
    ac_params: ACOptimizerParameters,
    grid_file: GridFile,
    worker_data: WorkerData,
    send_result_fn: Callable[[ResultUnion], None],
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    optimization_id: str,
    loadflow_result_fs: AbstractFileSystem,
    processed_gridfile_fs: AbstractFileSystem,
) -> tuple[OptimizerData, Strategy]:
    """Initialize the AC optimization and wait for the first DC results.

    Parameters are identical to `optimization_loop` plus the bound logger.

    Returns
    -------
    tuple[OptimizerData, Strategy]
        The initialized optimizer data and the initial topology message.
    """
    send_heartbeat_fn(
        OptimizationStartedHeartbeat(
            optimization_id=optimization_id,
        )
    )
    optimizer_data, initial_topology = initialize_optimization(
        session=worker_data.db,
        params=ac_params,
        optimization_id=optimization_id,
        grid_file=grid_file,
        loadflow_result_fs=loadflow_result_fs,
        processed_gridfile_fs=processed_gridfile_fs,
    )
    wait_for_first_dc_results(
        results_consumer=worker_data.result_consumer,
        session=worker_data.db,
        max_wait_time=ac_params.ga_config.max_initial_wait_seconds,
        optimization_id=optimization_id,
        heartbeat_fn=partial(
            send_heartbeat_fn,
            OptimizationStatsHeartbeat(
                optimization_id=optimization_id,
                wall_time=0,
                iteration=0,
                num_branch_topologies_tried=0,
                num_injection_topologies_tried=0,
            ),
        ),
    )
    send_result_fn(
        OptimizationStartedResult(
            initial_topology=initial_topology,
        )
    )
    return optimizer_data, initial_topology


def run_optimization_epochs(
    ac_params: ACOptimizerParameters,
    optimizer_data: OptimizerData,
    worker_data: WorkerData,
    send_result_fn: Callable[[ResultUnion], None],
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    optimization_id: str,
) -> None:
    """Run the iterative AC optimization phase.

    Parameters
    ----------
    ac_params : ACOptimizerParameters
        The parameters for the AC optimizer
    optimizer_data : OptimizerData
        The initialized optimizer data containing the curried functions and database session
    worker_data : WorkerData
        The dataclass with the results consumer and database
    send_result_fn : Callable[[ResultUnion], None]
        The function to send results
    send_heartbeat_fn : Callable[[HeartbeatUnion], None]
        The function to send heartbeats
    optimization_id : str
        The ID of the optimization run

    Returns
    -------
    None

    """
    start_time = time.time()
    last_full_run = start_time
    survivor_batch_size = ac_params.ga_config.runner_processes
    epoch = 1
    evaluated_topologies = 0
    survivor_topologies = []
    survivor_early_results = []

    while True:
        with structlog.contextvars.bound_contextvars(epoch=epoch):
            added_topos, _ = poll_results_topic(
                db=optimizer_data.session, consumer=worker_data.result_consumer, first_poll=epoch == 1
            )
            logger.debug("Imported topologies from result stream", imported_topology_count=len(added_topos))

            # Even though the separation into fast-failing and remaining contingencies is not strictly necessary when
            # enable_ac_rejection is False, we still run the N-1 analysis in two steps to keep the logic similar to the
            # default enable_ac_rejection=True case and avoid having too many if statements in the code.
            topologies, worst_k_results = run_fast_failing_epoch(
                optimizer_data=optimizer_data,
            )
            if ac_params.ga_config.enable_ac_rejection:
                success_topologies, success_early_stop_results = process_fast_failing_results(
                    optimizer_data=optimizer_data,
                    topologies=topologies,
                    fast_failing_results=worst_k_results,
                    send_result_fn=send_result_fn,
                    epoch=epoch,
                )
                evaluated_topologies += len(topologies)
                survivor_topologies.extend(success_topologies)
                survivor_early_results.extend(success_early_stop_results)
            else:
                survivor_topologies.extend(topologies)
                survivor_early_results.extend(worst_k_results)
                evaluated_topologies += len(topologies)

            enough_survivors = len(survivor_topologies) >= survivor_batch_size
            runtime_exceeded_since_last_full_run = (
                time.time() - last_full_run
            ) > ac_params.ga_config.remaining_loadflow_wait_seconds
            if enough_survivors or (runtime_exceeded_since_last_full_run and len(survivor_topologies) > 0):
                logger.debug(
                    f"Collected {len(survivor_topologies)} survivor topologies, running remaining contingencies evaluation"
                )
                evaluate_remaining_contingencies(
                    send_result_fn,
                    optimizer_data,
                    epoch,
                    survivor_topologies[:survivor_batch_size],
                    survivor_early_results[:survivor_batch_size],
                )
                survivor_topologies = survivor_topologies[survivor_batch_size:]
                survivor_early_results = survivor_early_results[survivor_batch_size:]
                epoch += 1
                last_full_run = time.time()

            send_heartbeat_fn(
                OptimizationStatsHeartbeat(
                    optimization_id=optimization_id,
                    wall_time=time.time() - start_time,
                    iteration=epoch,
                    num_branch_topologies_tried=evaluated_topologies - len(survivor_topologies),
                    num_injection_topologies_tried=0,
                )
            )

            if time.time() - start_time > ac_params.ga_config.runtime_seconds:
                if len(survivor_topologies) > 0:
                    logger.info(
                        f"Stopping optimization {optimization_id} at epoch {epoch} due to runtime limit"
                        f" with survivor strategies still present"
                        f" Running remaining contingencies evaluation before stopping"
                    )
                    evaluate_remaining_contingencies(
                        send_result_fn,
                        optimizer_data,
                        epoch,
                        survivor_topologies,
                        survivor_early_results,
                    )
                else:
                    logger.info(f"Stopping optimization at epoch {epoch} due to runtime limit with no survivor strategies")
                send_result_fn(OptimizationStoppedResult(epoch=epoch, reason="converged", message="runtime limit"))
                return


def summarize_optimization_run(
    optimization_id: str,
    grid_file: GridFile,
    worker_data: WorkerData,
    optimizer_data: OptimizerData,
    processed_gridfile_fs: AbstractFileSystem,
) -> None:
    """Write the optimization summary artifacts."""
    logger.info(f"Writing summary for optimization {optimization_id}")
    write_summary(
        grid_file=grid_file,
        db=worker_data.db,
        processed_gridfile_fs=processed_gridfile_fs,
        optimization_id=optimization_id,
        action_set=optimizer_data.action_set,
    )


def optimization_loop(
    ac_params: ACOptimizerParameters,
    grid_file: GridFile,
    worker_data: WorkerData,
    send_result_fn: Callable[[ResultUnion], None],
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    optimization_id: str,
    loadflow_result_fs: AbstractFileSystem,
    processed_gridfile_fs: AbstractFileSystem,
) -> None:
    """Run the main loop for the AC optimizer.

    This function will run the AC optimizer on the given grid files with the given parameters.

    Parameters
    ----------
    ac_params : ACOptimizerParameters
        The parameters for the AC optimizer
    grid_file : GridFile
        The grid file to optimize on
    worker_data : WorkerData
        The dataclass with the results consumer and database
    send_result_fn : Callable[[ResultUnion], None]
        The function to send results
    send_heartbeat_fn : Callable[[HeartbeatUnion], None]
        The function to send heartbeats
    optimization_id : str
        The ID of the optimization run
    loadflow_result_fs: AbstractFileSystem
        A filesystem where the loadflow results are stored. Loadflows will be stored here using the uuid generation process
        and passed as a StoredLoadflowReference which contains the subfolder in this filesystem.
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder was created
        which will be completed with the preprocess call to this function.
        Internally, only the data folder is passed around as a dirfs.
        Note that the unprocessed_gridfile_fs is not needed here anymore, as all preprocessing steps that need the
        unprocessed gridfiles were already done.
    """
    with structlog.contextvars.bound_contextvars(optimization_id=optimization_id):
        logger.info("Initializing optimization")

        try:
            optimizer_data, _ = initialize_optimization_run(
                ac_params=ac_params,
                grid_file=grid_file,
                worker_data=worker_data,
                send_result_fn=send_result_fn,
                send_heartbeat_fn=send_heartbeat_fn,
                optimization_id=optimization_id,
                loadflow_result_fs=loadflow_result_fs,
                processed_gridfile_fs=processed_gridfile_fs,
            )
        except AcNotConvergedError as e:
            # If the AC optimization did not converge in the base grid, we send a special message
            # to indicate that the optimization cannot be run.
            send_result_fn(OptimizationStoppedResult(reason="ac-not-converged", message=str(e)))
            logger.error(f"AC optimization {optimization_id} did not converge in the base grid: {e}")
            return
        except TimeoutError as e:
            # If the DC results did not arrive in time, we assume a failure on DC side and abandon the optimization
            send_result_fn(OptimizationStoppedResult(reason="dc-not-started", message=str(e)))
            logger.error(f"DC results for optimization {optimization_id} did not arrive in time: {e}")
            return
        except Exception as e:
            send_result_fn(OptimizationStoppedResult(reason="error", message=str(e)))
            logger.error(f"Error during initialization of optimization {optimization_id}: {e}")
            return

        logger.info(f"Starting optimization {optimization_id}")

        try:
            run_optimization_epochs(
                ac_params=ac_params,
                optimizer_data=optimizer_data,
                worker_data=worker_data,
                send_result_fn=send_result_fn,
                send_heartbeat_fn=send_heartbeat_fn,
                optimization_id=optimization_id,
            )
        except Exception as e:
            # Send a stop message to the results
            send_result_fn(OptimizationStoppedResult(reason="error", message=str(e)))
            logger.error(f"Error during optimization {optimization_id}: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return

        try:
            summarize_optimization_run(
                optimization_id=optimization_id,
                grid_file=grid_file,
                worker_data=worker_data,
                optimizer_data=optimizer_data,
                processed_gridfile_fs=processed_gridfile_fs,
            )
        except Exception as e:
            logger.error(f"Error while writing summary for optimization {optimization_id}: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return


def idle_loop(
    worker_data: WorkerData,
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    send_result_fn: Callable[[ResultUnion, str], None],
    heartbeat_interval_ms: int,
    max_command_age_hours: float,
) -> StartOptimizationCommand:
    """Run idle loop of the AC optimizer worker.

    This will be running when the worker is currently not optimizing
    This will wait until a StartOptimizationCommand is received and return it. In case a
    ShutdownCommand is received, the worker will exit with the exit code provided in the command.

    Parameters
    ----------
    worker_data : WorkerData
        The dataclass with the command consumer, results consumer and database
    send_heartbeat_fn : Callable[[HeartbeatUnion], None]
        A function to call when there were no messages received for a while.
    send_result_fn : Callable[[ResultUnion, str], None]
        A function to call to send results back to the results topic,
        used to send a message in case a command is too old.
    heartbeat_interval_ms : int
        The time to wait for a new command in milliseconds. If no command has been received, a
        heartbeat will be sent and then the receiver will wait for commands again.
    max_command_age_hours: float
        The maximum age of a command in hours.
        If a command is received that is older than this, the command will be ignored
        and a message will be sent to the results topic.


    Returns
    -------
    StartOptimizationCommand
        The start optimization command to start the optimization run with

    Raises
    ------
    SystemExit
        If a ShutdownCommand is received
    """
    send_heartbeat_fn(IdleHeartbeat())
    logger.info("Entering idle loop")
    while True:
        message = worker_data.command_consumer.poll(heartbeat_interval_ms / 1000)

        # Wait timeout exceeded - send a heartbeat and poll the results topic
        if not message:
            send_heartbeat_fn(IdleHeartbeat())
            poll_results_topic(
                db=worker_data.db,
                consumer=worker_data.result_consumer,
                first_poll=False,
            )
            continue

        message_value = message.value()
        if message_value is None:
            logger.warning("Received command without payload, dropping message")
            worker_data.command_consumer.commit()
            continue

        command = Command.model_validate_json(deserialize_message(message_value))

        if isinstance(command.command, ShutdownCommand):
            logger.info("Shutting down due to ShutdownCommand")
            worker_data.command_consumer.close()
            worker_data.result_consumer.close()
            raise SystemExit(command.command.exit_code)

        if isinstance(command.command, StartOptimizationCommand):
            time_of_command = datetime.fromisoformat(command.timestamp)
            if time_of_command < datetime.now() - timedelta(hours=max_command_age_hours):
                logger.warning(
                    f"Received command with timestamp from the past (timestamp: {time_of_command}, "
                    f"now: {datetime.now()}), skipping command"
                )
                send_result_fn(
                    OptimizationStoppedResult(
                        reason="command-too-old", message=f"Received outdated command: {command}. Skipping.."
                    ),
                    command.command.optimization_id,
                )
                worker_data.command_consumer.commit()
                continue
            with structlog.contextvars.bound_contextvars(
                optimization_id=command.command.optimization_id,
            ):
                return command.command

        # If we are here, we received a command that we do not know
        logger.warning(f"Received unknown command, dropping: {command} / {message.value}")
        worker_data.command_consumer.commit()


def warmup_result_storage(
    worker_data: WorkerData,
    heartbeat_fn: Callable[[HeartbeatUnion], None],
    heartbeat_interval_ms: int = 5000,
) -> None:
    """Go through the results topic and stores all published topology results in the database

    This way, when an optimization run starts, the local in-memory results storage is already populated with results, so
    no time is wasted in spooling through the results topic.

    Parameters
    ----------
    worker_data : WorkerData
        The dataclass with the results consumer and database
    heartbeat_fn : Callable[[HeartbeatUnion], None]
        A function to call to send heartbeats during the warmup loop, as it can take a while if there are many results to
        spool through.
    heartbeat_interval_ms : int
        The time interval in milliseconds to send heartbeats during the warmup loop, by default 5000 ms
    """
    logger.info("Starting warmup loop to spool through results topic and populate database")
    first_poll = True
    last_heartbeat_time = time.time()
    while True:
        added_topos, _ = poll_results_topic(
            db=worker_data.db,
            consumer=worker_data.result_consumer,
            first_poll=first_poll,
        )
        first_poll = False

        if added_topos == {}:
            break

        if time.time() - last_heartbeat_time > heartbeat_interval_ms / 1000:
            heartbeat_fn(StartupHeartbeat())
            last_heartbeat_time = time.time()


def main(
    args: Args,
    loadflow_result_fs: AbstractFileSystem,
    processed_gridfile_fs: AbstractFileSystem,
    producer: Producer,
    command_consumer: LongRunningKafkaConsumer,
    result_consumer: LongRunningKafkaConsumer,
) -> None:
    """Run the main AC worker loop.

    Parameters
    ----------
    args : Args
        The command line arguments
    loadflow_result_fs: AbstractFileSystem
        A filesystem where the loadflow results are stored. Loadflows will be stored here using the uuid generation process
        and passed as a StoredLoadflowReference which contains the subfolder in this filesystem.
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder was created
        which will be completed with the preprocess call to this function.
        Internally, only the data folder is passed around as a dirfs.
        Note that the unprocessed_gridfile_fs is not needed here anymore, as all preprocessing steps that need the
        unprocessed gridfiles were already done.
    producer : Producer
        A kafka producer to send heartbeats and results
    command_consumer : LongRunningKafkaConsumer
        A kafka consumer listening in for optimization commands
    result_consumer : LongRunningKafkaConsumer
        A kafka consumer listening in for results

    Raises
    ------
    SystemExit
        If the worker receives a ShutdownCommand
    """
    instance_id = str(uuid4())
    logger.info(
        f"Starting AC worker {instance_id} with config {args}, spooling through results topic to warm up result storage"
    )

    # We create two separate consumers for the command and result topics as we don't want to
    # catch results during the idle loop.
    worker_data = WorkerData(
        command_consumer=command_consumer,
        # Create a results consumer that will listen to results from any DC optimizers
        # Make sure to use a unique group.id for each instance to avoid conflicts
        result_consumer=result_consumer,
        producer=producer,
        db=create_session(),
    )

    def send_heartbeat(message: HeartbeatUnion, ping_commands: bool) -> None:
        logger.debug(f"Sending heartbeat: {message}", message_type=type(message).__name__)
        heartbeat = Heartbeat(
            optimizer_type=OptimizerType.AC,
            instance_id=instance_id,
            message=message,
        )
        Heartbeat.model_validate(heartbeat)  # Validate the heartbeat message
        worker_data.producer.produce(
            args.optimizer_heartbeat_topic,
            value=serialize_message(heartbeat.model_dump_json()),
            key=heartbeat.instance_id.encode(),
        )
        worker_data.producer.flush()
        if ping_commands:
            worker_data.command_consumer.heartbeat()

    def send_result(message: ResultUnion, optimization_id: str) -> None:
        logger.info(
            f"Sending result for optimization {optimization_id}: {message}",
            optimization_id=optimization_id,
            result_type=type(message).__name__,
        )
        result = Result(
            result=message,
            optimization_id=optimization_id,
            optimizer_type=OptimizerType.AC,
            instance_id=instance_id,
        )
        Result.model_validate(result)  # Validate the result message
        worker_data.producer.produce(
            args.optimizer_results_topic,
            value=serialize_message(result.model_dump_json()),
            key=result.optimization_id.encode(),
        )
        worker_data.producer.flush()

    send_heartbeat(StartupHeartbeat(), ping_commands=False)
    worker_data.command_consumer.start_processing()
    warmup_result_storage(
        worker_data=worker_data,
        heartbeat_fn=partial(send_heartbeat, ping_commands=True),
        heartbeat_interval_ms=args.heartbeat_interval_ms,
    )
    worker_data.command_consumer.stop_processing()

    logger.info("Finished warmup loop, entering main loop to wait for commands and run optimizations")

    while True:
        # During the idle loop, the result consumer is paused and only the command consumer is active
        command = idle_loop(
            worker_data=worker_data,
            send_heartbeat_fn=partial(send_heartbeat, ping_commands=False),
            send_result_fn=send_result,
            heartbeat_interval_ms=args.heartbeat_interval_ms,
            max_command_age_hours=args.max_command_age_hours,
        )

        # During the optimization loop, the command consumer is paused and the result consumer is active
        worker_data.command_consumer.start_processing()
        assert len(command.grid_files) == 1, "Exactly one grid file should be provided for the AC optimizer"
        optimization_loop(
            ac_params=command.ac_params,
            grid_file=command.grid_files[0],
            worker_data=worker_data,
            send_result_fn=partial(send_result, optimization_id=command.optimization_id),
            send_heartbeat_fn=partial(send_heartbeat, ping_commands=True),
            optimization_id=command.optimization_id,
            loadflow_result_fs=loadflow_result_fs,
            processed_gridfile_fs=processed_gridfile_fs,
        )
        worker_data.command_consumer.stop_processing()
        scrub_db(worker_data.db)
