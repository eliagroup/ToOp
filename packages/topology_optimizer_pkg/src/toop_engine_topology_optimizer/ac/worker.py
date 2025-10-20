"""The AC worker that listens to the kafka topics, organizes optimization runs, etc."""

import os
import sys
import time
import traceback
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable
from uuid import uuid4

import logbook
import tyro
from confluent_kafka import Producer
from sqlmodel import Session
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_topology_optimizer.ac.listener import poll_results_topic
from toop_engine_topology_optimizer.ac.optimizer import AcNotConvergedError, initialize_optimization, run_epoch
from toop_engine_topology_optimizer.ac.storage import create_session
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
)
from toop_engine_topology_optimizer.interfaces.messages.results import (
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    ResultUnion,
)

logger = logbook.Logger(__name__)


class Args(DCArgs):
    """Command line arguments for the AC worker.

    Mostly the same as the DC worker except for an additional loadflow results folder
    """

    loadflow_result_folder: Path = Path("loadflow_results")
    """The parent folder where all loadflow results are stored. In production this is a shared network
    filesystem between the backend and all workers. Loadflow results are too large to be sent directly
    over kafka. Only needed for AC"""


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


def optimization_loop(
    ac_params: ACOptimizerParameters,
    grid_files: list[GridFile],
    worker_data: WorkerData,
    send_result_fn: Callable[[ResultUnion], None],
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    optimization_id: str,
    loadflow_result_folder: Path,
) -> None:
    """Run the main loop for the AC optimizer.

    This function will run the AC optimizer on the given grid files with the given parameters.

    Parameters
    ----------
    ac_params : ACOptimizerParameters
        The parameters for the AC optimizer
    grid_files : list[GridFile]
        The grid files to optimize on
    worker_data : WorkerData
        The dataclass with the results consumer and database
    send_result_fn : Callable[[ResultUnion], None]
        The function to send results
    send_heartbeat_fn : Callable[[HeartbeatUnion], None]
        The function to send heartbeats
    optimization_id : str
        The ID of the optimization run
    loadflow_result_folder : Path
        The folder where the loadflow results are stored.
    """
    logger.info(f"Initializing optimization {optimization_id}")
    try:
        send_heartbeat_fn(
            OptimizationStartedHeartbeat(
                optimization_id=optimization_id,
            )
        )
        optimizer_data, initial_topology = initialize_optimization(
            session=worker_data.db,
            params=ac_params,
            optimization_id=optimization_id,
            grid_files=grid_files,
            loadflow_result_folder=loadflow_result_folder,
        )
        send_result_fn(
            OptimizationStartedResult(
                initial_topology=initial_topology,
            )
        )
    except AcNotConvergedError as e:
        # If the AC optimization did not converge in the base grid, we send a special message
        # to indicate that the optimization cannot be run.
        send_result_fn(OptimizationStoppedResult(reason="ac-not-converged", message=str(e)))
        logger.error(f"AC optimization {optimization_id} did not converge in the base grid: {e}")
        return
    except Exception as e:
        send_result_fn(OptimizationStoppedResult(reason="error", message=str(e)))
        logger.error(f"Error during initialization of optimization {optimization_id}: {e}")
        return

    logger.info(f"Starting optimization {optimization_id}")
    epoch = 1  # Start at epoch 1 so the initial topology will be epoch 0
    running = True
    start_time = time.time()
    while running:
        try:
            run_epoch(optimizer_data, worker_data.result_consumer, send_result_fn, epoch=epoch)
        except Exception as e:
            # Send a stop message to the results
            send_result_fn(OptimizationStoppedResult(reason="error", message=str(e)))
            logger.error(f"Error during optimization {optimization_id}, epoch {epoch}: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return
        epoch += 1

        if time.time() - start_time > ac_params.ga_config.runtime_seconds:
            logger.info(f"Stopping optimization {optimization_id} at epoch {epoch} due to runtime limit")
            send_result_fn(OptimizationStoppedResult(epoch=epoch, reason="converged", message="runtime limit"))
            running = False
            break

        send_heartbeat_fn(
            OptimizationStatsHeartbeat(
                optimization_id=optimization_id,
                wall_time=time.time() - start_time,
                iteration=epoch,
                num_branch_topologies_tried=0,
                num_injection_topologies_tried=0,
            )
        )


def idle_loop(
    worker_data: WorkerData,
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    heartbeat_interval_ms: int,
    parent_grid_folder: Path,
) -> StartOptimizationCommand:
    """Run idle loop of the AC optimizer worker.

    This will be running when the worker is currently not optimizing
    This will wait until a StartOptimizationCommand is received and return it. In case a
    ShutdownCommand is received, the worker will exit with the exit code provided in the command.

    Parameters
    ----------
    worker_data : WorkerData
        The dataclass with the command consumer, results consumer and database
    send_heartbeat_fn : callable
        A function to call when there were no messages received for a while.
    heartbeat_interval_ms : int
        The time to wait for a new command in milliseconds. If no command has been received, a
        heartbeat will be sent and then the receiver will wait for commands again.
    parent_grid_folder : Path
        The folder where all the grid files are stored. This will be prefixed to the static
        information files in the StartOptimizationCommand.

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

        command = Command.model_validate_json(message.value().decode())

        if isinstance(command.command, StartOptimizationCommand):
            # Prefix the gridfile folder to the static information files
            command.command.grid_files = [
                gf.model_copy(update={"grid_folder": str(parent_grid_folder / gf.grid_folder)})
                for gf in command.command.grid_files
            ]
            return command.command

        if isinstance(command.command, ShutdownCommand):
            logger.info("Shutting down due to ShutdownCommand")
            worker_data.command_consumer.close()
            worker_data.result_consumer.close()
            raise SystemExit(command.command.exit_code)

        # If we are here, we received a command that we do not know
        logger.warning(f"Received unknown command, dropping: {command} / {message.value}")
        worker_data.command_consumer.commit()


def main(args: Args) -> None:
    """Run the main AC worker loop.

    Which for some reason has the same command line arguments as the DC worker :D

    Parameters
    ----------
    args : Args
        The command line arguments

    Raises
    ------
    SystemExit
        If the worker receives a ShutdownCommand
    """
    instance_id = str(uuid4())
    logger.info(f"Starting AC worker {instance_id} with config {args}")

    args.loadflow_result_folder.mkdir(parents=True, exist_ok=True)
    if not args.processed_gridfile_folder.exists():
        raise FileNotFoundError(f"Processed gridfile folder {args.processed_gridfile_folder} does not exist. ")

    # We create two separate consumers for the command and result topics as we don't want to
    # catch results during the idle loop.
    worker_data = WorkerData(
        command_consumer=LongRunningKafkaConsumer(
            topic=args.optimizer_command_topic,
            group_id="ac_optimizer",
            bootstrap_servers=args.kafka_broker,
            client_id=instance_id,
        ),
        # Create a results consumer that will listen to results from any DC optimizers
        # Make sure to use a unique group.id for each instance to avoid conflicts
        result_consumer=LongRunningKafkaConsumer(
            topic=args.optimizer_results_topic,
            group_id=f"ac_listener_{instance_id}_{uuid4()}",
            bootstrap_servers=args.kafka_broker,
            client_id=instance_id,
        ),
        producer=Producer(
            {
                "bootstrap.servers": args.kafka_broker,
                "client.id": instance_id,
                "log_level": 2,
            },
            logger=getLogger(f"ac_worker_producer_{instance_id}"),
        ),
        db=create_session(),
    )

    def send_heartbeat(message: HeartbeatUnion, ping_commands: bool) -> None:
        heartbeat = Heartbeat(
            optimizer_type=OptimizerType.AC,
            instance_id=instance_id,
            message=message,
        )
        Heartbeat.model_validate(heartbeat)  # Validate the heartbeat message
        worker_data.producer.produce(
            args.optimizer_heartbeat_topic,
            value=heartbeat.model_dump_json().encode(),
            key=heartbeat.instance_id.encode(),
        )
        worker_data.producer.flush()
        if ping_commands:
            worker_data.command_consumer.heartbeat()

    def send_result(message: ResultUnion, optimization_id: str) -> None:
        result = Result(
            result=message,
            optimization_id=optimization_id,
            optimizer_type=OptimizerType.AC,
            instance_id=instance_id,
        )
        Result.model_validate(result)  # Validate the result message
        worker_data.producer.produce(
            args.optimizer_results_topic,
            value=result.model_dump_json().encode(),
            key=result.optimization_id.encode(),
        )
        worker_data.producer.flush()

    while True:
        # During the idle loop, the result consumer is paused and only the command consumer is active
        command = idle_loop(
            worker_data=worker_data,
            send_heartbeat_fn=partial(send_heartbeat, ping_commands=False),
            heartbeat_interval_ms=args.heartbeat_interval_ms,
            parent_grid_folder=args.processed_gridfile_folder,
        )

        # During the optimization loop, the command consumer is paused and the result consumer is active
        worker_data.command_consumer.start_processing()
        optimization_loop(
            ac_params=command.ac_params,
            grid_files=command.grid_files,
            worker_data=worker_data,
            send_result_fn=partial(send_result, optimization_id=command.optimization_id),
            send_heartbeat_fn=partial(send_heartbeat, ping_commands=True),
            optimization_id=command.optimization_id,
            loadflow_result_folder=args.loadflow_result_folder,
        )
        worker_data.command_consumer.stop_processing()


if __name__ == "__main__":
    logbook.StreamHandler(sys.stdout, level=logbook.INFO).push_application()
    logbook.compat.redirect_logging()
    if "AC_OPTIMIZER_CONFIG_FILE" in os.environ:
        with open(os.environ["AC_OPTIMIZER_CONFIG_FILE"], "r") as f:
            args = Args.model_validate_json(f.read())
    else:
        args = tyro.cli(Args)
    main(args)
