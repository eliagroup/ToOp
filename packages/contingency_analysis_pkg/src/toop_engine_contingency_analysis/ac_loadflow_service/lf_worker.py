# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions for the kafka communication of the ac loadflow worker.

General Idea:
- The worker will listen for commands on the preprocessing  kafka topic
- Once the initial conversion is done, it runs an initial loadflow
- Once the optimiuation is done,
The command contains a path to the pandapower or powsybl grid file
The command contains the N-1 Definition

- The worker will load the grid file and the N-1 definition

- The worker will run the N-1 analysis on the grid file with as many processes as possible
- The worker will send the results to a kafka topic
    - ErrorResult, if anything goes wrong
    - SuccessResult, if everything goes well even if loadflow fails
    - LoadflowStartedResult?
    # Use the LoadflowResultsClass as the result result
- The worker will send a heartbeat to a kafka topic every X seconds


Questions:
- Does it make sense to return the results in batches?
    - Faster results, but dont really tell the full story
- How to deal with grid updates?
    - Separate Service? (Load would be doubled)
- Passing the grid file as path valid?
    - Otherwise large files need to be passed as bytes
    which kafka supports but is not really intended

File: worker.py
Author:  Leonard Hilfrich
Created: 05/2024
"""

import time
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import pandapower
import tyro
from beartype.typing import Callable, Literal
from confluent_kafka import Producer
from fsspec import AbstractFileSystem
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from pypowsybl.network import Network
from toop_engine_contingency_analysis.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_grid_helpers.pandapower.pandapower_helpers import load_pandapower_from_fs
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_powsybl_from_fs
from toop_engine_interfaces.loadflow_result_helpers_polars import (
    concatenate_loadflow_results_polars,
    save_loadflow_results_polars,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.loadflow_commands import (
    LoadflowServiceCommand,
    ShutdownCommand,
    StartCalculationCommand,
)
from toop_engine_interfaces.messages.lf_service.loadflow_heartbeat import LoadflowHeartbeat, LoadflowStatusInfo
from toop_engine_interfaces.messages.lf_service.loadflow_results import (
    ErrorResult,
    LoadflowBaseResult,
    LoadflowStartedResult,
    LoadflowStreamResult,
    LoadflowSuccessResult,
)
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message

logger = getLogger(__name__)


@dataclass
class LoadflowWorkerArgs:
    """Holds arguments which must be provided at the launch of the worker.

    Contains arguments that static for each loadflow run.
    """

    kafka_broker: str = "localhost:9092"
    """The Kafka broker to connect to."""

    loadflow_command_topic: str = "loadflow_commands"
    """The Kafka topic to listen for commands on."""

    loadflow_results_topic: str = "loadflow_results"
    """The topic to push results to."""

    loadflow_heartbeat_topic: str = "loadflow_heartbeat"
    """The topic to push heartbeats to."""

    heartbeat_interval_ms: int = 1000
    """The interval in milliseconds to send heartbeats."""

    instance_id: str = "loadflow_worker"
    """The instance id of the worker, used to identify the worker in the logs."""

    processed_gridfile_folder: Path = Path("processed_gridfiles")
    """A folder where pre-processed grid files are stored - this should be a NFS share together with the backend and
    optimizer."""

    loadflow_result_folder: Path = Path("loadflow_results")
    """A folder where the loadflow results are stored - this should be a NFS share together with the backend and
    optimizer."""

    n_processes: int = 1
    """The number of processes to use for the loadflow calculation. If 1, the analysis is run sequentially.
    If > 1, the analysis is run in parallel"""


def idle_loop(
    consumer: LongRunningKafkaConsumer,
    send_heartbeat_fn: Callable[[], None],
    heartbeat_interval_ms: int,
) -> StartCalculationCommand:
    """Start the idle loop of the worker.

    This will be running when the worker is currently not preprocessing
    This will wait until a StartCalculationCommand is received and return it. In case a
    ShutdownCommand is received, the worker will exit with the exit code provided in the command.

    Parameters
    ----------
    consumer : Consumer
        The initialized Kafka consumer to listen for commands on.
    send_heartbeat_fn : callable
        A function to call when there were no messages received for a while.
    heartbeat_interval_ms : int
        The time to wait for a new command in milliseconds. If no command has been received, a
        heartbeat will be sent and then the receiver will wait for commands again.

    Returns
    -------
    StartOptimizationCommand
        The start optimization command to start the optimization run with
    """
    send_heartbeat_fn()
    logger.info("Entering idle loop")
    while True:
        message = consumer.poll(timeout=heartbeat_interval_ms / 1000.0)

        # Wait timeout exceeded
        if not message:
            send_heartbeat_fn()
            continue

        command = LoadflowServiceCommand.model_validate_json(deserialize_message(message.value()))

        if isinstance(command.command, StartCalculationCommand):
            return command.command

        if isinstance(command.command, ShutdownCommand):
            consumer.commit()
            consumer.consumer.close()
            raise SystemExit(command.command.exit_code)

        # If we are here, we received a command that we do not know
        logger.warning(f"Received unknown command, dropping: {command}")
        consumer.commit()


def solver_loop(
    command: StartCalculationCommand,
    producer: Producer,
    processed_grid_path: Path,
    loadflow_solver_path: Path,
    heartbeat_fn: Callable,
    instance_id: str,
    n_processes: int,
    results_topic: str,
) -> None:
    """Start the solver loop of the worker.

    This will be running when the worker is currently solving the loadflow
    This will wait until a StartCalculationCommand is received and return it. In case a
    ShutdownCommand is received, the worker will exit with the exit code provided in the command.

    Parameters
    ----------
    command : StartCalculationCommand
        The command to start the optimization run with.
    producer : KafkaProducer
        The initialized Kafka producer to send results to.
    processed_grid_path : Path
        The path to the pre-processed grid files. This is used to load the grid files.
    loadflow_solver_path : Path
        The path to the loadflow solver results. This is used to save the loadflow results.
    heartbeat_fn : Callable
        A function to call to send a heartbeat message to the kafka topic.
    instance_id : str
        The instance id of the worker, used to identify the worker in the logs.
    n_processes : int
        The number of processes to use for the optimization run. If 1, the analysis is run sequentially.
        If > 1, the analysis is run in parallel
        Paralelization is done by splitting the contingencies into chunks and running each chunk in a separate process
    results_topic : str
        The topic to push results to.
    """
    start_time = time.time()
    dirfs = DirFileSystem(str(loadflow_solver_path))
    try:
        if command.grid_data.n_1_definition is None:
            raise ValueError("No N-1 definition provided. This is currently not supported.")
        n_minus_1_definition = command.grid_data.n_1_definition

        for job in command.jobs:
            producer.produce(
                results_topic,
                value=serialize_message(
                    LoadflowBaseResult(
                        job_id=job.id,
                        instance_id=instance_id,
                        loadflow_id=command.loadflow_id,
                        runtime=0.0,
                        result=LoadflowStartedResult(),
                    ).model_dump_json()
                ),
                key=command.loadflow_id.encode(),
            )
            job_loadflow_results_polars = LoadflowResultsPolars(job_id=job.id)
            for i, grid in enumerate(command.grid_data.grid_files):
                heartbeat_fn(
                    command.loadflow_id, time.time() - start_time, f"Loadflow Calculation run started for timestep {i}"
                )
                net = load_base_grid(processed_grid_path / grid, command.grid_data.grid_type)
                timestep_result_polars = get_ac_loadflow_results(
                    net=net, n_minus_1_definition=n_minus_1_definition, timestep=i, job_id=job.id, n_processes=n_processes
                )
                job_loadflow_results_polars = concatenate_loadflow_results_polars(
                    [job_loadflow_results_polars, timestep_result_polars]
                )
                ref = save_loadflow_results_polars(dirfs, job.id, job_loadflow_results_polars)
                if i < len(command.grid_data.grid_files) - 1:
                    result_msg = LoadflowStreamResult(
                        loadflow_reference=ref,
                        solved_timesteps=list(range(i + 1)),
                        remainging_timesteps=list(range(i + 1, len(command.grid_data.grid_files))),
                    )
                else:
                    result_msg = LoadflowSuccessResult(loadflow_reference=ref)

                producer.produce(
                    topic=results_topic,
                    value=serialize_message(
                        LoadflowBaseResult(
                            job_id=job.id,
                            loadflow_id=command.loadflow_id,
                            instance_id=instance_id,
                            runtime=time.time() - start_time,
                            result=result_msg,
                        ).model_dump_json()
                    ),
                    key=command.loadflow_id.encode(),
                )
    except Exception as e:
        logger.error(f"Error while processing {command.loadflow_id}: {e}")
        producer.produce(
            topic=results_topic,
            value=serialize_message(
                LoadflowBaseResult(
                    job_id=command.loadflow_id,
                    instance_id=instance_id,
                    loadflow_id=command.loadflow_id,
                    runtime=time.time() - start_time,
                    result=ErrorResult(error=str(e)),
                ).model_dump_json()
            ),
            key=command.loadflow_id.encode(),
        )


def load_base_grid_fs(
    filesystem: AbstractFileSystem,
    grid_path: Path,
    grid_type: Literal["pandapower", "powsybl", "ucte", "cgmes"],
) -> pandapower.pandapowerNet | Network:
    """Load the base grid from the grid file.

    Force loading pandapower if grid type is pandapower, otherwise load powsybl.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The filesystem to load the grid from
    grid_path : Path
        The grid to load
    grid_type: Literal["pandapower", "powsybl", "ucte", "cgmes"]
        The type of the grid, either "pandapower", "powsybl", "ucte" or "cgmes".

    Returns
    -------
    PandapowerNet | Network
        The loaded grid

    Raises
    ------
    ValueError
        If the grid type is not supported.
    """
    if grid_type == "pandapower":
        return load_pandapower_from_fs(filesystem, grid_path)
    if grid_type in ["powsybl", "ucte", "cgmes"]:
        return load_powsybl_from_fs(filesystem, grid_path)
    raise ValueError(f"Unknown grid type: {grid_type}")


def load_base_grid(
    grid_path: Path, grid_type: Literal["pandapower", "powsybl", "ucte", "cgmes"]
) -> pandapower.pandapowerNet | Network:
    """Load the base grid from the grid file.

    Parameters
    ----------
    grid_path : Path
        The grid to load
    grid_type: Literal["pandapower", "powsybl", "ucte", "cgmes"]
        The type of the grid, either "pandapower", "powsybl", "ucte" or "cgmes".

    Returns
    -------
    PandapowerNet | Network
        The loaded grid

    Raises
    ------
    ValueError
        If the grid type is not supported.
    """
    return load_base_grid_fs(LocalFileSystem(), grid_path, grid_type)


def main(args: LoadflowWorkerArgs) -> None:
    """Start main function of the worker."""
    logger.info(f"Starting importer instance {args.instance_id}")
    consumer = LongRunningKafkaConsumer(
        topic=args.loadflow_command_topic,
        group_id="loadflow-worker",
        bootstrap_servers=args.kafka_broker,
        client_id=args.instance_id,
    )

    producer = Producer(
        {
            "bootstrap.servers": args.kafka_broker,
            "client.id": args.instance_id,
            "log_level": 2,
        },
        logger=logger,
    )

    def heartbeat_idle() -> None:
        producer.produce(
            args.loadflow_heartbeat_topic,
            value=serialize_message(
                LoadflowHeartbeat(
                    idle=True,
                    status_info=None,
                ).model_dump_json()
            ),
            key=args.instance_id.encode("utf-8"),
        )
        producer.flush()

    def heartbeat_fn(job_id: str, runtime: float, message: str = "") -> None:
        producer.produce(
            args.loadflow_heartbeat_topic,
            value=serialize_message(
                LoadflowHeartbeat(
                    idle=False,
                    status_info=LoadflowStatusInfo(
                        loadflow_id=job_id,
                        runtime=runtime,
                        message=message,
                    ),
                ).model_dump_json()
            ),
            key=args.instance_id.encode("utf-8"),
        )
        producer.flush()
        consumer.heartbeat()

    while True:
        command = idle_loop(
            consumer=consumer,
            send_heartbeat_fn=heartbeat_idle,
            heartbeat_interval_ms=args.heartbeat_interval_ms,
        )
        consumer.start_processing()
        solver_loop(
            command=command,
            producer=producer,
            processed_grid_path=args.processed_gridfile_folder,
            loadflow_solver_path=args.loadflow_result_folder,
            heartbeat_fn=heartbeat_fn,
            instance_id=args.instance_id,
            n_processes=args.n_processes,
            results_topic=args.loadflow_results_topic,
        )
        producer.flush()
        consumer.stop_processing()


if __name__ == "__main__":
    args = tyro.cli(LoadflowWorkerArgs)
    main(args)
