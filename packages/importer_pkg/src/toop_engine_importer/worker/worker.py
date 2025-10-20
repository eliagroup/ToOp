"""Module contains functions for the kafka communication in the importer repo.

File: worker.py
Author:  Nico Westerbeck
Created: 2024
"""

import os
import sys
import time
import traceback
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import jax
import logbook
import tyro
from confluent_kafka import Producer
from pydantic import BaseModel
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_importer.worker.preprocessor import import_grid_model, preprocess
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    Command,
    ShutdownCommand,
    StartPreprocessingCommand,
)
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import (
    PreprocessHeartbeat,
    PreprocessStage,
    PreprocessStatusInfo,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    ErrorResult,
    PreprocessingStartedResult,
    Result,
)

logger = logbook.Logger(__name__)


class Args(BaseModel):
    """Holds arguments which must be provided at the launch of the worker.

    Contains arguments that static for each preprocessing run.
    """

    kafka_broker: str = "localhost:9092"
    """The Kafka broker to connect to."""

    importer_command_topic: str = "importer_commands"
    """The Kafka topic to listen for commands on."""

    importer_results_topic: str = "importer_results"
    """The topic to push results to."""

    importer_heartbeat_topic: str = "importer_heartbeat"
    """The topic to push heartbeats to."""

    heartbeat_interval_ms: int = 1000
    """The interval in milliseconds to send heartbeats."""

    unprocessed_gridfile_folder: Path = Path("gridfiles")
    """A folder where unprocessed grid files are uploaded to - this should be an NFS share together with the backend"""

    processed_gridfile_folder: Path = Path("processed_gridfiles")
    """A folder where pre-processed grid files are stored - this should be a NFS share together with the backend and
    optimizer"""

    loadflow_result_folder: Path = Path("loadflow_results")
    """A folder where the loadflow results are stored - this should be a NFS share together with the backend and
    optimizer. The importer needs this to store the initial loadflows"""


def adjust_folders(
    preprocessing_command: StartPreprocessingCommand,
    unprocessed_gridfile_folder: Path,
    processed_gridfile_folder: Path,
) -> StartPreprocessingCommand:
    """Adjust the folders of all files in the StartPreprocessingCommand, prepending the gridfile_folders.

    Parameters
    ----------
    preprocessing_command : StartPreprocessingCommand
        The command in which to adjust file paths
    unprocessed_gridfile_folder : Path
        The folder where unprocessed grid files are uploaded to, this will be prepended to input grid files (ucte/cgmes) and
        black/whitelists
    processed_gridfile_folder : Path
        The folder where pre-processed grid files are stored, this will be prepended to the output grid file

    Returns
    -------
    StartPreprocessingCommand
        The command with adjusted file paths
    """
    importer_parameters = preprocessing_command.importer_parameters.model_copy(
        update={
            "grid_model_file": unprocessed_gridfile_folder / preprocessing_command.importer_parameters.grid_model_file,
            "white_list_file": (
                unprocessed_gridfile_folder / preprocessing_command.importer_parameters.white_list_file
                if preprocessing_command.importer_parameters.white_list_file is not None
                else None
            ),
            "black_list_file": (
                unprocessed_gridfile_folder / preprocessing_command.importer_parameters.black_list_file
                if preprocessing_command.importer_parameters.black_list_file is not None
                else None
            ),
            "data_folder": processed_gridfile_folder / preprocessing_command.importer_parameters.data_folder,
            "ignore_list_file": unprocessed_gridfile_folder / preprocessing_command.importer_parameters.ignore_list_file
            if preprocessing_command.importer_parameters.ignore_list_file is not None
            else None,
            "contingency_list_file": (
                unprocessed_gridfile_folder / preprocessing_command.importer_parameters.contingency_list_file
                if preprocessing_command.importer_parameters.contingency_list_file is not None
                else None
            ),
        }
    )
    command = preprocessing_command.model_copy(update={"importer_parameters": importer_parameters})
    command.model_validate(command)
    return command


def idle_loop(
    consumer: LongRunningKafkaConsumer,
    send_heartbeat_fn: Callable[[], None],
    heartbeat_interval_ms: int,
) -> StartPreprocessingCommand:
    """Start the idle loop of the worker.

    This will be running when the worker is currently not preprocessing
    This will wait until a StartPreprocessingCommand is received and return it. In case a
    ShutdownCommand is received, the worker will exit with the exit code provided in the command.

    Parameters
    ----------
    consumer : LongRunningKafkaConsumer
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
        message = consumer.poll(timeout=heartbeat_interval_ms / 1000)

        # Wait timeout exceeded
        if not message:
            send_heartbeat_fn()
            continue

        command = Command.model_validate_json(message.value().decode("utf-8"))

        if isinstance(command.command, StartPreprocessingCommand):
            return command.command

        consumer.commit()
        if isinstance(command.command, ShutdownCommand):
            consumer.consumer.close()
            raise SystemExit(command.command.exit_code)

        # If we are here, we received a command that we do not know
        logger.warning(f"Received unknown command, dropping: {command}")


def main(args: Args) -> None:
    """Start main function of the worker."""
    instance_id = str(uuid4())
    logger.info(f"Starting importer instance {instance_id} with arguments {args}")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_logging_level", "INFO")

    consumer = LongRunningKafkaConsumer(
        topic=args.importer_command_topic,
        bootstrap_servers=args.kafka_broker,
        group_id="importer-worker",
        client_id=instance_id,
    )

    producer = Producer(
        {
            "bootstrap.servers": args.kafka_broker,
            "client.id": instance_id,
            "log_level": 2,
        },
        logger=getLogger("confluent_kafka.producer"),
    )

    def heartbeat_idle() -> None:
        producer.produce(
            args.importer_heartbeat_topic,
            value=PreprocessHeartbeat(
                idle=True,
                status_info=None,
                instance_id=instance_id,
            )
            .model_dump_json()
            .encode(),
            key=instance_id.encode("utf-8"),
        )
        producer.flush()

    def heartbeat_working(
        stage: PreprocessStage,
        message: Optional[str],
        preprocess_id: str,
        start_time: float,
    ) -> None:
        logger.info(f"Preprocessing stage {stage} for job {preprocess_id} after {time.time() - start_time}s: {message}")
        producer.produce(
            args.importer_heartbeat_topic,
            value=PreprocessHeartbeat(
                idle=False,
                status_info=PreprocessStatusInfo(
                    preprocess_id=preprocess_id,
                    runtime=time.time() - start_time,
                    stage=stage,
                    message=message,
                ),
                instance_id=instance_id,
            )
            .model_dump_json()
            .encode(),
            key=preprocess_id.encode("utf-8"),
        )
        producer.flush()
        # Ping the command consumer to show we are still alive
        consumer.heartbeat()

    while True:
        command = idle_loop(
            consumer=consumer,
            send_heartbeat_fn=heartbeat_idle,
            heartbeat_interval_ms=args.heartbeat_interval_ms,
        )
        consumer.start_processing()
        command = adjust_folders(
            command,
            args.unprocessed_gridfile_folder,
            args.processed_gridfile_folder,
        )
        start_time = time.time()
        heartbeat_fn = partial(
            heartbeat_working,
            preprocess_id=command.preprocess_id,
            start_time=start_time,
        )
        producer.produce(
            args.importer_results_topic,
            value=Result(
                preprocess_id=command.preprocess_id,
                runtime=0,
                result=PreprocessingStartedResult(),
            )
            .model_dump_json()
            .encode(),
            key=command.preprocess_id.encode(),
        )
        producer.flush()
        heartbeat_fn("start", "Preprocessing run started")

        try:
            importer_results = import_grid_model(start_command=command, status_update_fn=heartbeat_fn)

            result = preprocess(
                start_command=command,
                import_results=importer_results,
                status_update_fn=heartbeat_fn,
                loadflow_result_folder=args.loadflow_result_folder,
            )

            heartbeat_fn("end", "Preprocessing run done")

            producer.produce(
                topic=args.importer_results_topic,
                value=Result(
                    preprocess_id=command.preprocess_id,
                    runtime=time.time() - start_time,
                    result=result,
                )
                .model_dump_json()
                .encode(),
                key=command.preprocess_id.encode(),
            )
        except Exception as e:
            logger.error(f"Error while processing {command.preprocess_id}", e)
            logger.error(f"Traceback: {traceback.format_exc()}")
            producer.produce(
                topic=args.importer_results_topic,
                value=Result(
                    preprocess_id=command.preprocess_id,
                    runtime=time.time() - start_time,
                    result=ErrorResult(error=str(e)),
                )
                .model_dump_json()
                .encode(),
                key=command.preprocess_id.encode(),
            )
        producer.flush()
        consumer.stop_processing()


if __name__ == "__main__":
    logbook.StreamHandler(sys.stdout, level=logbook.INFO).push_application()
    logbook.compat.redirect_logging()
    if "IMPORTER_CONFIG_FILE" in os.environ:
        with open(os.environ["IMPORTER_CONFIG_FILE"], "r") as f:
            args = Args.model_validate_json(f.read())
    else:
        args = tyro.cli(Args)
    main(args)
