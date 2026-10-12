# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os
import time

import docker
import psycopg
import pytest
from beartype.typing import Generator
from docker import DockerClient
from docker.models.containers import Container
from sqlalchemy import Engine
from sqlmodel import SQLModel, Session, create_engine
from toop_engine_topology_optimizer.database.command_models import (
    ActiveWorker,
    OptimizationJob,
    StageExecutionHistory,
    StageWorkItem,
)

POSTGRES_USER = "test_user"
POSTGRES_PASSWORD = "test_password"
POSTGRES_DATABASE = "test_db"
POSTGRES_INTERNAL_PORT = 5432

COMMAND_DATABASE_TABLES = [
    OptimizationJob.__table__,
    StageWorkItem.__table__,
    StageExecutionHistory.__table__,
    ActiveWorker.__table__,
]


@pytest.fixture(scope="session")
def docker_client() -> Generator[DockerClient, None, None]:
    client = docker.from_env()
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="session")
def postgres_container(docker_client: DockerClient) -> Generator[Container, None, None]:
    container = docker_client.containers.run(
        "postgres:15",
        name=f"test_topology_optimizer_postgres_{os.urandom(4).hex()}",
        environment={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DATABASE,
        },
        ports={f"{POSTGRES_INTERNAL_PORT}/tcp": None},
        detach=True,
        remove=True,
    )

    for attempt in range(30):
        result = container.exec_run(f"pg_isready -U {POSTGRES_USER} -d {POSTGRES_DATABASE}")
        if result.exit_code == 0:
            break

        if attempt == 29:
            container.stop()
            msg = "PostgreSQL container failed to start in time"
            raise TimeoutError(msg)

        time.sleep(1)

    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def postgres_connection_string(postgres_container: Container) -> str:
    postgres_container.reload()
    host_port = postgres_container.attrs["NetworkSettings"]["Ports"][f"{POSTGRES_INTERNAL_PORT}/tcp"][0]["HostPort"]
    connection_string = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{host_port}/{POSTGRES_DATABASE}"

    for attempt in range(30):
        try:
            with psycopg.connect(
                dbname=POSTGRES_DATABASE,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host="localhost",
                port=int(host_port),
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
            return connection_string
        except psycopg.OperationalError:
            if attempt == 29:
                raise
            time.sleep(1)

    raise RuntimeError("PostgreSQL connection string could not be validated")


@pytest.fixture
def command_database_engine(postgres_connection_string: str) -> Generator[Engine, None, None]:
    engine = create_engine(postgres_connection_string)
    SQLModel.metadata.drop_all(engine, tables=COMMAND_DATABASE_TABLES)
    SQLModel.metadata.create_all(engine, tables=COMMAND_DATABASE_TABLES)

    try:
        yield engine
    finally:
        SQLModel.metadata.drop_all(engine, tables=COMMAND_DATABASE_TABLES)
        engine.dispose()


@pytest.fixture
def command_database_session(command_database_engine: Engine) -> Generator[Session, None, None]:
    with Session(command_database_engine, autobegin=False, expire_on_commit=False) as session:
        yield session