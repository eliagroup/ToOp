# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Test-only pickle helpers for NetworkData fixtures."""

import pickle
from pathlib import Path

from beartype.typing import Union
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from toop_engine_dc_solver.preprocess.network_data import NetworkData


def save_network_data_fs(filesystem: AbstractFileSystem, filename: Union[str, Path], network_data: NetworkData) -> None:
    """Save network data to a filesystem-backed pickle for tests."""
    with filesystem.open(str(filename), "wb") as file:
        pickle.dump(network_data, file)


def save_network_data(filename: Union[str, Path], network_data: NetworkData) -> None:
    """Save network data to a local pickle for tests."""
    save_network_data_fs(LocalFileSystem(), filename, network_data)


def load_network_data_fs(filesystem: AbstractFileSystem, filename: Union[str, Path]) -> NetworkData:
    """Load network data from a filesystem-backed pickle for tests."""
    with filesystem.open(str(filename), "rb") as file:
        return pickle.load(file)


def load_network_data(filename: Union[str, Path]) -> NetworkData:
    """Load network data from a local pickle for tests."""
    return load_network_data_fs(LocalFileSystem(), filename)
