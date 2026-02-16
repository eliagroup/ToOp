# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Support functions for the AbstractFileSystem"""

import shutil
from pathlib import Path

import numpy as np
from beartype.typing import TypeVar, Union
from fsspec import AbstractFileSystem
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_pydantic_model_fs(
    filesystem: AbstractFileSystem,
    file_path: Union[str, Path],
    pydantic_model: BaseModel,
    indent: int = 2,
    make_dir: bool = True,
) -> None:
    """Save a Pydantic model to a JSON file.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to use to save the N-1 definition.
    file_path : Union[str, Path]
        The path to the json file to save the N-1 definition to.
    pydantic_model : Nminus1Definition
        The N-1 definition to save.
    indent: int
        The indent for the model dump
    make_dir: bool
        Whether to create the directory if it does not exist.
    """
    if make_dir:
        filesystem.makedirs(Path(file_path).parent.as_posix(), exist_ok=True)
    with filesystem.open(str(file_path), "w") as f:
        f.write(pydantic_model.model_dump_json(indent=indent))


def load_pydantic_model_fs(filesystem: AbstractFileSystem, file_path: Union[str, Path], model_class: type[T]) -> T:
    """Load a pydantic model from a json file

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to use to load the model.
    file_path : Union[str, Path]
        The path to the json file to load the model from.
    model_class : type[T]
        The pydantic model class to load.

    Returns
    -------
    T
        The loaded pydantic model instance.
    """
    with filesystem.open(str(file_path), "r") as f:
        data = f.read()
    return model_class.model_validate_json(data)


def load_numpy_filesystem(filesystem: AbstractFileSystem, file_path: Union[str, Path]) -> np.ndarray:
    """Load a numpy file from a file system

    Loads a file from the file system used by the interface. The file_path is expected to be
    relative to the filesystem.
    uses np.load(f)

    Parameter
    ---------
    filesystem: AbstractFileSystem
        The filesystem to load the numpy file from.
    file_path: Union[str, Path]
        The path relative to the filesystem.

    Returns
    -------
    np.ndarray
        array, tuple, dict, etc.
        Data stored in the file. For ``.npz`` files, the returned instance
        of NpzFile class must be closed to avoid leaking file descriptors.
        See npyio.py

    """
    with filesystem.open(str(file_path), "rb") as f:
        return np.load(f)


def save_numpy_filesystem(
    filesystem: AbstractFileSystem, file_path: Union[str, Path], numpy_array: np.ndarray, make_dir: bool = True
) -> None:
    """Save a numpy array to a file in the filesystem

    Saves a numpy array to a file to the file system used by the interface. The file_path is expected to be
    relative to the filesystem.
    uses np.save(f)

    Parameter
    ---------
    filesystem: AbstractFileSystem
        The filesystem to load the numpy file from.
    file_path: Union[str, Path]
        The path relative to the filesystem.
    numpy_array:np.ndarray
        The numpy array to save
    make_dir: bool
        Whether to create the directory if it does not exist.
    """
    if make_dir:
        filesystem.makedirs(Path(file_path).parent.as_posix(), exist_ok=True)
    with filesystem.open(str(file_path), "wb") as f:
        np.save(f, numpy_array)


def copy_file_fs(
    src_fs: AbstractFileSystem,
    src_path: Union[str, Path],
    dest_fs: AbstractFileSystem,
    dest_path: Union[str, Path],
    make_dir: bool = True,
) -> None:
    """Copy a file from one filesystem to another.

    Parameters
    ----------
    src_fs: AbstractFileSystem
        The source filesystem.
    src_path: Union[str, Path]
        The path to the file in the source filesystem.
    dest_fs: AbstractFileSystem
        The destination filesystem.
    dest_path: Union[str, Path]
        The path to the file in the destination filesystem.
    make_dir: bool
        create parent folder if not exists.

    """
    # create parent directories
    if make_dir:
        dest_fs.makedirs(Path(dest_path).parent.as_posix(), exist_ok=True)
    with src_fs.open(str(src_path), "rb") as src_file:
        with dest_fs.open(str(dest_path), "wb") as dest_file:
            shutil.copyfileobj(src_file, dest_file)
