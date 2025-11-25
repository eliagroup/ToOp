"""Support functions for the AbstractFileSystem"""

from pathlib import Path
from typing import TypeVar

import numpy as np
from beartype.typing import Union
from fsspec import AbstractFileSystem
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_pydantic_model_fs(
    filesystem: AbstractFileSystem, file_path: Path, pydantic_model: BaseModel, indent: int = 2, make_dir: bool = True
) -> None:
    """Save an N-1 definition to a json file

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to use to save the N-1 definition.
    file_path : Path
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


def load_pydantic_model_fs(filesystem: AbstractFileSystem, file_path: Path, model_class: type[T]) -> T:
    """Load a pydantic model from a json file

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to use to load the model.
    file_path : Path
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
    file_pah: str
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
    file_pah: str
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
