from pathlib import Path

import numpy as np
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_interfaces.asset_topology import Busbar
from toop_engine_interfaces.filesystem_helper import (
    copy_file_fs,
    load_numpy_filesystem,
    load_pydantic_model_fs,
    save_numpy_filesystem,
    save_pydantic_model_fs,
)


def test_save_load_pydantic_model_fs(tmp_path):
    fs = DirFileSystem(tmp_path)
    pydantic_model = Busbar(int_id=1, grid_model_id="busbar1")

    file_path = "pydantic/test_pydantic_class.json"
    save_pydantic_model_fs(filesystem=fs, file_path=file_path, pydantic_model=pydantic_model)
    loaded_model = load_pydantic_model_fs(filesystem=fs, file_path=file_path, model_class=Busbar)
    assert loaded_model == pydantic_model

    # test Path
    file_path = Path("pydantic/test_pydantic_class.json")
    save_pydantic_model_fs(filesystem=fs, file_path=file_path, pydantic_model=pydantic_model)
    loaded_model = load_pydantic_model_fs(filesystem=fs, file_path=file_path, model_class=Busbar)
    assert loaded_model == pydantic_model


def test_save_load_numpy_filesystem(tmp_path):
    fs = DirFileSystem(tmp_path)
    numpy_array = np.random.rand(5)

    file_path = "numpy/test_numpy_array.npy"
    save_numpy_filesystem(filesystem=fs, file_path=file_path, numpy_array=numpy_array)
    loaded_numpy_array = load_numpy_filesystem(filesystem=fs, file_path=file_path)
    assert np.array_equal(loaded_numpy_array, numpy_array)

    # test Path
    file_path = Path("numpy/test_numpy_array.npy")
    save_numpy_filesystem(filesystem=fs, file_path=file_path, numpy_array=numpy_array)
    loaded_numpy_array = load_numpy_filesystem(filesystem=fs, file_path=file_path)
    assert np.array_equal(loaded_numpy_array, numpy_array)


def test_copy_file_with_make_dir_true(tmp_path):
    """Test copying a file with make_dir=True creates parent directories."""

    src_fs = tmp_path / "src"
    dest_fs = tmp_path / "dest"
    src_file = "file.txt"
    src_fs.mkdir()
    with open(src_fs / src_file, "w") as f:
        f.write("Test content")
    src_fsspec = DirFileSystem(src_fs)
    dest_fsspec = DirFileSystem(dest_fs)

    copy_file_fs(src_fs=src_fsspec, src_file=src_file, dest_fs=dest_fsspec, dest_file="subdir/file.txt", make_dir=True)
    assert (dest_fs / "subdir").exists()
    with open(dest_fs / "subdir/file.txt", "r") as f:
        content = f.read()
    assert content == "Test content"

    copy_file_fs(src_fs=src_fsspec, src_file=src_file, dest_fs=dest_fsspec, dest_file="file.txt", make_dir=True)
    assert (dest_fs / "file.txt").exists()
    with open(dest_fs / "file.txt", "r") as f:
        content = f.read()
    assert content == "Test content"

    # failing subdir not created
    with pytest.raises(FileNotFoundError):
        copy_file_fs(
            src_fs=src_fsspec, src_file=src_file, dest_fs=dest_fsspec, dest_file="nonexistent_dir/file.txt", make_dir=False
        )
