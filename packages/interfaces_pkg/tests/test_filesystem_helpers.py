import numpy as np
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_interfaces.asset_topology import Busbar
from toop_engine_interfaces.filesystem_helper import (
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


def test_save_load_numpy_filesystem(tmp_path):
    fs = DirFileSystem(tmp_path)
    numpy_array = np.random.rand(5)

    file_path = "numpy/test_numpy_array.npy"
    save_numpy_filesystem(filesystem=fs, file_path=file_path, numpy_array=numpy_array)
    loaded_numpy_array = load_numpy_filesystem(filesystem=fs, file_path=file_path)
    assert np.array_equal(loaded_numpy_array, numpy_array)
