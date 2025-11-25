from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_interfaces.asset_topology import Busbar
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs, save_pydantic_model_fs


def test_save_load_pydantic_model_fs(tmp_path):
    fs = DirFileSystem(tmp_path)
    pydantic_model = Busbar(int_id=1, grid_model_id="busbar1")

    file_path = "pydantic/test_pydantic_class.json"
    save_pydantic_model_fs(filesystem=fs, file_path=file_path, pydantic_model=pydantic_model)
    loaded_model = load_pydantic_model_fs(filesystem=fs, file_path=file_path, model_class=Busbar)
    assert loaded_model == pydantic_model
