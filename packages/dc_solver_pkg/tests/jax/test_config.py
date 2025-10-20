import json
import os
from dataclasses import replace

from toop_engine_dc_solver.jax.config import (
    default_config,
    read_config_from_file,
    save_config,
)


def test_config(tmp_path: str) -> None:
    filename = os.path.join(tmp_path, "config.json")
    config = default_config()

    assert not os.path.exists(filename)
    save_config(config, filename)
    assert os.path.exists(filename)

    config2 = read_config_from_file(filename)
    assert config == config2


def test_save_read(tmp_path: str) -> None:
    filename = os.path.join(tmp_path, "config.json")

    config = default_config()
    config = replace(config, single_precision=True, batch_size_bsdf=19, batch_size_injection=43)
    save_config(config, filename)
    config2 = read_config_from_file(filename)

    assert config2 == config


def test_empty_file(tmp_path: str) -> None:
    filename = os.path.join(tmp_path, "config.json")

    with open(filename, "w") as f:
        json.dump({}, f)

    config = read_config_from_file(filename)
    assert config == default_config()
