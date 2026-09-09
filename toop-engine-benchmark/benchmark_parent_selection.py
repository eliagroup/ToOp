# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Run a DC-first benchmark study for all configured parent-selection modes.

Example
-------
uv run python toop-engine-benchmark/benchmark_parent_selection.py \
    ga.runtime_seconds=10 seeds=[3] ac_validation.enabled=false
"""

import logging
import os
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hydra
import structlog
from omegaconf import DictConfig, OmegaConf
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_topology_optimizer.benchmark.parent_selection_study import (
    ACValidationConfiguration,
    DescriptorResolutionConfiguration,
    GridSpecification,
    run_parent_selection_study,
)
from toop_engine_topology_optimizer.dc.repertoire.parent_selection import ParentSelectionMode
from toop_engine_topology_optimizer.interfaces.messages.dc_params import BatchedMEParameters, LoadflowSolverParameters

logger = structlog.get_logger(__name__)


def configure_benchmark_logging() -> None:
    """Limit benchmark output to progress bars and actionable errors."""
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s", force=True)
    for logger_name in ("absl", "hydra", "jax", "jaxlib", "matplotlib", "pandapower", "pypowsybl", "xla"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _as_dict(config: DictConfig) -> dict[str, Any]:
    """Resolve an OmegaConf mapping into a regular Python dictionary."""
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Expected a mapping configuration section.")
    return values


def _build_grid_specifications(config: DictConfig) -> tuple[GridSpecification, ...]:
    """Convert configured grid mappings into immutable study specifications."""
    specifications = []
    for grid_config in config.grids:
        specifications.append(
            GridSpecification(
                identifier=str(grid_config.identifier),
                source_path=Path(str(grid_config.source_path)),
                grid_type=cast("str", grid_config.get("grid_type", "powsybl")),
            )
        )
    return tuple(specifications)


@hydra.main(config_path="configs", config_name="parent_selection", version_base="1.2")
def main(config: DictConfig) -> None:
    """Run all configured grid, parent-selection mode, and seed combinations.

    Parameters
    ----------
    config : DictConfig
        Hydra study configuration.
    """
    configure_benchmark_logging()
    grids = _build_grid_specifications(config)
    modes = tuple(cast(ParentSelectionMode, str(mode)) for mode in config.modes)
    seeds = tuple(int(seed) for seed in config.seeds)
    output_directory = Path(str(config.output_directory))
    try:
        outcomes = run_parent_selection_study(
            grids=grids,
            modes=modes,
            seeds=seeds,
            output_directory=output_directory,
            ga_parameters=BatchedMEParameters(**_as_dict(config.ga)),
            loadflow_parameters=LoadflowSolverParameters(**_as_dict(config.lf)),
            preprocessing_parameters=PreprocessParameters(**_as_dict(config.preprocessing)),
            descriptor_resolution=DescriptorResolutionConfiguration(**_as_dict(config.descriptor_resolution)),
            ac_validation=ACValidationConfiguration(**_as_dict(config.ac_validation)),
            show_progress=bool(config.show_progress),
            stop_on_error=bool(config.stop_on_error),
        )
    except KeyboardInterrupt:
        raise
    except Exception as error:
        logger.error("Parent-selection study failed", error=str(error))
        raise SystemExit(1) from None
    logger.info(
        "Parent-selection study completed",
        output_directory=str(output_directory),
        completed_runs=len(outcomes),
    )


if __name__ == "__main__":
    main()
