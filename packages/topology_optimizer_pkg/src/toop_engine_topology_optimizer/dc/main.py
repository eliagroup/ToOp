# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Launcher for the Map-Elites optimizer.

example args:

--fixed_files /workspaces/AICoE_HPC_RL_Optimizer/data/static_information.hdf5 \
--stats_dir /workspaces/AICoE_HPC_RL_Optimizer/stats/ \
--tensorboard_dir /workspaces/AICoE_HPC_RL_Optimizer/stats/tensorboard/ \
--ga_config.target_metrics overload_energy_n_1 1.0 # The metrics to optimize with their weight \
--ga_config.me_descriptors.0.metric split_subs \
--ga_config.me_descriptors.0.num_cells 5 \
--ga_config.me_descriptors.1.metric switching_distance \
--ga_config.me_descriptors.1.num_cells 45 # The metrics to use as descriptors along with their maximum value \
--ga_config.observed_metrics overload_energy_n_1 split_subs switching_distance \
    # All the relevant metrics, including target, descriptors and extra metrics you want to see in the report \
--ga_config.substation_unsplit_prob 0.5 # Instinctively better suited for mapelites \
--ga_config.substation_split_prob 0.5 # Instinctively better suited for mapelites \
--ga_config.plot # Enable plot generation and saving in stats_dir/plots/ \
--ga_config.iterations_per_epoch 50 # Basically how often you wanna get a report. Suggested range : 50 - 1000 \
--ga_config.runtime_seconds 300 # How many seconds to run the optimization for \

The first descriptor metric is the vertical axis, the second the horizontal axis.
"""

import datetime
import json
import os
import sys
import time
from functools import partial
from typing import Optional

import jax
import logbook
import numpy as np
import tyro
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseModel, Field
from tensorboardX import SummaryWriter
from toop_engine_interfaces.types import MetricType
from toop_engine_topology_optimizer.dc.ga_helpers import (
    EmitterState,
    init_running_means,
    update_running_means,
)
from toop_engine_topology_optimizer.dc.genetic_functions.initialization import (
    get_repertoire_metrics,
)
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import summarize
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import (
    DiscreteMapElitesRepertoire,
)
from toop_engine_topology_optimizer.dc.repertoire.plotting import plot_repertoire
from toop_engine_topology_optimizer.dc.worker.optimizer import OptimizerData, initialize_optimization, run_epoch
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    DoubleLimitsSetpoint,
    LoadflowSolverParameters,
)
from tqdm import tqdm

logger = logbook.Logger(__name__)


class CLIArgs(BaseModel):
    """The arguments for a CLI invocation, mostly equal to OptimizationStartCommand."""

    ga_config: BatchedMEParameters = Field(default_factory=BatchedMEParameters)
    """The configuration for the genetic algorithm"""

    lf_config: LoadflowSolverParameters = Field(default_factory=LoadflowSolverParameters)
    """The configuration for the loadflow solver"""

    fixed_files: tuple[str, ...] = ()
    """The file containing the static information. You can pass multiple files here"""

    tensorboard_dir: str = "tensorboard"
    """The directory to store the tensorboard logs"""

    stats_dir: str = "stats"
    """The directory to store the json summaries"""

    summary_frequency: int = 10
    """How often to write tensorboard summaries"""

    checkpoint_frequency: int = 100
    """How often to write json summaries"""

    double_limits: tuple[float, float] = (1.0, 1.0)
    """
    The double limits to use in the form (lower_limit, upper_limit).
    lower_limit: float
        The relative lower limit to set, for branches whose n-1 flows are below the lower limit
    upper_limit: float
        The relative upper_limit determining at what relative load a branch is considered overloaded.
        Branches in the band between lower and upper limit are considered overloaded if more load is added.
    """


def log_tensorboard(
    fitness: float,
    metrics: dict[MetricType, float],
    iteration: int,
    writer: SummaryWriter,
) -> None:
    """Log fitness and metrics to tensorboard.

    Parameters
    ----------
    fitness : float
        The fitness value
    metrics : dict[MetricType, float]
        The metrics to log
    iteration : int
        The current iteration number
    writer : SummaryWriter
        The tensorboard writer
    """
    writer.add_scalar("fitness", fitness, iteration)
    for key, value in metrics.items():
        writer.add_scalar(key, value, iteration)


def write_summary(
    optimizer_data: OptimizerData,
    repertoire: DiscreteMapElitesRepertoire,
    emitter_state: EmitterState,
    iteration: Optional[int],
    folder: str,
    args_dict: dict,
    n_cells_per_dim: tuple[int, ...],
    descriptor_metrics: tuple[str, ...],
    plot: bool,
    processed_gridfile_fs: AbstractFileSystem,
) -> dict:
    """Write a summary to a json file.

    Watch out : here, the optimizer data is out of sync with the jax_data

    Parameters
    ----------
    optimizer_data : OptimizerData
        The optimizer data of the optimization. This includes a jax_data, however as the jax_data
        is updated per-iteration and the optimizer_data only per-epoch, we need to pass the
        jax_data separately
    repertoire : DiscreteMapElitesRepertoire
        The current repertoire for this iteration, will be used instead of the one in the optimizer_data
    emitter_state : EmitterState
        The emitter state for this iteration, will be used instead of the one in the optimizer_data
    iteration : Optional[int]
        The iteration number, if None, will write to res.json, otherwise to res_{iteration}.json
    folder : str
        The folder to write the summary to, relative to the processed_gridfile_fs
    args_dict : dict
        The arguments used for invocation in a dict format, will be added to the summary for
        documentation purposes
    n_cells_per_dim : tuple[int, ...]
        The number of cells per dimension for MAP-Elites
    descriptor_metrics : tuple[str, ...]
        The descriptor metrics to use for MAP-Elites
    plot : bool
        Whether to plot the repertoire
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder was created
        which will be completed with the preprocess call to this function.
        Internally, only the data folder is passed around as a dirfs.
        Note that the unprocessed_gridfile_fs is not needed here anymore, as all preprocessing steps that need the
        unprocessed gridfiles were already done.

    Returns
    -------
    dict
        The summary that was written
    """
    processed_gridfile_fs.makedirs(folder, exist_ok=True)

    # Here we assume that contingency_ids are the same for all topos in the repertoire
    contingency_ids = optimizer_data.solver_configs[0].contingency_ids
    summary = summarize(
        repertoire=repertoire,
        emitter_state=emitter_state,
        initial_fitness=optimizer_data.initial_fitness,
        initial_metrics=optimizer_data.initial_metrics,
        contingency_ids=contingency_ids,
    )
    summary.update(
        {
            "args": args_dict,
            "iteration": iteration,
        }
    )
    filename = "res.json" if iteration is None else f"res_{iteration}.json"
    with processed_gridfile_fs.open(os.path.join(folder, filename), "w") as f:
        json.dump(summary, f)
    if plot:
        plot_repertoire(
            repertoire.fitnesses[
                : np.prod(n_cells_per_dim)
            ],  # Only take the first depth layer. The best fitnesses are in the first depth layer already.
            iteration,
            folder,
            n_cells_per_dim=n_cells_per_dim,
            descriptor_metrics=descriptor_metrics,
            save_plot=plot,
        )
    return summary


def main(
    args: CLIArgs,
    processed_gridfile_fs: AbstractFileSystem,
) -> dict:
    """Run main optimization function for CLI execution.

    Parameters
    ----------
    args : CLIArgs
        The arguments for the optimization
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder was created
        which will be completed with the preprocess call to this function.
        Internally, only the data folder is passed around as a dirfs.
        Note that the unprocessed_gridfile_fs is not needed here anymore, as all preprocessing steps that need the
        unprocessed gridfiles were already done.

    Returns
    -------
    dict
        The final results of the optimization
    """
    jax.config.update("jax_enable_x64", True)

    logger.info(f"Starting with config {args}")

    args_dict = args.model_dump()

    partial_write_summary = partial(
        write_summary,
        folder=args.stats_dir,
        n_cells_per_dim=[desc.num_cells for desc in args.ga_config.me_descriptors],
        descriptor_metrics=[desc.metric for desc in args.ga_config.me_descriptors],
        plot=args.ga_config.plot,
        args_dict=args_dict,
        processed_gridfile_fs=processed_gridfile_fs,
    )

    optimizer_data, stats, initial_topology = initialize_optimization(
        params=DCOptimizerParameters(
            ga_config=args.ga_config,
            loadflow_solver_config=args.lf_config,
            summary_frequency=args.summary_frequency,
            check_command_frequency=args.summary_frequency,
            double_limits=DoubleLimitsSetpoint(lower=args.double_limits[0], upper=args.double_limits[1])
            if args.double_limits != (1.0, 1.0)
            else None,
        ),
        optimization_id="CLI",
        static_information_files=args.fixed_files,
        processed_gridfile_fs=processed_gridfile_fs,
    )

    running_means = init_running_means(
        n_outages=optimizer_data.jax_data.dynamic_informations[0].n_nminus1_cases,
        n_devices=len(jax.devices()) if args.lf_config.distributed else 1,
    )

    logger.info(f"Optimization started: {stats}")

    writer = SummaryWriter(f"{args.tensorboard_dir}/{datetime.datetime.now()}")
    writer.add_hparams(args_dict, {})

    # Log initial results
    log_tensorboard(
        fitness=initial_topology.timesteps[0].metrics.fitness,
        metrics=initial_topology.timesteps[0].metrics.extra_scores,
        iteration=0,
        writer=writer,
    )

    epoch = 0
    with tqdm(
        total=args.ga_config.runtime_seconds,
        bar_format="{l_bar}{bar}[{elapsed}<{remaining}]{postfix}",
        postfix={
            "f0": optimizer_data.initial_fitness,
            "f": optimizer_data.initial_fitness,
            "br/s": 0,
            "inj/s": 0,
            "lfs": 0,
            "epoch": 0,
        },
    ) as pbar:
        while time.time() - running_means.start_time < args.ga_config.runtime_seconds:
            optimizer_data = run_epoch(optimizer_data)

            with jax.default_device(jax.devices("cpu")[0]):
                repertoire = (
                    jax.tree_util.tree_map(lambda x: x[0], optimizer_data.jax_data.repertoire)
                    if args.lf_config.distributed
                    else optimizer_data.jax_data.repertoire
                )
                emitter_state = (
                    jax.tree_util.tree_map(lambda x: x[0], optimizer_data.jax_data.emitter_state)
                    if args.lf_config.distributed
                    else optimizer_data.jax_data.emitter_state
                )

                fitness, metrics = get_repertoire_metrics(repertoire, args.ga_config.observed_metrics)
                log_tensorboard(fitness, metrics, epoch, writer)
                partial_write_summary(
                    optimizer_data,
                    repertoire,
                    emitter_state,
                    iteration=epoch,
                )

                running_means = update_running_means(running_means=running_means, emitter_state=emitter_state)
            pbar.update(time.time() - running_means.start_time - pbar.n)
            pbar.set_postfix(
                {
                    "f0": optimizer_data.initial_fitness,
                    "f": fitness,
                    "br/s": running_means.br_per_sec.get(),
                    "inj/s": running_means.inj_per_sec.get(),
                    "lfs": running_means.total_inj_combis * running_means.n_outages,
                    "epoch": epoch,
                }
            )
            epoch += 1

    final_results = partial_write_summary(
        optimizer_data,
        jax.tree_util.tree_map(lambda x: x[0], optimizer_data.jax_data.repertoire)
        if args.lf_config.distributed
        else optimizer_data.jax_data.repertoire,
        jax.tree_util.tree_map(lambda x: x[0], optimizer_data.jax_data.emitter_state)
        if args.lf_config.distributed
        else optimizer_data.jax_data.emitter_state,
        iteration=None,
    )
    return final_results


if __name__ == "__main__":
    logbook.StreamHandler(sys.stdout, level=logbook.INFO).push_application()
    args = tyro.cli(CLIArgs)
    file_system = LocalFileSystem()
    main(args, file_system)
