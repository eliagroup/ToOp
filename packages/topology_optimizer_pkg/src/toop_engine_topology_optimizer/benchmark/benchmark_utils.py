"""
A module that contains methods to run the full end-to-end pipeline from grid file to optimization results.

Note that these methods are simply wrappers to use existing functionality in a sequence.
"""

from __future__ import annotations

import json
import os
import shutil

# Keep warnings under control
import warnings
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal, Optional, Tuple

import jax
import logbook
import pandapower

# Domain-specific imports (may raise if not available in the environment)
import pypowsybl
from omegaconf import DictConfig
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_dc_solver.postprocess.postprocess_pandapower import (
    PandapowerRunner,
)
from toop_engine_dc_solver.postprocess.postprocess_pandapower import (
    apply_disconnections as pandapower_apply_disconnections,
)
from toop_engine_dc_solver.postprocess.postprocess_pandapower import (
    apply_topology as pandapower_apply_topology,
)
from toop_engine_dc_solver.postprocess.postprocess_powsybl import (
    PowsyblRunner,
)
from toop_engine_dc_solver.postprocess.postprocess_powsybl import (
    apply_disconnections as powsybl_apply_disconnections,
)
from toop_engine_dc_solver.postprocess.postprocess_powsybl import (
    apply_topology as powsybl_apply_topology,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_grid_helpers.powsybl.single_line_diagram.get_single_line_diagram_custom import (
    get_single_line_diagram_custom,
)
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_interfaces.folder_structure import (
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    PreprocessParameters,
    UcteImporterParameters,
)
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import (
    empty_status_update_fn,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import StaticInformationStats
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition
from toop_engine_interfaces.stored_action_set import ActionSet
from toop_engine_topology_optimizer.dc.main import CLIArgs
from toop_engine_topology_optimizer.dc.main import main as opt_main

# Local project imports
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    LoadflowSolverParameters,
)
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

# JAX configuration
jax.config.update("jax_enable_x64", True)

# Logging setup
logger = logbook.Logger(__name__)
logbook.StderrHandler().push_application()


@dataclass
class PipelineConfig:
    """Configuration class for the pipeline, specifying file paths and runtime parameters."""

    root_path: Path = Path("./")
    """Root directory for data storage."""

    iteration_name: str = ""
    """Name of the current iteration or experiment."""

    file_name: str = "grid.xiidm"
    """Name of the primary data file, a ZIP archive for CGMES or an xiidm file."""

    static_info_relpath: str = PREPROCESSING_PATHS["static_information_file_path"]
    """Relative path to the static information file, as defined in PREPROCESSING_PATHS."""

    initial_topology_subpath: str = PREPROCESSING_PATHS["initial_topology_path"]
    """Relative path to the initial topology file, as defined in PREPROCESSING_PATHS."""

    omp_num_threads: int = 1
    """Number of OpenMP threads to use for parallel processing."""

    num_cuda_devices: int = 1
    """Number of CUDA devices (GPUs) to utilize."""

    grid_type: Literal["powsybl", "pandapower"] = "powsybl"
    """Type of grid model being used (power system analysis libraries)."""


def set_environment_variables(cfg: DictConfig) -> None:
    """Set environment variables for OpenMP, CUDA, and XLA based on the provided configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing the following keys:
        - omp_num_threads (int): Number of OpenMP threads.
        - num_cuda_devices (int): Number of CUDA devices.
        - xla_force_host_platform_device_count (Optional[int]): Number of XLA host platform devices.

    Returns
    -------
    None
    """
    os.environ["OMP_NUM_THREADS"] = str(cfg["omp_num_threads"])
    logger.debug(f"Set OMP_NUM_THREADS to {os.environ['OMP_NUM_THREADS']}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in range(0, cfg["num_cuda_devices"])])
    logger.debug(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")

    if cfg["xla_force_host_platform_device_count"] is not None:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cfg['xla_force_host_platform_device_count']}"
        logger.debug(f"Set XLA_FLAGS to {os.environ.get('XLA_FLAGS', 'Not Set')}")


def run_task_process(cfg: DictConfig, conn: Optional[Connection] = None) -> Optional[dict]:
    """Execute a task process based on the provided configuration.

    This function initializes parameters for a genetic algorithm and a load flow solver
    from the given configuration, constructs command-line arguments, and runs the main
    optimization function. If a connection object is provided, the result is sent through
    the connection; otherwise, the result is returned directly.
    Args:
        cfg (DictConfig): Configuration object containing parameters for the task.
        conn (Connection, optional): Connection object for inter-process communication. Defaults to None.

    Returns
    -------
    Optional[dict]
        The result of the optimization process if no connection is provided; otherwise, None.
    """
    logger.info(f"************Experiment name: {cfg['task_name']}****************")

    try:
        ga_params = BatchedMEParameters(**{k: v for k, v in cfg.ga_config.items() if v is not None})
        lf_params = LoadflowSolverParameters(**{k: v for k, v in cfg.lf_config.items() if v is not None})
    except (TypeError, ValueError) as e:
        error_msg = f"Invalid configuration parameters: {e}"
        logger.error(error_msg)
        if conn:
            conn.send({"error": error_msg})
            conn.close()
            return None
        raise

    if "{task_name}" in cfg.tensorboard_dir:
        cfg.tensorboard_dir = cfg.tensorboard_dir.replace("{task_name}", cfg.task_name)

    if "{task_name}" in cfg.stats_dir:
        cfg.stats_dir = cfg.stats_dir.replace("{task_name}", cfg.task_name)

    # Combine all config data into the CLIArgs Pydantic model
    cli_args = {
        "ga_config": ga_params,
        "lf_config": lf_params,
        "fixed_files": tuple(cfg.fixed_files) if cfg.fixed_files is not None else None,
        "double_precision": cfg.double_precision if cfg.double_precision is not None else None,
        "tensorboard_dir": cfg.tensorboard_dir if cfg.tensorboard_dir is not None else None,
        "stats_dir": cfg.stats_dir if cfg.stats_dir is not None else None,
        "summary_frequency": cfg.summary_frequency if cfg.summary_frequency is not None else None,
        "checkpoint_frequency": cfg.checkpoint_frequency if cfg.checkpoint_frequency is not None else None,
        "double_limits": tuple(cfg.double_limits) if cfg.double_limits is not None else None,
    }

    # Remove None values
    cli_args = {key: value for key, value in cli_args.items() if value is not None}
    cli_args = CLIArgs(**cli_args)

    try:
        result = opt_main(cli_args)
    except KeyboardInterrupt:
        # Let it propagate
        raise
    except (RuntimeError, ValueError, OSError) as e:
        # Only catch specific exceptions we know how to handle
        error_msg = f"Optimization failed: {e}"
        logger.error(error_msg)
        if conn:
            conn.send({"error": error_msg})
            conn.close()
            return None
        raise

    if conn:
        try:
            conn.send(result)
        except (BrokenPipeError, ConnectionError) as e:
            logger.warning(f"Failed to send result through connection: {e}")
        finally:
            conn.close()
        return None
    return result


def get_paths(cfg: PipelineConfig) -> Tuple[Path, Path, Path, Path]:
    """Generate and return key file and directory paths for a pipeline iteration.

    This function constructs paths for the iteration directory, a specific file within that directory,
    a data folder based on the file name, and an optimizer snapshot directory within the data folder.
    It ensures that the optimizer snapshot directory exists, and raises an error if the specified file does not exist.

    Parameters
    ----------
    cfg : PipelineConfig
        Configuration object containing root_path, iteration_name, and file_name attributes.

    Returns
    -------
    Tuple[Path, Path, Path, Path]
        A tuple containing:
            - iteration_path (Path): Path to the iteration directory.
            - file_path (Path): Path to the specified file within the iteration directory.
            - data_folder (Path): Path to the data folder derived from the file name.
            - optimizer_snapshot_dir (Path): Path to the optimizer snapshot directory within the data folder.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist at the constructed file_path.
    """
    iteration_path = cfg.root_path / cfg.iteration_name
    if ".zip" in cfg.file_name:
        file_path = cfg.root_path / cfg.file_name
        data_folder = cfg.root_path / file_path.stem
        data_folder.mkdir(parents=True, exist_ok=True)
    else:
        file_path = iteration_path / cfg.file_name
        data_folder = iteration_path / file_path.stem

    optimizer_snapshot_dir = data_folder / "optimizer_snapshot"
    optimizer_snapshot_dir.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        raise FileNotFoundError(f"grid file not found: {file_path.resolve()}")

    return iteration_path, file_path, data_folder, optimizer_snapshot_dir


def prepare_importer_parameters(
    file_path: Path, data_folder: Path, area_settings: Optional[AreaSettings] = None
) -> UcteImporterParameters | CgmesImporterParameters:
    """
    Build importer parameter objects depending on file suffix.

    Parameters
    ----------
    file_path : Path
        The path to the grid model file. The file suffix determines the type of importer parameters to create.
    data_folder : Path
        The path to the folder containing additional data required for the import process.
    area_settings : Optional[AreaSettings], optional
        The area settings to use for the importer. If not provided, a default AreaSettings object is created.

    Returns
    -------
    UcteImporterParameters | CgmesImporterParameters
        An instance of either `UcteImporterParameters` or `CgmesImporterParameters`, depending on the file suffix.

    Notes
    -----
    If `file_path` has a `.uct` suffix, a `UcteImporterParameters` object is returned.
    Otherwise, a `CgmesImporterParameters` object is returned.
    """
    if area_settings is None:
        area_settings = AreaSettings(
            cutoff_voltage=10,
            control_area=[""],
            view_area=[""],
            nminus1_area=[""],
            cross_border_limits_n0=None,
            cross_border_limits_n1=None,
        )

    importer_parameter_dict = {
        "grid_model_file": file_path,
        "data_folder": data_folder,
        "white_list_file": None,
        "black_list_file": None,
        "area_settings": area_settings,
    }

    if file_path.suffix == ".uct":
        return UcteImporterParameters(**importer_parameter_dict)
    return CgmesImporterParameters(**importer_parameter_dict)


def run_preprocessing(
    importer_parameters: UcteImporterParameters | CgmesImporterParameters,
    data_folder: Path,
    preprocessing_parameters: PreprocessParameters,
    is_pandapower_net: bool = False,
) -> tuple[StaticInformationStats, StaticInformation]:
    """
    Run importer preprocessing and extract static information.

    Parameters
    ----------
    importer_parameters : UcteImporterParameters | CgmesImporterParameters
        Parameters required by the importer for file conversion.
    data_folder : Path
        Path to the folder where data files are stored and processed.
    preprocessing_parameters : PreprocessParameters
        Parameters for preprocessing and loading the grid.
    is_pandapower_net : bool, optional
        Whether to use pandapower for loading the grid. Default is False.

    Returns
    -------
    info : StaticInformationStats
        Statistics and metadata about the static information extracted from the grid.
    static_information : StaticInformation
        The extracted static information from the grid.
    """
    logger.info("Starting file conversion via preprocessing.convert_file")
    import_result = preprocessing.convert_file(
        importer_parameters=importer_parameters, status_update_fn=empty_status_update_fn
    )

    tqdm.write(
        (
            f"Converted {importer_parameters.grid_model_file.stem} - "
            f"subs: {import_result.n_relevant_subs}, "
            f"lines n-1: {import_result.n_line_for_nminus1}, "
            f"trafos n-1: {import_result.n_trafo_for_nminus1}, "
            f"lines reward: {import_result.n_line_for_reward}, "
            f"trafos reward: {import_result.n_trafo_for_reward}, "
            f"lines disconnectable: {import_result.n_line_disconnectable}, "
            f"low impedance lines: {import_result.n_low_impedance_lines}, "
            f"branches across switch: {import_result.n_branch_across_switch}"
        )
    )

    jax.clear_caches()
    info, static_information, _ = load_grid(
        data_folder=data_folder,
        pandapower=is_pandapower_net,
        status_update_fn=empty_status_update_fn,
        parameters=preprocessing_parameters,
    )

    logger.info(", ".join([f"{k}: {v}" for k, v in dict(info).items()]))

    # Create zip archives for convenience (keeps original behaviour)
    file_to_zip = data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    zip_path = data_folder / "static_information.zip"
    shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", root_dir=file_to_zip.parent, base_dir=file_to_zip.name)

    logger.notice("Preprocessing completed.")
    return info, static_information


def run_dc_optimization(dc_optim_config: dict) -> dict:
    """
    Run the DC optimization process with the specified configuration.

    This function sets up the necessary environment variables for optimization,
    including the number of OpenMP threads, CUDA devices, and optionally the XLA
    host platform device count. It then calls the optimization routine with the
    provided configuration.

    Parameters
    ----------
    dc_optim_config : DictConfig
        Configuration object containing optimization parameters. Expected keys:
        - "omp_num_threads": int, number of OpenMP threads to use.
        - "num_cuda_devices": int, number of CUDA devices to make visible.
        - "xla_force_host_platform_device_count": int or None, optional XLA device count.

    Returns
    -------
    dict
        The result of the optimization process.
    """
    #  Set the env variables
    set_environment_variables(dc_optim_config)

    res = run_task_process(DictConfig(dc_optim_config))
    return res


def load_action_set(action_set_path: Path) -> ActionSet:
    """
    Load an ActionSet object from a JSON file.

    Parameters
    ----------
    action_set_path : Path
        The file path to the JSON file containing the action set definition.

    Returns
    -------
    ActionSet
        An instance of ActionSet initialized with the data from the JSON file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file is not a valid JSON.
    TypeError
        If the JSON structure does not match the ActionSet constructor requirements.
    """
    with open(action_set_path, "r") as f:
        action_set_json = json.load(f)
    return ActionSet(**action_set_json)


def apply_topology_and_save(
    grid_path: Path,
    actions: list[Optional[int]],
    disconnections: list[Optional[int]],
    action_set: ActionSet,
    save_path: Path,
    is_pandapower_grid: bool = False,
) -> pypowsybl.network.Network:
    """
    Apply a set of topology actions and disconnections to the grid, then saves the modified network to a specified path.

    Parameters
    ----------
    grid_path : Path
        Path to the input grid file to be loaded.
    actions : iterable[Optional[int]]
        Iterable of actions to be applied to the network topology.
    disconnections : iterable[Optional[int]]
        Iterable of disconnections to be applied to the network.
    action_set : ActionSet
        Set of possible actions that can be applied to the network.
    save_path : Path
        Path where the modified network will be saved.
    is_pandapower_grid : bool, optional
        Whether the grid is a pandapower grid. Default is False (powsybl grid).

    Returns
    -------
    pypowsybl.network.Network
        The modified network after applying the topology changes and disconnections.
    """
    # Load grid
    base_net = pypowsybl.network.load(grid_path) if not is_pandapower_grid else pandapower.from_json(grid_path)

    # Apply topology and disconnections
    apply_topology = pandapower_apply_topology if is_pandapower_grid else powsybl_apply_topology
    apply_disconnections = pandapower_apply_disconnections if is_pandapower_grid else powsybl_apply_disconnections

    modified_net, _ = apply_topology(net=base_net, actions=actions, action_set=action_set)
    modified_net = apply_disconnections(modified_net, disconnections=disconnections, action_set=action_set)

    modified_net.save(save_path) if not is_pandapower_grid else modified_net.to_json(save_path)
    logger.notice(f"Saved modified network to {save_path}")
    return modified_net


def calculate_and_save_loadflow_results(
    runner: AbstractLoadflowRunner,
    topology_path: Path,
    actions: Optional[list[int]] = None,
    disconnections: Optional[list[int]] = None,
) -> None:
    """
    Calculate AC and DC loadflow results using the provided runner and saves the results as CSV files.

    Parameters
    ----------
    runner : AbstractLoadflowRunner
        An instance capable of running AC and DC loadflow calculations.
    topology_path : Path
        The directory path where the result CSV files will be saved.
    actions : Optional[list[int]], optional
        A list of action indices to apply during the loadflow calculation.
    disconnections : Optional[list[int]], optional
        A list of disconnection indices to apply during the loadflow calculation.

    Returns
    -------
    None
        This function does not return anything. It saves the results to CSV files:
        - "ac_loadflow_results.csv"
        - "dc_loadflow_results.csv"
        in the specified `topology_path` directory.
    """
    if actions is None:
        actions = []
    if disconnections is None:
        disconnections = []
    ac_loadflow_results = runner.run_ac_loadflow(actions, disconnections)
    ac_loadflow_results.branch_results.collect().write_csv(topology_path / "ac_loadflow_results.csv")
    dc_loadflow_results = runner.run_dc_loadflow(actions, disconnections)
    dc_loadflow_results.branch_results.collect().write_csv(topology_path / "dc_loadflow_results.csv")


def create_loadflow_runner(
    data_folder: Path, grid_file_path: Path, n_processes: int = 1, pandaflow_runner: bool = False
) -> AbstractLoadflowRunner:
    """
    Create and configure a PowsyblRunner instance for loadflow analysis.

    This function initializes a `PowsyblRunner` with a single process, loads the n-1 definition and action set
    from the specified `data_folder`, and loads the base grid from the provided `grid_file_path`.
    If the required files are not found, it raises a `FileNotFoundError`.

    Parameters
    ----------
    data_folder : Path
        The directory containing the n-1 definition and action set files.
    grid_file_path : Path
        The path to the base grid file to be loaded.
    n_processes : int, optional
        The number of processes to use for the loadflow runner. Default is 1.
    pandaflow_runner : bool, optional
        Whether to use a PandapowerRunner instead of PowsyblRunner. Default is False.

    Returns
    -------
    AbstractLoadflowRunner
        An instance of `PowsyblRunner` configured with the loaded n-1 definition, action set, and base grid.

    Raises
    ------
    FileNotFoundError
        If the n-1 definition file or action set file is not found in the specified `data_folder`.
    """
    runner = PowsyblRunner(n_processes=n_processes) if not pandaflow_runner else PandapowerRunner(n_processes=n_processes)
    n_minus1_def_path = data_folder / "nminus1_definition.json"
    if n_minus1_def_path.exists():
        logger.info(f"Loading n-1 definition from: {n_minus1_def_path}")
        n_minus_1_def = load_nminus1_definition(n_minus1_def_path)
        runner.store_nminus1_definition(n_minus_1_def)
    else:
        raise FileNotFoundError(f"N-1 definition file not found: {n_minus1_def_path}")

    runner.load_base_grid(grid_file_path)

    action_set_path = data_folder / PREPROCESSING_PATHS["action_set_file_path"]
    if action_set_path.exists():
        logger.info(f"Loading action set from: {action_set_path}")
        action_set = load_action_set(action_set_path)

    if action_set is not None:
        runner.store_action_set(action_set)
    else:
        raise FileNotFoundError(f"Action set file not found: {action_set_path}")

    return runner


def save_slds_of_split_stations(
    action_set: ActionSet, actions: list[int], output_dir: Path, network: pypowsybl.network.Network
) -> None:
    """
    Save single line diagrams (SLDs) as SVG files for a set of split stations.

    For each station specified by the given actions, this function generates a single line diagram (SLD)
    using the provided network and saves it as an SVG file in the specified output directory.

    Note that this method is applicable only for Powsybl networks.

    Parameters
    ----------
    action_set : ActionSet
        The set of available actions, containing local actions and their associated grid model IDs.
    actions : list of int
        List of action indices corresponding to the stations for which SLDs should be generated and saved.
    output_dir : Path
        The directory where the SLD SVG files will be saved. Files are saved under the 'sld' subdirectory.
    network : pypowsybl.network.Network
        The network object used to generate the single line diagrams.

    Returns
    -------
    None
        This function does not return anything. It saves SVG files to disk as a side effect.

    Notes
    -----
    - Each SLD is saved with the filename format '{station_id}_sld.svg' under the 'sld' subdirectory of `output_dir`.
    - Assumes that `get_single_line_diagram_custom` returns an object with a `_content` attribute containing SVG data.
    - Requires that the logger is properly configured.
    """
    split_stations = [
        (action_set.local_actions[action].grid_model_id, action_set.local_actions[action].name) for action in actions
    ]
    # Run ac loadflow
    pypowsybl.loadflow.run_ac(network)
    for station_id, station_name in split_stations:
        # Generate and save SLD for the station
        vl_id = network.get_buses(attributes=["voltage_level_id"]).loc[station_id, "voltage_level_id"]
        svg = get_single_line_diagram_custom(network, vl_id)
        sld_path = output_dir / "sld" / f"{station_name}_sld.svg"
        sld_path.parent.mkdir(parents=True, exist_ok=True)

        with open(sld_path, "w", encoding="utf-8") as f:
            f.write(svg._content)
        logger.info(f"Saved SLD for station {station_name} to {sld_path}")


def perform_ac_analysis(
    data_folder: Path, optimisation_run_path: Path, ac_validation_cfg: DictConfig, pandapower_runner: bool = False
) -> list[Path]:
    """Perform AC loadflow and n-1 analysis on the best k topologies from the optimization results.

    Parameters
    ----------
    data_folder : Path
        Path to the folder containing the input grid data (e.g., 'grid.xiidm').
    optimisation_run_path : Path
        Path to the optimizer snapshot directory containing 'res.json' and where results will be saved.
    ac_validation_cfg : DictConfig
        Configuration dictionary for AC validation, including:
        - 'n_processes': int, number of processes to use for loadflow analysis.
        - 'k_best_topos': int, number of best topologies to validate.
    pandapower_runner : bool, optional
        Whether to use a PandapowerRunner for loadflow analysis. Default is False (PowsyblRunner).

    Returns
    -------
    list[Path]
        List of paths to the directories where the results for the specified topologies are saved.
        If no paths are generated (e.g., no topologies found), returns an empty list.

    Raises
    ------
    FileNotFoundError
        If the result JSON file ('res.json') is not found in the specified optimisation_run_path.

    Notes
    -----
    This function applies the selected topology, runs AC loadflow analysis, and saves the results,
    including SLDs (Single Line Diagrams) of split stations, in the specified output directory.

    """
    res_json_path = optimisation_run_path / "res.json"
    grid_path = data_folder / "grid.xiidm"
    if not res_json_path.exists():
        raise FileNotFoundError(f"Result JSON file not found: {res_json_path}")
    with open(res_json_path, "r") as f:
        res = json.load(f)

    best_topos = res["best_topos"]
    logger.notice("Starting AC validation stage...")
    if len(best_topos) == 0 or best_topos is None:
        logger.warning("No topologies found in DC optimization results. Skipping AC analysis.")
        return []

    if ac_validation_cfg.get("k_best_topos", 1) > len(best_topos):
        logger.warning(
            f"Only {len(best_topos)} topologies available, you requested top {ac_validation_cfg.get('k_best_topos', 1)}."
        )

    n_assessed_topos = min(ac_validation_cfg.get("k_best_topos", 1), len(best_topos))
    logger.notice(f"Performing AC analysis on the top {n_assessed_topos} topologies...")

    topology_paths = []
    for topology_index in range(n_assessed_topos):
        topology_path = optimisation_run_path / f"topology_{topology_index}"
        topology_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Topology stored in: {topology_path}")

        actions = best_topos[topology_index].get("actions")
        disconnections = best_topos[topology_index].get("disconnections")

        out_modified = topology_path / "modified_network.xiidm"

        loadflow_runner = create_loadflow_runner(
            data_folder, grid_path, n_processes=ac_validation_cfg.get("n_processes", 1), pandaflow_runner=pandapower_runner
        )
        action_set = loadflow_runner.action_set

        logger.info("Applying topology and saving modified network...")
        modified_net = apply_topology_and_save(grid_path, actions, disconnections, action_set, out_modified)
        loadflow_runner.load_base_grid(out_modified)

        logger.notice("Running AC loadflow...")
        calculate_and_save_loadflow_results(loadflow_runner, topology_path, actions, disconnections)

        logger.info("Saving SLDs of split stations...")

        if not pandapower_runner:
            # Only for powsybl networks, we get the SLDs for the split stations
            save_slds_of_split_stations(action_set, actions, topology_path, modified_net)
        topology_paths.append(topology_path)
    logger.notice("AC validation completed.")
    return topology_paths


def run_dc_optimization_stage(
    dc_optim_config: dict,
    optimizer_snapshot_dir: Path,
) -> Path:
    """
    Run the DC optimization stage and save the results.

    Parameters
    ----------
    dc_optim_config : dict
        Configuration dictionary for the DC optimization stage.
    optimizer_snapshot_dir : Path
        Directory where the optimizer snapshot results will be saved.

    Returns
    -------
    Path
        Path to the directory of this run's results.
    """
    logger.notice("Running DC optimization...")
    optimisation_result = run_dc_optimization(dc_optim_config=dc_optim_config)
    logger.notice("DC optimization completed.")

    # Save optimizer results as res.json in optimizer_snapshot folder inside grid_path
    # Find next available run directory inside optimizer_snapshot_dir
    run_idx = 0
    while True:
        run_dir = optimizer_snapshot_dir / f"run_{run_idx}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            break
        run_idx += 1

    res_json_path = run_dir / "res.json"
    with open(res_json_path, "w") as f:
        json.dump(optimisation_result, f, indent=2, default=str)
    logger.notice(f"Saved optimizer results to {res_json_path}")
    return run_dir


def run_pipeline(
    pipeline_cfg: PipelineConfig,
    importer_parameters: UcteImporterParameters | CgmesImporterParameters,
    preprocessing_parameters: PreprocessParameters,
    dc_optim_config: DictConfig,
    ac_validation_cfg: DictConfig,
    run_preprocessing_stage: bool = True,
    run_optimization_stage: bool = True,
    run_ac_validation_stage: bool = True,
    optimisation_run_dir: Optional[Path] = None,
) -> list[Path]:
    """
    Run the end-to-end pipeline including topology copying, preprocessing, DC optimization, and AC validation.

    Parameters
    ----------
    pipeline_cfg : PipelineConfig
        Configuration object for the pipeline, containing paths and settings.
    importer_parameters : UcteImporterParameters | CgmesImporterParameters
        Parameters for the grid model importer, specifying the source grid model file and related options.
    preprocessing_parameters : PreprocessParameters
        Parameters for the preprocessing stage, controlling data preparation and transformation.
    dc_optim_config : DictConfig
        Configuration dictionary for the DC optimization stage.
    ac_validation_cfg: DictConfig,
        Configuration dictionary for the AC validation stage, including:
        - 'n_processes': int, number of processes to use for loadflow analysis.
        - 'k_best_topos': int, number of best topologies to validate.
    run_preprocessing_stage : bool, optional
        Whether to run the preprocessing stage. Default is True.
    run_optimization_stage : bool, optional
        Whether to run the DC optimization stage. Default is True.
    run_ac_validation_stage : bool, optional
        Whether to run the AC validation stage. Default is True.
    optimisation_run_dir : Optional[Path], optional
        If provided, this directory will be used for AC validation instead of running optimization.

    Returns
    -------
    Path
        Path to the directory where the results of the AC validation stage are saved.

    Notes
    -----
    - The function logs progress at each major step.
    - The optimizer results are saved as a JSON file in a new run directory under the optimizer snapshot directory.
    - AC validation is performed if optimization results are available.
    """
    logger.notice(f"Starting pipeline with config: {pipeline_cfg}")
    iteration_path, file_path, data_folder, optimizer_snapshot_dir = get_paths(pipeline_cfg)
    logger.notice(f"Paths resolved: iteration_path={iteration_path}, file_path={file_path}, data_folder={data_folder}")

    # Copy initial topology
    if run_preprocessing_stage:
        logger.notice("Running preprocessing stage...")
        run_preprocessing(
            importer_parameters,
            data_folder,
            preprocessing_parameters,
            is_pandapower_net=True if pipeline_cfg.grid_type == "pandapower" else False,
        )
    else:
        logger.notice("Skipping preprocessing stage as per configuration.")

    if run_optimization_stage:
        # DC Optimization stage. Can be skipped if results already exist. But if this is set to False, a valid
        # optimisation_run_dir must be provided for the AC validation stage. The run_dir should contain a res.json file.
        # which will be used by the AC validation stage.
        run_dir = run_dc_optimization_stage(
            dc_optim_config,
            optimizer_snapshot_dir,
        )
    else:
        logger.notice("Skipping DC optimization stage as per configuration.")
        logger.notice("Using run directory passed as input for AC validation stage.")
        run_dir = optimisation_run_dir
        if run_dir is None or (not run_dir.exists() and run_ac_validation_stage):
            raise ValueError(
                "No valid run directory provided for AC validation stage. "
                "Please provide the path of a valid directory where res.json exists."
            )

    topology_paths = []
    if run_ac_validation_stage:
        topology_paths = perform_ac_analysis(
            data_folder,
            run_dir,
            ac_validation_cfg=ac_validation_cfg,
            pandapower_runner=True if pipeline_cfg.grid_type == "pandapower" else False,
        )
    else:
        logger.notice("Skipping AC validation stage as per configuration.")
    logger.notice("Pipeline completed.")

    return topology_paths
