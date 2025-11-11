"""Factory methods for creating objects of classes defined in preprocess_results.py"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Union

from toop_engine_interfaces.messages.protobuf_schema.lf_service.stored_loadflow_reference_pb2 import StoredLoadflowReference
from toop_engine_interfaces.messages.protobuf_schema.preprocess.preprocess_results_pb2 import (
    ErrorResult,
    PowerFactoryImportResult,
    PreprocessingStartedResult,
    PreprocessingSuccessResult,
    Result,
    StaticInformationStats,
    UcteImportResult,
)
from toop_engine_interfaces.types import MetricType


def create_ucte_import_result(  # noqa: PLR0913
    data_folder: Path,
    n_relevant_subs: int = 0,
    n_low_impedance_lines: int = 0,
    n_branch_across_switch: int = 0,
    n_line_for_nminus1: int = 0,
    n_line_for_reward: int = 0,
    n_line_disconnectable: int = 0,
    n_trafo_for_nminus1: int = 0,
    n_trafo_for_reward: int = 0,
    n_trafo_disconnectable: int = 0,
    n_tie_line_for_reward: int = 0,
    n_tie_line_for_nminus1: int = 0,
    n_tie_line_disconnectable: int = 0,
    n_dangling_line_for_nminus1: int = 0,
    n_generator_for_nminus1: int = 0,
    n_load_for_nminus1: int = 0,
    n_switch_for_nminus1: int = 0,
    n_switch_for_reward: int = 0,
    n_white_list: Optional[int] = 0,
    n_white_list_applied: Optional[int] = 0,
    n_black_list: Optional[int] = 0,
    n_black_list_applied: Optional[int] = 0,
) -> UcteImportResult:
    """
    Create a UcteImportResult object with the provided preprocessing statistics.

    Parameters
    ----------
    data_folder : Path
        Path to the folder containing the data.
    n_relevant_subs : int, optional
        Number of relevant substations. Default is 0.
    n_low_impedance_lines : int, optional
        Number of low impedance lines. Default is 0.
    n_branch_across_switch : int, optional
        Number of branches across switches. Default is 0.
    n_line_for_nminus1 : int, optional
        Number of lines considered for N-1 analysis. Default is 0.
    n_line_for_reward : int, optional
        Number of lines considered for reward calculation. Default is 0.
    n_line_disconnectable : int, optional
        Number of disconnectable lines. Default is 0.
    n_trafo_for_nminus1 : int, optional
        Number of transformers considered for N-1 analysis. Default is 0.
    n_trafo_for_reward : int, optional
        Number of transformers considered for reward calculation. Default is 0.
    n_trafo_disconnectable : int, optional
        Number of disconnectable transformers. Default is 0.
    n_tie_line_for_reward : int, optional
        Number of tie lines considered for reward calculation. Default is 0.
    n_tie_line_for_nminus1 : int, optional
        Number of tie lines considered for N-1 analysis. Default is 0.
    n_tie_line_disconnectable : int, optional
        Number of disconnectable tie lines. Default is 0.
    n_dangling_line_for_nminus1 : int, optional
        Number of dangling lines considered for N-1 analysis. Default is 0.
    n_generator_for_nminus1 : int, optional
        Number of generators considered for N-1 analysis. Default is 0.
    n_load_for_nminus1 : int, optional
        Number of loads considered for N-1 analysis. Default is 0.
    n_switch_for_nminus1 : int, optional
        Number of switches considered for N-1 analysis. Default is 0.
    n_switch_for_reward : int, optional
        Number of switches considered for reward calculation. Default is 0.
    n_white_list : Optional[int], optional
        Number of items in the white list. Default is 0.
    n_white_list_applied : Optional[int], optional
        Number of white list items applied. Default is 0.
    n_black_list : Optional[int], optional
        Number of items in the black list. Default is 0.
    n_black_list_applied : Optional[int], optional
        Number of black list items applied. Default is 0.

    Returns
    -------
    UcteImportResult
        An instance of UcteImportResult containing the provided preprocessing statistics.
    """
    return UcteImportResult(
        data_folder=str(data_folder),
        n_relevant_subs=n_relevant_subs,
        n_low_impedance_lines=n_low_impedance_lines,
        n_branch_across_switch=n_branch_across_switch,
        n_line_for_nminus1=n_line_for_nminus1,
        n_line_for_reward=n_line_for_reward,
        n_line_disconnectable=n_line_disconnectable,
        n_trafo_for_nminus1=n_trafo_for_nminus1,
        n_trafo_for_reward=n_trafo_for_reward,
        n_trafo_disconnectable=n_trafo_disconnectable,
        n_tie_line_for_reward=n_tie_line_for_reward,
        n_tie_line_for_nminus1=n_tie_line_for_nminus1,
        n_tie_line_disconnectable=n_tie_line_disconnectable,
        n_dangling_line_for_nminus1=n_dangling_line_for_nminus1,
        n_generator_for_nminus1=n_generator_for_nminus1,
        n_load_for_nminus1=n_load_for_nminus1,
        n_switch_for_nminus1=n_switch_for_nminus1,
        n_switch_for_reward=n_switch_for_reward,
        n_white_list=n_white_list,
        n_white_list_applied=n_white_list_applied,
        n_black_list=n_black_list,
        n_black_list_applied=n_black_list_applied,
    )


def create_static_information_stats(  # noqa: PLR0913
    time: Optional[str] = None,
    fp_dtype: str = "",
    has_double_limits: bool = False,
    n_branches: int = 0,
    n_nodes: int = 0,
    n_branch_outages: int = 0,
    n_multi_outages: int = 0,
    n_injection_outages: int = 0,
    n_busbar_outages: int = 0,
    n_nminus1_cases: int = 0,
    n_controllable_psts: int = 0,
    n_monitored_branches: int = 0,
    n_timesteps: int = 0,
    n_relevant_subs: int = 0,
    n_disc_branches: int = 0,
    overload_energy_n0: float = 0.0,
    overload_energy_n1: float = 0.0,
    n_actions: int = 0,
    max_station_branch_degree: int = 0,
    max_station_injection_degree: int = 0,
    mean_station_branch_degree: float = 0.0,
    mean_station_injection_degree: float = 0.0,
    reassignable_branch_assets: int = 0,
    reassignable_injection_assets: int = 0,
    max_reassignment_distance: int = 0,
) -> StaticInformationStats:
    """
    Create stats about the static information class.

    Parameters
    ----------
    time : Optional[str], default=None
        The timestep that was optimized, if given.
    fp_dtype : str, default=""
        A string representation of the floating point type used in the static informations,
        e.g. 'float32' or 'float64'.
    has_double_limits : bool, default=False
        Whether the static information has max_mw_flow_limited set or not.
    n_branches : int, default=0
        The number of branches in the PTDF matrix.
    n_nodes : int, default=0
        The number of nodes in the PTDF matrix.
    n_branch_outages : int, default=0
        How many branch outages are part of the N-1 computation.
    n_multi_outages : int, default=0
        How many multi-outages are part of the N-1 computation.
    n_injection_outages : int, default=0
        How many injection outages are part of the N-1 computation.
    n_busbar_outages : int, default=0
        How many busbar outages are part of the N-1 computation.
    n_nminus1_cases : int, default=0
        How many N-1 cases are there in total.
    n_controllable_psts : int, default=0
        How many controllable phase shifting transformers are in the grid.
    n_monitored_branches : int, default=0
        How many branches are monitored.
    n_timesteps : int, default=0
        How many timesteps are optimized at the same time.
    n_relevant_subs : int, default=0
        How many relevant substations are in the grid.
    n_disc_branches : int, default=0
        How many disconnectable branches are in the definition.
    overload_energy_n0 : float, default=0.0
        What is the N-0 overload energy of the unsplit configuration.
    overload_energy_n1 : float, default=0.0
        What is the N-1 overload energy of the unsplit configuration.
    n_actions : int, default=0
        How many actions have been precomputed in the action set. This is the size of the branch action set,
        note that combinations of actions within that set are possible (product set wise) if multiple substations are split.
    max_station_branch_degree : int, default=0
        The maximum number of branches connected to any station in the grid.
    max_station_injection_degree : int, default=0
        The maximum number of injections connected to any station in the grid.
    mean_station_branch_degree : float, default=0.0
        The average number of branches connected to any station in the grid.
    mean_station_injection_degree : float, default=0.0
        The average number of injections connected to any station in the grid.
    reassignable_branch_assets : int, default=0
        The total number of reassignable branch assets in the grid, i.e. how many branches are connected to any of the
        stations.
    reassignable_injection_assets : int, default=0
        The total number of reassignable injection assets in the grid, i.e. how many injections are connected to any of the
        stations.
    max_reassignment_distance : int, default=0
        The maximum reassignment distance associated with any action.

    Returns
    -------
    StaticInformationStats
        An instance containing the static information statistics.
    """
    return StaticInformationStats(
        time=time,
        fp_dtype=fp_dtype,
        has_double_limits=has_double_limits,
        n_branches=n_branches,
        n_nodes=n_nodes,
        n_branch_outages=n_branch_outages,
        n_multi_outages=n_multi_outages,
        n_injection_outages=n_injection_outages,
        n_busbar_outages=n_busbar_outages,
        n_nminus1_cases=n_nminus1_cases,
        n_controllable_psts=n_controllable_psts,
        n_monitored_branches=n_monitored_branches,
        n_timesteps=n_timesteps,
        n_relevant_subs=n_relevant_subs,
        n_disc_branches=n_disc_branches,
        overload_energy_n0=overload_energy_n0,
        overload_energy_n1=overload_energy_n1,
        n_actions=n_actions,
        max_station_branch_degree=max_station_branch_degree,
        max_station_injection_degree=max_station_injection_degree,
        mean_station_branch_degree=mean_station_branch_degree,
        mean_station_injection_degree=mean_station_injection_degree,
        reassignable_branch_assets=reassignable_branch_assets,
        reassignable_injection_assets=reassignable_injection_assets,
        max_reassignment_distance=max_reassignment_distance,
    )


def create_power_factory_import_result() -> PowerFactoryImportResult:
    """Statistics and results from an import process of PowerFactory data, TODO fill"""
    result = PowerFactoryImportResult()
    result.grid_type = "power_factory"
    return result


def create_preprocessing_success_result(
    data_folder: Path,
    initial_loadflow: StoredLoadflowReference,
    initial_metrics: Dict[MetricType, float],
    static_information_stats: StaticInformationStats,
    importer_results: Union[UcteImportResult, PowerFactoryImportResult],
    result_type: Literal["preprocessing_success"] = "preprocessing_success",
) -> PreprocessingSuccessResult:
    """
    Create a PreprocessingSuccessResult object with the provided preprocessing data.

    Parameters
    ----------
    data_folder : Path
        The path where the entry point where the timestep data folder structure starts.
        The folder structure is defined in dc_solver.interfaces.folder_structure.
        Can be on a temp dir.
    initial_loadflow : StoredLoadflowReference
        The initial AC loadflow results, i.e. the N-1 analysis without any actions applied to the grid
    initial_metrics : Dict[MetricType, float]
        Dictionary mapping metric types to their corresponding float values.
    static_information_stats : StaticInformationStats
        Statistical information about the static data.
    importer_results : Union[UcteImportResult, PowerFactoryImportResult]
        Results from the data importer, either UcteImportResult or PowerFactoryImportResult.
    result_type : Literal["preprocessing_success"], optional
        Type of the result, defaults to "preprocessing_success".

    Returns
    -------
    PreprocessingSuccessResult
        The result object containing all preprocessing success information.
    """
    res = PreprocessingSuccessResult(
        data_folder=str(data_folder),
        initial_loadflow=initial_loadflow,
        initial_metrics=initial_metrics,
        static_information_stats=static_information_stats,
        result_type=result_type,
    )
    if isinstance(importer_results, UcteImportResult):
        res.ucte_result.CopyFrom(importer_results)
    elif isinstance(importer_results, PowerFactoryImportResult):
        res.power_factory_result.CopyFrom(importer_results)

    return res


def create_preprocessing_started_result() -> PreprocessingStartedResult:
    """Create a message that is sent when the preprocessing process has started"""
    result = PreprocessingStartedResult()
    result.result_type = "preprocessing_started"
    return result


def create_error_result(error: str) -> ErrorResult:
    """Create a message that is sent if an error occurred"""
    error_ob = ErrorResult(error=error)
    error_ob.result_type = "error"
    return error_ob


def create_result(
    preprocess_id: str,
    runtime: float,
    result: Union[ErrorResult, PreprocessingSuccessResult, PreprocessingStartedResult],
    instance_id: str = "",
) -> Result:
    """Create a Result message encapsulating the preprocessing result.

    Parameters
    ----------
    preprocess_id : str
        The unique identifier for the preprocessing task.
    runtime : float
        The time taken for the preprocessing task in seconds.
    result : Union[ErrorResult, PreprocessingSuccessResult, PreprocessingStartedResult]
        The result of the preprocessing task.
    instance_id : str, optional
        The identifier of the instance where the preprocessing was performed, by default "".

    Returns
    -------
    Result
        The Result message encapsulating the preprocessing result.

    Raises
    ------
    ValueError
        If runtime is not greater than 0.
    """
    if runtime <= 0:
        raise ValueError("runtime must be greater than 0")
    res = Result(
        preprocess_id=preprocess_id,
        runtime=runtime,
        result=result,
        instance_id=instance_id,
    )

    res.timestamp = str(datetime.now())
    res.uuid = str(uuid.uuid4())
    return res
