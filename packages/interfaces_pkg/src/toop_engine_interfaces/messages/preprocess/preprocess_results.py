"""classes defined in preprocess_results.py"""

import uuid
from datetime import datetime
from pathlib import Path

from beartype.typing import Literal, Optional, Union
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference
from toop_engine_interfaces.types import MetricType


class UcteImportResult(BaseModel):
    """Statistics and results from an import process of UCTE data"""

    data_folder: Path
    """The path where the entry point where the timestep data folder structure starts.
    The folder structure is defined in dc_solver.interfaces.folder_structure.
    Can be on a temp dir"""

    n_relevant_subs: NonNegativeInt = 0
    """The number of relevant substations"""

    n_low_impedance_lines: NonNegativeInt = 0
    """The number of low impedance lines that have been converted to a switch"""

    n_branch_across_switch: NonNegativeInt = 0
    """The number of branches across a switch that have been removed"""

    n_line_for_nminus1: NonNegativeInt = 0
    """The number of lines in the N-1 definition"""

    n_line_for_reward: NonNegativeInt = 0
    """The number of lines that are observed"""

    n_line_disconnectable: NonNegativeInt = 0
    """The number of lines that are disconnectable"""

    n_trafo_for_nminus1: NonNegativeInt = 0
    """The number of trafos in the N-1 definition"""

    n_trafo_for_reward: NonNegativeInt = 0
    """The number of trafos that are observed"""

    n_trafo_disconnectable: NonNegativeInt = 0
    """The number of trafos in the N-1 definition"""

    n_tie_line_for_reward: NonNegativeInt = 0
    """The number of tie lines that are observed"""

    n_tie_line_for_nminus1: NonNegativeInt = 0
    """The number of tie lines in the N-1 definition"""

    n_tie_line_disconnectable: NonNegativeInt = 0
    """The number of tie lines that are disconnectable"""

    n_dangling_line_for_nminus1: NonNegativeInt = 0
    """The number of dangling lines in the N-1 definition"""

    n_generator_for_nminus1: NonNegativeInt = 0
    """The number of generators in the N-1 definition"""

    n_load_for_nminus1: NonNegativeInt = 0
    """The number of loads in the N-1 definition"""

    n_switch_for_nminus1: NonNegativeInt = 0
    """The number of switches in the N-1 definition"""

    n_switch_for_reward: NonNegativeInt = 0
    """The number of switches that are observed"""

    n_white_list: Optional[NonNegativeInt] = 0
    """The number of elements in the whitelist in total"""

    n_white_list_applied: Optional[NonNegativeInt] = 0
    """The number of elements in the whitelist that were successfully matched and applied"""

    n_black_list: Optional[NonNegativeInt] = 0
    """The number of elements in the blacklist in total"""

    n_black_list_applied: Optional[NonNegativeInt] = 0
    """The number of elements in the blacklist that were successfully matched and applied"""

    grid_type: Literal["ucte"] = "ucte"
    """The discriminator for the ImportResult Union"""


class StaticInformationStats(BaseModel):
    """Stats about the static information class"""

    time: Optional[str] = None
    """The timestep that was optimized, if given"""

    fp_dtype: str = ""
    """A string representation of the floating point type used in the static informations, e.g. 'float32' or 'float64'."""

    has_double_limits: bool = False
    """Whether the static information has max_mw_flow_limited set or not"""

    n_branches: NonNegativeInt = 0
    """The number of branches in the PTDF matrix"""

    n_nodes: NonNegativeInt = 0
    """The number of nodes in the PTDF matrix"""

    n_branch_outages: NonNegativeInt = 0
    """How many branch outages are part of the N-1 computation"""

    n_multi_outages: NonNegativeInt = 0
    """How many multi-outages are part of the N-1 computation"""

    n_injection_outages: NonNegativeInt = 0
    """How many injection outages are part of the N-1 computation"""

    n_busbar_outages: NonNegativeInt = 0
    """How many busbar outages are part of the N-1 computation"""

    n_nminus1_cases: NonNegativeInt = 0
    """How many N-1 cases are there in total"""

    n_controllable_psts: NonNegativeInt = 0
    """How many controllable phase shifting transformers are in the grid"""

    n_monitored_branches: NonNegativeInt = 0
    """How many branches are monitored"""

    n_timesteps: NonNegativeInt = 0
    """How many timesteps are optimized at the same time"""

    n_relevant_subs: NonNegativeInt = 0
    """How many relevant substations are in the grid"""

    n_disc_branches: NonNegativeInt = 0
    """How many disconnectable branches are in the definition"""

    overload_energy_n0: float = 0.0
    """What is the N-0 overload energy of the unsplit configuration"""

    overload_energy_n1: float = 0.0
    """What is the N-1 overload energy of the unsplit configuration"""

    n_actions: NonNegativeInt = 0
    """How many actions have been precomputed in the action set. This
    is the size of the branch action set, note that combinations of actions within that set are
    possible (product set wise) if multiple substations are split"""

    max_station_branch_degree: NonNegativeInt = 0
    """The maximum number of branches connected to any station in the grid"""

    max_station_injection_degree: NonNegativeInt = 0
    """The maximum number of injections connected to any station in the grid"""

    mean_station_branch_degree: NonNegativeFloat = 0.0
    """The average number of branches connected to any station in the grid"""

    mean_station_injection_degree: NonNegativeFloat = 0.0
    """The average number of injections connected to any station in the grid"""

    reassignable_branch_assets: NonNegativeInt = 0
    """The total number of reassignable branch assets in the grid, i.e. how many branches are
    connected to any of the stations"""

    reassignable_injection_assets: NonNegativeInt = 0
    """The total number of reassignable injection assets in the grid, i.e. how many injections are
    connected to any of the stations"""

    max_reassignment_distance: NonNegativeInt = 0
    """The maximum reassignment distance associated with any action"""


class PowerFactoryImportResult(BaseModel):
    """Statistics and results from an import process of PowerFactory data, TODO fill"""

    grid_type: Literal["power_factory"] = "power_factory"
    """The discriminator for the ImportResult Union"""


class PreprocessingSuccessResult(BaseModel):
    """Results of a preprocessing run, mainly including the static_information and network_data files."""

    data_folder: Path
    """The path where the entry point where the timestep data folder structure starts.
    The folder structure is defined in dc_solver.interfaces.folder_structure.
    Can be on a temp dir"""

    initial_loadflow: StoredLoadflowReference
    """The initial AC loadflow results, i.e. the N-1 analysis without any actions applied to the grid."""

    initial_metrics: dict[MetricType, float]
    """The initial metrics computed for the loadflow results"""

    static_information_stats: StaticInformationStats
    """Statistics about the static information file that was produced"""

    importer_results: Union[UcteImportResult, PowerFactoryImportResult] = Field(discriminator="grid_type")
    """The results of the importer process"""

    result_type: Literal["preprocessing_success"] = "preprocessing_success"
    """The discriminator for the Result Union"""


class PreprocessingStartedResult(BaseModel):
    """A message that is sent when the preprocessing process has started"""

    result_type: Literal["preprocessing_started"] = "preprocessing_started"
    """The discriminator for the Result Union"""


class ErrorResult(BaseModel):
    """A message that is sent if an error occurred"""

    error: str
    """The error message"""

    result_type: Literal["error"] = "error"
    """The discriminator for the Result Union"""


class Result(BaseModel):
    """A generic class for result, holding either a successful or an unsuccessful result"""

    preprocess_id: str
    """The preprocess_id that was sent in the preprocess_command, used to identify the result"""

    instance_id: str = ""
    """The instance id of the importer worker that created this result"""

    runtime: NonNegativeFloat
    """The runtime in seconds that the preprocessing took until the result"""

    result: Union[ErrorResult, PreprocessingSuccessResult, PreprocessingStartedResult] = Field(discriminator="result_type")
    """The actual result data in a discriminated union"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for this result message, used to avoid duplicates during processing"""

    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
    """When the result was sent"""
