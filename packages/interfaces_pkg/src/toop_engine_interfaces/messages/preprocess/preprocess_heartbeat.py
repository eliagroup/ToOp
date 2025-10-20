"""Contains the message classes for a preprocessing worker's heartbeat messages"""

import uuid
from datetime import datetime

import logbook
from beartype.typing import Literal, Optional, TypeAlias
from pydantic import BaseModel, Field

logger = logbook.Logger(__name__)

ConvertToJaxStage: TypeAlias = Literal[
    "convert_to_jax_started",
    "convert_tot_stat",
    "convert_relevant_inj",
    "convert_masks",
    "switching_distance_info",
    "pad_out_branch_actions",
    "convert_rel_bb_outage_data",
    "create_static_information",
    "filter_branch_actions",
    "unsplit_n2_analysis",
    "bb_outage_baseline_analysis",
    "convert_to_jax_done",
]

NumpyPreprocessStage: TypeAlias = Literal[
    "preprocess_started",
    "extract_network_data_from_interface",
    "filter_relevant_nodes",
    "assert_network_data",
    "compute_ptdf_if_not_given",
    "compute_psdf_if_not_given",
    "add_nodal_injections_to_network_data",
    "combine_phaseshift_and_injection",
    "compute_bridging_branches",
    "exclude_bridges_from_outage_masks",
    "reduce_branch_dimension",
    "reduce_node_dimension",
    "filter_disconnectable_branches_nminus2",
    "compute_branch_topology_info",
    "compute_electrical_actions",
    "enumerate_station_realizations",
    "remove_relevant_subs_without_actions",
    "simplify_asset_topology",
    "convert_multi_outages",
    "filter_inactive_injections",
    "compute_injection_topology_info",
    "process_injection_outages",
    "add_missing_asset_topo_info",
    "add_bus_b_columns_to_ptdf",
    "enumerate_injection_actions",
    "preprocess_bb_outage",
    "preprocess_done",
]

LoadGridStage: TypeAlias = Literal[
    "load_grid_into_loadflow_solver_backend",
    "compute_base_loadflows",
    "save_artifacts",
]

InitialLoadflowStage: TypeAlias = Literal["prepare_contingency_analysis", "run_contingency_analysis"]

ImporterStage: TypeAlias = Literal[
    "start",
    "load_ucte",
    "get_topology_model",
    "modify_low_impedance_lines",
    "modify_branches_over_switches",
    "apply_cb_list",
    "cross_border_current",
    "get_masks",
    "end",
]

PreprocessStage: TypeAlias = Literal[
    ImporterStage, NumpyPreprocessStage, ConvertToJaxStage, LoadGridStage, InitialLoadflowStage
]


class PreprocessStatusInfo(BaseModel):
    """A status info to inform about an ongoing preprocess action."""

    preprocess_id: str
    """The id of the preprocess job."""

    runtime: float
    """The amount of time since the start of the optimization."""

    stage: PreprocessStage
    """The stage in which the preprocessing job currently is."""

    message: Optional[str]
    """An optional message"""


class PreprocessHeartbeat(BaseModel):
    """A message class for heartbeats from the preprocessing worker.

    When idle, this just sends a hello, and when preprocessing it also conveys a status
    update at which stage the preprocessing is so it can be tracked in the frontend.
    """

    idle: bool
    """Whether the worker is idle"""

    status_info: Optional[PreprocessStatusInfo] = None
    """If not idle, a status update"""

    instance_id: str = ""
    """The ID of the worker instance that sent this heartbeat."""

    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
    """When the heartbeat was sent"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for this heartbeat message, used to avoid duplicates during processing"""


def empty_status_update_fn(stage: PreprocessStage, message: Optional[str]) -> None:
    """Log an empty status update to logging.

    Use this function when no status_update_fn is provided.
    """
    if message is None:
        logger.info(f"Preprocessing stage {stage}")
    else:
        logger.info(f"Preprocessing stage {stage}, {message}")
