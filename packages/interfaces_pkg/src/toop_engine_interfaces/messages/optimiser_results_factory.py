"""Factory functions for creating protobuf messages in `messages.optimizer.results`.

These factories simplify the creation of protobuf objects used to communicate
optimization results such as metrics, topologies, and lifecycle events
between optimizers and backend systems.

The corresponding protobuf schema defines messages for:

- **Metrics**: Optimization fitness and additional score metrics
- **Topology**: A single network configuration for a timestep
- **Strategy**: A collection of topologies across timesteps
- **TopologyPushResult**: The set of strategies sent at each epoch
- **OptimizationStoppedResult**: Indicates optimization termination
- **OptimizationStartedResult**: Indicates optimization start
- **Result**: A generic wrapper for any optimizer result message

Each function validates inputs to ensure protobuf compatibility and guards
against malformed or inconsistent result payloads.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional, TypeAlias, Union

import numpy as np
from toop_engine_interfaces.messages.optimiser_ac_dc_commons_factory import OptimizerType
from toop_engine_interfaces.messages.protobuf_schema.lf_service.stored_loadflow_reference_pb2 import (
    StoredLoadflowReference as StoredLoadflowResultsPb,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_results_pb2 import (
    Metrics as PbMetrics,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_results_pb2 import (
    OptimizationStartedResult as PbOptimizationStartedResult,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_results_pb2 import (
    OptimizationStoppedResult as PbOptimizationStoppedResult,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_results_pb2 import (
    Result as PbResult,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_results_pb2 import (
    Strategy as PbStrategy,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_results_pb2 import (
    Topology as PbTopology,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_results_pb2 import (
    TopologyPushResult as PbTopologyPushResult,
)
from toop_engine_interfaces.messages.protobuf_schema.preprocess.preprocess_results_pb2 import (
    StaticInformationStats as PbStaticInformationStats,
)

ResultUnion: TypeAlias = Union[
    PbTopologyPushResult,
    PbOptimizationStoppedResult,
    PbOptimizationStartedResult,
]


def create_metrics(
    fitness: float,
    extra_scores: Optional[Dict[str, float]],
    worst_k_contingency_cases: Optional[List[str]] = None,
) -> PbMetrics:
    """
    Create a Metrics message.

    Parameters
    ----------
    fitness : float
        The current best fitness value (optimized quantity).
    extra_scores : dict of str to float, optional
        Additional metric scores (e.g., max_flow_n_0, losses, etc.).
    worst_k_contingency_cases : list of str, optional
        IDs of the k worst contingencies for contingency analysis.

    Returns
    -------
    PbMetrics
        A protobuf `Metrics` instance.

    Raises
    ------
    ValueError
        If fitness is NaN or non-finite.
    """
    metrics = PbMetrics(
        fitness=fitness,
        extra_scores=extra_scores,
    )
    if worst_k_contingency_cases is not None:
        metrics.worst_k_contingency_cases.extend(worst_k_contingency_cases)
    return metrics


def create_topology(
    actions: List[int],
    disconnections: List[int],
    pst_setpoints: List[int],
    metrics: PbMetrics,
    loadflow_results: Optional[StoredLoadflowResultsPb] = None,
) -> PbTopology:
    """
    Create a Topology message.

    Parameters
    ----------
    actions : list of int
        Indices of applied branch/injection reconfiguration actions.
        Automatically sorted and deduplicated.
    disconnections : list of int
        Indices of disconnected branches. Automatically sorted and deduplicated.
    pst_setpoints : list of int
        PST tap setpoints for controllable transformers.
    metrics : PbMetrics
        Metrics of this topology at the current timestep.
    loadflow_results : StoredLoadflowResultsPb
        Reference to loadflow results stored on disk.

    Returns
    -------
    PbTopology
        A protobuf `Topology` instance.
    """
    assert np.all(np.array(actions) >= 0), "Actions must be non-negative integers"
    assert np.all(np.array(disconnections) >= 0), "Disconnections must be non-negative integers"
    actions_sorted = sorted(set(actions or []))
    disconnections_sorted = sorted(set(disconnections or []))
    pst_values = pst_setpoints or []

    topo = PbTopology(
        actions=actions_sorted,
        disconnections=disconnections_sorted,
        pst_setpoints=pst_values,
        metrics=metrics,
    )
    if loadflow_results is not None:
        topo.loadflow_results.CopyFrom(loadflow_results)
    return topo


def create_strategy(timesteps: List[PbTopology]) -> PbStrategy:
    """
    Create a Strategy message.

    Parameters
    ----------
    timesteps : list of PbTopology
        List of topologies representing each timestep.

    Returns
    -------
    PbStrategy
        A protobuf `Strategy` instance.

    Raises
    ------
    ValueError
        If the list is empty or contains non-Topology elements.
    """
    if not timesteps:
        raise ValueError("Strategy must contain at least one topology.")
    if not all(isinstance(t, PbTopology) for t in timesteps):
        raise ValueError("All timesteps must be instances of PbTopology.")

    return PbStrategy(timesteps=timesteps)


def create_topology_push_result(
    strategies: List[PbStrategy],
    message_type: Literal["topology_push"] = "topology_push",
    epoch: Optional[int] = None,
) -> PbTopologyPushResult:
    """
    Create a TopologyPushResult message.

    Parameters
    ----------
    strategies : list of PbStrategy
        Strategies to push to the master.
    message_type : Literal["topology_push"]
        The result type identifier (do not change).
    epoch : Optional[int]
        Optimization epoch, for backend visualization.

    Returns
    -------
    PbTopologyPushResult
        A protobuf `TopologyPushResult` instance.

    Raises
    ------
    ValueError
        If inputs are invalid or empty.
    """
    if not message_type:
        raise ValueError("message_type must be a non-empty string.")
    if not strategies:
        raise ValueError("At least one strategy must be provided.")
    if epoch < 0:
        raise ValueError("epoch must be non-negative.")

    return PbTopologyPushResult(
        message_type=message_type,
        strategies=strategies,
        epoch=epoch,
    )


def create_optimization_stopped_result(
    message_type: Literal["stopped"] = "stopped",
    reason: Literal["error", "stopped", "converged", "ac-not-converged", "unknown"] = "unknown",
    message: Optional[str] = "",
    epoch: Optional[int] = None,
) -> PbOptimizationStoppedResult:
    """
    Create an OptimizationStoppedResult message.

    Parameters
    ----------
    message_type : str
        The result type identifier (do not change).
    reason : str
        Reason for stopping the optimization.
        Valid values: "Error", "Stopped", "Converged", "ac-not-converged".
    message : str, optional
        Additional stop message, e.g. an error explanation.
    epoch : int
        Epoch when the optimization stopped.

    Returns
    -------
    PbOptimizationStoppedResult
        A protobuf `OptimizationStoppedResult` instance.

    Raises
    ------
    ValueError
        If reason or message_type are empty or invalid.
    """
    res = PbOptimizationStoppedResult(
        message_type=message_type,
        reason=reason,
        message=message or "",
    )
    if epoch is not None:
        res.epoch = epoch
    return res


def create_optimization_started_result(
    message_type: str,
    initial_topology: PbStrategy,
    initial_stats: Optional[List[PbStaticInformationStats]] = None,
) -> PbOptimizationStartedResult:
    """
    Create an OptimizationStartedResult message.

    Parameters
    ----------
    message_type : str
        The result type identifier (do not change).
    initial_topology : PbStrategy
        The initial topology including starting metrics.
    initial_stats : list of Struct, optional
        Initial statistics for each timestep.

    Returns
    -------
    PbOptimizationStartedResult
        A protobuf `OptimizationStartedResult` instance.
    """
    if not message_type:
        raise ValueError("message_type must be non-empty.")
    if not isinstance(initial_topology, PbStrategy):
        raise ValueError("initial_topology must be a PbStrategy instance.")

    return PbOptimizationStartedResult(
        message_type=message_type,
        initial_topology=initial_topology,
        initial_stats=initial_stats,
    )


def create_result(
    result: ResultUnion,
    optimization_id: str,
    optimizer_type: OptimizerType,
    instance_id: str = "",
    uuid: str = str(uuid.uuid4()),
    timestamp: str = str(datetime.now()),
) -> PbResult:
    """
    Create a Result message wrapping any optimizer result.

    Parameters
    ----------
    result : one of (PbTopologyPushResult, PbOptimizationStoppedResult, PbOptimizationStartedResult)
        The actual result message to wrap.
    optimization_id : str
        The optimization ID associated with this result.
    optimizer_type : OptimizerType
        The optimizer type (e.g., "DC", "AC").
    instance_id : str
        The unique optimizer instance ID.
    uuid : str = str(uuid.uuid4())
        Unique identifier for this result message.
    timestamp : str = str(datetime.now())
        When the result was sent (ISO 8601 timestamp).

    Returns
    -------
    PbResult
        A protobuf `Result` instance.

    Raises
    ------
    ValueError
        If any mandatory identifier is missing.
    """
    # Determine the correct oneof field
    if isinstance(result, PbTopologyPushResult):
        kwargs = {"topology_push": result}
    elif isinstance(result, PbOptimizationStoppedResult):
        kwargs = {"stopped": result}
    elif isinstance(result, PbOptimizationStartedResult):
        kwargs = {"optimization_started": result}
    else:
        raise ValueError("Unsupported result message type.")

    return PbResult(
        **kwargs,
        optimization_id=optimization_id,
        optimizer_type=optimizer_type.value,
        instance_id=instance_id,
        uuid=uuid,
        timestamp=timestamp,
    )
