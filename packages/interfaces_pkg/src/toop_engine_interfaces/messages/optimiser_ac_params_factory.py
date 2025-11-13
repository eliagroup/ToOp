"""Factory methods for creating AC optimizer protobuf messages with validation.

These factories build `ACGAParameters` and `ACOptimizerParameters` messages
from the compiled protobuf schema, and perform light input validation similar
to what Pydantic would enforce.

Example
-------
>>> from factories.ac_parameters_factory import create_ac_ga_parameters
>>> params = create_ac_ga_parameters(runtime_seconds=3600, seed=42)
>>> params.runtime_seconds
3600
"""

from typing import List, Optional

from toop_engine_interfaces.messages.protobuf_schema.lf_service import stored_loadflow_reference_pb2
from toop_engine_interfaces.messages.protobuf_schema.optimizer import ac_dc_commons_pb2, ac_optimizer_params_pb2


def create_ac_ga_parameters(  # noqa: PLR0913, C901, PLR0912
    runtime_seconds: int = 30,
    pull_prob: float = 0.9,
    me_descriptors: Optional[List[ac_dc_commons_pb2.DescriptorDef]] = None,
    reconnect_prob: float = 0.05,
    close_coupler_prob: float = 0.05,
    n_worst_contingencies: int = 20,
    seed: int = 42,
    timestep_processes: int = 1,
    runner_processes: int = 1,
    runner_batchsize: Optional[int] = 0,
    filter_strategy: Optional[ac_dc_commons_pb2.FilterStrategy] = None,
    enable_ac_rejection: bool = True,
    reject_convergence_threshold: float = 1.0,
    reject_overload_threshold: float = 0.95,
    reject_critical_branch_threshold: float = 1.1,
    early_stop_validation: bool = True,
    early_stopping_non_convergence_percentage_threshold: float = 0.1,
) -> ac_optimizer_params_pb2.ACGAParameters:
    """
    Create and validate an `ACGAParameters` protobuf message.

    Parameters
    ----------
    runtime_seconds : int
        The maximum runtime of the AC optimization in seconds.
    pull_prob : float, optional
        The probability of pulling a strategy from the DC repertoire. Must be between 0 and 1.
    me_descriptors : list of DescriptorDef, optional
        The descriptors for the aggregated map elites repertoire.
    reconnect_prob : float, optional
        The probability of reconnecting a disconnected branch in a strategy. Must be between 0 and 1.
    close_coupler_prob : float, optional
        The probability of closing an opened coupler in a strategy. Must be between 0 and 1.
    n_worst_contingencies : int, optional
        Number of worst contingencies to consider.
    seed : int, optional
        The seed for the random number generator.
    timestep_processes : int, optional
        Number of processes to spawn for computing timesteps in parallel.
    runner_processes : int, optional
        Number of processes for computing N-1 cases per timestep in parallel.
    runner_batchsize : int, optional
        Batch size for N-1 cases.
    filter_strategy : FilterStrategy, optional
        The filter strategy used to discard or prioritize strategies.
    enable_ac_rejection : bool, optional
        Whether to enable AC rejection.
    reject_convergence_threshold : float, optional
        Max acceptable non-convergence ratio for split vs unsplit cases.
    reject_overload_threshold : float, optional
        Minimum improvement required in overload energy.
    reject_critical_branch_threshold : float, optional
        Maximum tolerated increase in critical branches.
    early_stop_validation : bool, optional
        Whether to enable early stopping.
    early_stopping_non_convergence_percentage_threshold : float, optional
        Non-convergence rate threshold for early stopping.

    Returns
    -------
    ACGAParameters
        A validated protobuf message.

    Raises
    ------
    ValueError
        If numeric parameters are invalid or probabilities fall outside [0, 1].
    """
    # --- Basic validations (similar to what Pydantic would enforce) ---
    if runtime_seconds <= 0:
        raise ValueError("`runtime_seconds` must be positive.")

    for name, prob in {
        "pull_prob": pull_prob,
        "reconnect_prob": reconnect_prob,
        "close_coupler_prob": close_coupler_prob,
    }.items():
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"`{name}` must be between 0 and 1.")

    if n_worst_contingencies < 0:
        raise ValueError("`n_worst_contingencies` cannot be negative.")
    if timestep_processes <= 0:
        raise ValueError("`timestep_processes` must be positive.")
    if runner_processes <= 0:
        raise ValueError("`runner_processes` must be positive.")
    if runner_batchsize < 0:
        raise ValueError("`runner_batchsize` must be positive.")
    if early_stopping_non_convergence_percentage_threshold < 0.0:
        raise ValueError("`early_stopping_non_convergence_percentage_threshold` cannot be negative.")
    if pull_prob + reconnect_prob + close_coupler_prob != 1.0:
        raise ValueError("The probabilities must sum to one")

    # --- Default me_descriptors if not provided ---
    if me_descriptors is None:
        me_descriptors = [
            ac_dc_commons_pb2.DescriptorDef(metric="split_subs", num_cells=2, range=(0, 5)),
            ac_dc_commons_pb2.DescriptorDef(metric="switching_distance", num_cells=5, range=(0, 50)),
            ac_dc_commons_pb2.DescriptorDef(metric="disconnected_branches", num_cells=2),
        ]

    # --- Construct message ---
    msg = ac_optimizer_params_pb2.ACGAParameters(
        runtime_seconds=runtime_seconds,
        pull_prob=pull_prob,
        reconnect_prob=reconnect_prob,
        close_coupler_prob=close_coupler_prob,
        n_worst_contingencies=n_worst_contingencies,
        seed=seed,
        timestep_processes=timestep_processes,
        runner_processes=runner_processes,
        enable_ac_rejection=enable_ac_rejection,
        reject_convergence_threshold=reject_convergence_threshold,
        reject_overload_threshold=reject_overload_threshold,
        reject_critical_branch_threshold=reject_critical_branch_threshold,
        early_stop_validation=early_stop_validation,
        early_stopping_non_convergence_percentage_threshold=early_stopping_non_convergence_percentage_threshold,
    )

    if runner_batchsize is not None:
        msg.runner_batchsize = runner_batchsize

    if me_descriptors:
        msg.me_descriptors.extend(me_descriptors)
    if filter_strategy:
        msg.filter_strategy.CopyFrom(filter_strategy)

    return msg


def create_ac_optimizer_parameters(
    initial_loadflow: Optional[stored_loadflow_reference_pb2.StoredLoadflowReference] = None,
    ga_config: Optional[ac_optimizer_params_pb2.ACGAParameters] = None,
) -> ac_optimizer_params_pb2.ACOptimizerParameters:
    """
    Create and validate an `ACOptimizerParameters` protobuf message.

    Parameters
    ----------
    initial_loadflow : StoredLoadflowReference, optional
        An initial AC loadflow reference used for double-limit computation.
        If provided, it must contain valid file references or identifiers.
    ga_config : ACGAParameters, optional
        Configuration of the AC genetic algorithm.

    Returns
    -------
    ACOptimizerParameters
        A validated protobuf message.

    Raises
    ------
    ValueError
        If `ga_config` is missing or invalid.
    """
    if ga_config is None:
        ga_config = create_ac_ga_parameters()

    msg = ac_optimizer_params_pb2.ACOptimizerParameters(ga_config=ga_config)

    if initial_loadflow:
        msg.initial_loadflow.CopyFrom(initial_loadflow)

    return msg
