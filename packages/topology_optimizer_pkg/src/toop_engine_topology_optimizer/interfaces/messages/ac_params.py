# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The parameters for the AC optimizer.

On AC, some subtelties are different to the DC optimization such as that the optimization is not
batched, and the parameters are slightly different.
"""

from beartype.typing import Optional
from pydantic import BaseModel, Field, PositiveInt, confloat, model_validator
from toop_engine_interfaces.loadflow_result_filter import LoadflowResultFilter
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_topology_optimizer.interfaces.messages.commons import FilterStrategy


class ACGAParameters(BaseModel):
    """Parameters for the AC genetic algorithm"""

    runtime_seconds: PositiveInt = 180
    """The maximum runtime of the AC optimization in seconds"""

    n_worst_contingencies: PositiveInt = 20
    """How many worst contingencies to consider for the initial metrics, i.e. the top k contingencies
    that are used to compute the initial metrics. This is used to compute the top_k_overloads_n_1"""

    include_non_converging_loadflows_in_worst_k: bool = True
    """Whether non-converging contingencies should always be appended to the worst-k contingency cases.
    These will be added on top of the worst-k contingencies that are selected based on the overload energy.
    So if k=20 and there are 3 non-converging contingencies, the worst-k will contain 23 contingencies in total.
    If this is set to False, the non-converging contingencies will be ignored and the worst-k will only contain
    the contingencies with the highest overload energy."""

    seed: int = 42
    """The seed for the random number generator"""

    runner_processes: PositiveInt = 1
    """How many processes to spawn for computing the N-1 cases in each timestep in parallel. Note
    that this multiplies with contingency_processes and you might run out of memory if you set both
    too high"""

    contingency_processes: PositiveInt = 1
    """How many processes to spawn for computing the contingencies of each strategy in parallel. Note
    that this multiplies with runner_processes and you might run out of memory if you set both too high"""

    worst_k_runner_processes: PositiveInt = 1
    """How many processes to spawn per topology during the worst-k stage."""

    worst_k_contingency_processes: PositiveInt = 1
    """How many processes to spawn for computing the contingencies of each strategy in parallel during the worst-k stage."""

    remaining_loadflow_wait_seconds: confloat(ge=0.0) = 30.0
    """Maximum time to keep collecting non-rejected strategies before starting the remaining
    contingency evaluation, even if the survivor threshold has not been reached."""

    filter_strategy: Optional[FilterStrategy] = None
    """The filter strategy to use for the optimization, used to filter out strategies
    based on the discriminator, median or dominator filter."""

    enable_ac_rejection: bool = True
    """Whether to enable the AC rejection, i.e. no messages will be sent to the results topic in case of non-acceptance."""

    reject_convergence_threshold: float = 1.0
    """The rejection threshold for the convergence rate, i.e. the split case must have at most the same amount of
    non converging loadflows as the unsplit case or it will be rejected."""

    reject_overload_threshold: float = 0.95
    """The rejection threshold for the overload energy improvement, i.e. the split case must have at least 5% lower
    overload energy than the unsplit case or it will be rejected."""

    reject_critical_branch_threshold: float = 1.1
    """The rejection threshold for the critical branches increase, i.e. the split case must have less than 10% more
    critical branches than the unsplit case or it will be rejected."""

    reject_voltage_jump_threshold: float = 1.1
    """The rejection threshold for the voltage jump count increase, i.e. the split case must have less than 10% more
    critical voltage jumps than the unsplit case or it will be rejected."""

    reject_critical_va_diff_threshold: float = 1.1
    """The rejection threshold for the critical voltage-angle-difference count increase, i.e. the split case must have
    less than 10% more critical voltage-angle differences than the unsplit case or it will be rejected."""

    enable_critical_voltage_rejection: bool = False
    """Whether to use critical voltage jumps and voltage-angle-difference counts as an acceptance/rejection criterion.

    The associated metrics are still computed and reported when this flag is disabled.
    """

    critical_voltage_jump_percent: float = 5.0
    """Voltage jumps larger than this percentage are counted as critical in the AC metrics."""

    critical_va_diff_degree: float = 20.0
    """Voltage angle differences larger than this value in degrees are counted as critical in the AC metrics."""

    early_stop_validation: bool = True
    """Whether to enable early stopping during the optimization process."""

    early_stopping_non_convergence_percentage_threshold: float = 0.1
    """The threshold for the early stopping criterion, i.e. if the percentage of non-converging cases is greater than
    this value, the ac validation will be stopped early."""

    max_initial_wait_seconds: PositiveInt = 60
    """The maximum amount of seconds to wait for the initial DC results. If no results have arrived within this time, we
    assume the DC optimizer had some problem and abort the optimization run."""

    result_filter: LoadflowResultFilter = Field(default_factory=LoadflowResultFilter)
    """Policy for dropping loadflow result rows that carry no decision value, applied by the runners as results are
    produced.

    A full result set is stored for every evaluated candidate topology, so this scales with population x generations.
    The default is inert and keeps every row.

    The AC metrics are computed from the same filtered results, so a policy set here has to leave every row those metrics
    read: see :meth:`_validate_result_filter_against_metrics`.

    One metric cannot be protected by validation. ``max_flow_n_0`` and ``max_flow_n_1`` are the maximum loading over
    *all* branch rows, so any threshold above zero can in principle drop every row they read. That only happens when no
    branch anywhere reaches the threshold, and the maximum is then reported as ``0.0`` rather than its true, small value
    - the metric degrades to "below the filter threshold" instead of an exact number. No rejection criterion reads it, so
    this is a reporting artefact rather than a behaviour change, but it is worth knowing before reading a filtered run's
    max-flow series.
    """

    @model_validator(mode="after")
    def _validate_result_filter_against_metrics(self) -> "ACGAParameters":
        """Reject filter policies that would silently starve the metrics computed from the same results.

        Returns
        -------
        ACGAParameters
            The validated parameters.

        Raises
        ------
        ValueError
            If the node filter could drop rows that ``count_voltage_jumps`` needs, if the branch filter could drop rows
            that the overload metrics need, or if either sub-filter would drop the basecase rows that the N-0 metrics
            are asserted to be computable from.
        """
        # Loading above which a branch counts towards the *overload* metrics. ``count_critical_branches`` and both
        # ``compute_overload_energy`` variants key on a loading above this, so a threshold at or below it cannot change
        # any of them. It does not protect ``max_flow_n_0``/``max_flow_n_1``, which read every branch row - see the note
        # on :attr:`result_filter`.
        overload_metric_loading_threshold = 1.0

        node_filters = self.result_filter.node_filters
        if node_filters.is_active():
            jump_threshold = node_filters.vm_basecase_deviation_above
            if jump_threshold is None or jump_threshold > self.critical_voltage_jump_percent:
                raise ValueError(
                    "result_filter.node_filters.vm_basecase_deviation_above must be set to at most "
                    f"critical_voltage_jump_percent ({self.critical_voltage_jump_percent}), but is {jump_threshold}. "
                    "voltage_jump_count_n_1 is computed from the filtered results, so a bus that jumps far enough to be "
                    "critical while staying inside its voltage band would be dropped before it could be counted."
                )

        loading_threshold = self.result_filter.branch_filters.loading_above
        if loading_threshold is not None and loading_threshold > overload_metric_loading_threshold:
            raise ValueError(
                f"result_filter.branch_filters.loading_above must be at most {overload_metric_loading_threshold}, but is "
                f"{loading_threshold}. The overload metrics are computed from the filtered results and count branches "
                f"above {overload_metric_loading_threshold} of their rating, which a higher threshold would drop first."
            )

        return self


class ACOptimizerParameters(BaseModel):
    """The set of parameters that are used in the AC optimizer only"""

    initial_loadflow: Optional[StoredLoadflowReference] = None
    """If an initial AC loadflow was computed before the start of the optimization run, this can
    be passed and will be used e.g. to compute double limits. It will be sent back through the
    initial topology push."""

    ga_config: ACGAParameters = ACGAParameters()
    """The genetic algorithm configuration"""
