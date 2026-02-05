# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""A utility class for loading config values from a file.

This is nicer than having a lot of command line parameters for an invocation.
Eventually, these values need to be copied into the static information to be effective in the solver.
"""

import json
from dataclasses import asdict, dataclass

from beartype.typing import Optional
from jax_dataclasses import replace as jax_replace
from toop_engine_dc_solver.jax.types import MetricType, StaticInformation


@dataclass
class Parameters:
    """Holds parameter about the hardware the solver runs on.

    Stores configuration items that are not directly dependent on the grid.
    """

    number_most_affected: int
    """The number of worst contingency results to track for a topology."""

    number_most_affected_n_0: int
    """The number of worst N-0 flows to track for a topology"""

    number_max_out_in_most_affected: Optional[int]
    """The number of worst contingency results to track for a single outage. If not given, up
    to number_most_affected contingency results can be stored for a single outage."""

    batch_size_bsdf: int
    """The batch size for the BSDF computation, static flow computation and LODF matrix computation.
    A high bsdf batch size will incur a memory penalty as a high number of ptdf and lodf matrices
    need to be stored"""

    batch_size_injection: int
    """The batch size for the injection computation and contingency analysis"""

    buffer_size_injection: Optional[int]
    """The buffer size for how many injection batches can be stored. The theoretical upper bound is
    batch_size_bsdf * max_injections_per_topology / batch_size_injections, however likely a lower
    value is sufficient. If None, the buffer size is determined greedily"""

    limit_n_subs: Optional[int]
    """Limit the number of affected substations. If None, the number of affected substations is
    determined greedily"""

    single_precision: bool
    """Whether to use single precision (true) or double precision (false) for the computations"""

    aggregation_metric: MetricType = "max_flow_n_1"
    """The metric to use for selecting the best injection in bruteforce mode. This will be passed to
    aggregate_n_1_matrix"""

    distributed: bool = False
    """Whether to use distributed computation. This will use the jax pmap function to parallelize
    the computation over multiple devices."""


def update_static_information(static_information: StaticInformation, config: Parameters) -> StaticInformation:
    """Update the static information with a new config.

    If buffer_size_injection or limit_n_subs is None (greedy), the value from the static information
    is used as these have to be determined based on the computations to do.

    Parameters
    ----------
    static_information
        The previous static information dataclass
    config
        The new config

    Returns
    -------
        The updated static information dataclass
    """
    number_most_affected = min(
        config.number_most_affected,
        static_information.n_branches_monitored * static_information.n_nminus1_cases,
    )
    number_most_affected_n_0 = min(
        config.number_most_affected_n_0,
        static_information.n_branches_monitored,
    )
    number_max_out_in_most_affected = (
        min(
            config.number_max_out_in_most_affected,
            static_information.n_branches_monitored,
        )
        if config.number_max_out_in_most_affected is not None
        else None
    )

    return jax_replace(
        static_information,
        solver_config=jax_replace(
            static_information.solver_config,
            number_most_affected=number_most_affected,
            number_most_affected_n_0=number_most_affected_n_0,
            number_max_out_in_most_affected=number_max_out_in_most_affected,
            batch_size_bsdf=config.batch_size_bsdf,
            batch_size_injection=config.batch_size_injection,
            buffer_size_injection=(
                config.buffer_size_injection
                if config.buffer_size_injection is not None
                else static_information.solver_config.buffer_size_injection
            ),
            limit_n_subs=(
                config.limit_n_subs if config.limit_n_subs is not None else static_information.solver_config.limit_n_subs
            ),
            aggregation_metric=config.aggregation_metric,
            distributed=config.distributed,
        ),
    )


def default_config() -> Parameters:
    """Get default values with which you can run the solver if performance is not critical

    Returns
    -------
        Parameters: A default solver config
    """
    return Parameters(
        number_most_affected=30,
        number_most_affected_n_0=30,
        number_max_out_in_most_affected=5,
        batch_size_bsdf=8,
        batch_size_injection=256,
        buffer_size_injection=None,
        limit_n_subs=None,
        single_precision=False,
        aggregation_metric="max_flow_n_1",
        distributed=False,
    )


def read_config_from_file(filename: str) -> Parameters:
    """Read the config for the solver from a json file.

    Uses default values for everything that's not provided

    Parameters
    ----------
    filename
        The json file to read from

    Returns
    -------
    parameters : Parameters
        The populated solver config dataclass.
    """
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    parameters = Parameters(
        number_most_affected=int(data.get("number_most_affected", 30)),
        number_most_affected_n_0=int(data.get("number_most_affected_n_0", 30)),
        number_max_out_in_most_affected=int(data.get("number_max_out_in_most_affected", 5)),
        batch_size_bsdf=int(data.get("batch_size_bsdf", 8)),
        batch_size_injection=int(data.get("batch_size_injection", 256)),
        buffer_size_injection=(
            int(data["buffer_size_injection"])
            if ("buffer_size_injection" in data and data["buffer_size_injection"] is not None)
            else None
        ),
        limit_n_subs=(int(data["limit_n_subs"]) if ("limit_n_subs" in data and data["limit_n_subs"] is not None) else None),
        single_precision=bool(data.get("single_precision", False)),
        aggregation_metric=str(data.get("aggregation_metric", "max_flow_n_1")),
        distributed=bool(data.get("distributed", False)),
    )
    return parameters


def save_config(config: Parameters, filename: str) -> None:
    """Save the config to a json file

    Parameters
    ----------
    config
        The config to save
    filename
        The json file to save the config to
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(asdict(config), file)
