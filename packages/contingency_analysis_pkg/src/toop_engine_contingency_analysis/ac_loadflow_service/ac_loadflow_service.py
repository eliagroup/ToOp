# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Get the results of the AC Contingency Analysis for the given network"""

import pypowsybl
from beartype.typing import Optional
from pypowsybl.network import Network as PowsyblNetwork
from toop_engine_contingency_analysis.pypowsybl import (
    run_contingency_analysis_powsybl,
)
from toop_engine_interfaces.loadflow_result_filter import LoadflowResultFilter
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


def get_security_analysis_results(
    net: PowsyblNetwork,
    n_minus_1_definition: Nminus1Definition,
    timestep: int = 0,
    job_id: str = "",
    n_processes: int = 1,
    batch_size: Optional[int] = None,
    security_analysis_params: pypowsybl.loadflow.Parameters | dict | None = None,
    result_filter: Optional[LoadflowResultFilter] = None,
) -> LoadflowResultsPolars:
    """Get the results of the security analysis for the given network

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to run the contingency analysis on
    n_minus_1_definition: Nminus1Definition
        The N-1 definition to use for the contingency analysis. Contains outages and monitored elements
    timestep: int, default=0
        The timestep of the results. Used to identify the results in the database
    job_id: str, default=""
        The job id of the current job
    n_processes: int, default=1
        The number of processes to use for the contingency analysis. If 1, the analysis is run sequentially.
        If > 1, the analysis is run in parallel
        Paralelization is done by splitting the contingencies into chunks and running each chunk in a separate process
    batch_size: int, optional
        The maximum number of outage contingencies per batch. Powsybl executes batches sequentially; pandapower uses
        batches for parallelization.
        If None, all are computed at the same time
    security_analysis_params: pypowsybl.loadflow.Parameters or dict, optional
        Loadflow parameters to use for the computation.
        dict for pandapower, pypowsybl.loadflow.Parameters for powsybl. If None, default parameters are used.
    result_filter: LoadflowResultFilter, optional
        Policy for dropping result rows that carry no decision value, applied inside the contingency lib as the results
        are produced. If None, every row is kept.

    Returns
    -------
    LoadflowResultsPolars
        The results of the Contingency analysis
    """
    lf_results = run_contingency_analysis_powsybl(
        net,
        n_minus_1_definition,
        job_id,
        timestep,
        n_processes=n_processes,
        method="ac",
        polars=True,
        lf_params=security_analysis_params if isinstance(security_analysis_params, pypowsybl.loadflow.Parameters) else None,
        result_filter=result_filter,
        batch_size=batch_size,
    )

    return lf_results
