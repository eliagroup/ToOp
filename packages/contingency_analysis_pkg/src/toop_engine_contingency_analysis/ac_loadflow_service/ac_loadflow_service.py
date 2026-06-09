# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Get the results of the AC Contingency Analysis for the given network"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import pypowsybl
from beartype.typing import Optional
from pypowsybl.network import Network as PowsyblNetwork
from toop_engine_interfaces.nminus1_definition import Nminus1Definition

if TYPE_CHECKING:
    from pandapower import pandapowerNet
    from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars


def _is_missing_pandapower_dependency(exc: ModuleNotFoundError) -> bool:
    """Return whether the import failed because pandapower is not installed."""
    return (exc.name == "pandapower" or (exc.name is not None and exc.name.startswith("pandapower."))) or (
        "pandapower" in str(exc)
    )


def _is_pandapower_network(net: object) -> bool:
    """Return whether the provided network is a pandapower network."""
    try:
        pandapower_network_type = import_module("pandapower").pandapowerNet
    except ModuleNotFoundError as exc:
        if not _is_missing_pandapower_dependency(exc):
            raise
        return False
    return isinstance(net, pandapower_network_type)


def get_ac_loadflow_results(
    net: pandapowerNet | PowsyblNetwork,
    n_minus_1_definition: Nminus1Definition,
    timestep: int = 0,
    job_id: str = "",
    n_processes: int = 1,
    batch_size: Optional[int] = None,
    lf_params: pypowsybl.loadflow.Parameters | dict | None = None,
) -> LoadflowResultsPolars:
    """Get the results of the AC loadflow for the given network

    Parameters
    ----------
    net : pp.pandapowerNet | pypowsybl.network.Network
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
        The size of the batches to use for the parallelization.
        This is ignored for Powsybl at the moment.
        If None, the batch size is computed based on the number of contingencies and the number of processes.
    lf_params: pypowsybl.loadflow.Parameters or dict, optional
        Loadflow parameters to use for the computation.
        dict for pandapower, pypowsybl.loadflow.Parameters for powsybl. If None, default parameters are used.

    Returns
    -------
    LoadflowResultsPolars
        The results of the Contingency analysis

    Raises
    ------
    ValueError
        If the network is not a PandapowerNetwork or PowsyblNetwork
    """
    if _is_pandapower_network(net):
        pandapower_module = import_module("toop_engine_contingency_analysis.pandapower")
        pandapower_schema_module = import_module("toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas")

        cfg = pandapower_schema_module.ContingencyAnalysisConfig(
            runpp_kwargs=lf_params if isinstance(lf_params, dict) else None,
            method="ac",
            polars=True,
            parallel=pandapower_schema_module.ParallelConfig(
                n_processes=n_processes,
                batch_size=batch_size,
            ),
        )
        lf_results = pandapower_module.run_contingency_analysis_pandapower(
            net,
            n_minus_1_definition,
            job_id,
            timestep,
            cfg=cfg,
        )
    elif isinstance(net, PowsyblNetwork):
        pypowsybl_module = import_module("toop_engine_contingency_analysis.pypowsybl")

        lf_results = pypowsybl_module.run_contingency_analysis_powsybl(
            net,
            n_minus_1_definition,
            job_id,
            timestep,
            n_processes=n_processes,
            method="ac",
            polars=True,
            lf_params=lf_params if isinstance(lf_params, pypowsybl.loadflow.Parameters) else None,
        )
    else:
        raise ValueError("net must be a pandapowerNet or powsybl network")

    return lf_results
