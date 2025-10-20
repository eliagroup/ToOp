"""Get the results of the AC Contingency Analysis for the given network"""

from beartype.typing import Optional
from pandapower import pandapowerNet as PandapowerNetwork
from pypowsybl.network import Network as PowsyblNetwork
from toop_engine_contingency_analysis.pandapower import (
    run_contingency_analysis_pandapower,
)
from toop_engine_contingency_analysis.pypowsybl import (
    run_contingency_analysis_powsybl,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


def get_ac_loadflow_results(
    net: PandapowerNetwork | PowsyblNetwork,
    n_minus_1_definition: Nminus1Definition,
    timestep: int = 0,
    job_id: str = "",
    n_processes: int = 1,
    batch_size: Optional[int] = None,
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

    Returns
    -------
    LoadflowResultsPolars
        The results of the Contingency analysis

    Raises
    ------
    ValueError
        If the network is not a PandapowerNetwork or PowsyblNetwork
    """
    if isinstance(net, PandapowerNetwork):
        lf_results = run_contingency_analysis_pandapower(
            net,
            n_minus_1_definition,
            job_id,
            timestep,
            n_processes=n_processes,
            batch_size=batch_size,
            method="ac",
            polars=True,
        )
    elif isinstance(net, PowsyblNetwork):
        lf_results = run_contingency_analysis_powsybl(
            net,
            n_minus_1_definition,
            job_id,
            timestep,
            n_processes=n_processes,
            method="ac",
            polars=True,
        )
    else:
        raise ValueError("net must be a pandapowerNet or powsybl network")

    return lf_results
