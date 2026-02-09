# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for building pandapower N-1 contingency definitions and Pandera-validated convergence result DataFrames."""

import numpy as np
import pandapower
import pandas as pd
import pandera as pa
import pandera.typing as pat
from beartype.typing import Literal
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import PandapowerContingency
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    get_globally_unique_id,
)
from toop_engine_interfaces.loadflow_results import (
    ConvergedSchema,
)
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    LoadflowParameters,
    Nminus1Definition,
)


@pa.check_types
def get_convergence_df(
    timestep: int, contingency: PandapowerContingency, status: Literal["NO_CALCULATION", "CONVERGED", "FAILED"]
) -> pat.DataFrame[ConvergedSchema]:
    """Get the convergence dataframe for the given network and contingency

    Parameters
    ----------
    timestep : int
        The timestep of the results
    contingency: PandapowerContingency
        The contingency to compute the node results for
    status : Literal["NO_CALCULATION", "CONVERGED", "FAILED"]
        The status of the loadflow calculation. Can be one of:
        - "NO_CALCULATION": No loadflow was run, e.g. because no elements were outaged
        - "CONVERGED": The loadflow converged successfully
        - "FAILED": The loadflow failed to converge

    Returns
    -------
    pat.DataFrame[ConvergedSchema]
        The convergence dataframe for the given network and contingency
    """
    convergence_df = pd.DataFrame()
    convergence_df["timestep"] = [timestep]
    convergence_df["contingency"] = contingency.unique_id
    convergence_df["contingency_name"] = contingency.name or ""
    convergence_df["status"] = status
    convergence_df.set_index(["timestep", "contingency"], inplace=True)
    # fill missing columns with NaN
    convergence_df["warnings"] = ""
    convergence_df["iteration_count"] = np.nan

    return convergence_df


def get_full_nminus1_definition_pandapower(net: pandapower.pandapowerNet) -> Nminus1Definition:
    """Get the full N-1 definition from a pandapower network.

    This function retrieves the N-1 definition from a pandapower network, including:
        Monitored Elements
            all lines, trafos, buses and switches
        Contingencies
            all lines, trafos, generators and loads
            Basecase contingency with name "BASECASE"

    Parameters
    ----------
    net : pypowsybl.network.Network
        The Powsybl network to retrieve the N-1 definition from.

    Returns
    -------
    Nminus1Definition
        The complete N-1 definition for the given pandapower network.
    """
    lines = [
        GridElement(id=get_globally_unique_id(id, "line"), name=row["name"] or "", type="lines", kind="branch")
        for id, row in net.line.iterrows()
    ]
    trafo2w = [
        GridElement(id=get_globally_unique_id(id, "trafo"), name=row["name"] or "", type="trafo", kind="branch")
        for id, row in net.trafo.iterrows()
    ]
    trafos3w = [
        GridElement(id=get_globally_unique_id(id, "trafo3w"), name=row["name"] or "", type="trafo3w", kind="branch")
        for id, row in net.trafo3w.iterrows()
    ]

    branch_elements = [*lines, *trafo2w, *trafos3w]

    switches = [
        GridElement(id=get_globally_unique_id(id, "switch"), name=row["name"] or "", type="switch", kind="switch")
        for id, row in net.switch.iterrows()
    ]
    buses = [
        GridElement(id=get_globally_unique_id(id, "bus"), name=row["name"] or "", type="bus", kind="bus")
        for id, row in net.bus.iterrows()
    ]
    monitored_elements = [*branch_elements, *switches, *buses]

    generators = [
        GridElement(id=get_globally_unique_id(id, "gen"), name=row["name"] or "", type="gen", kind="injection")
        for id, row in net.gen.iterrows()
    ]
    loads = [
        GridElement(id=get_globally_unique_id(id, "load"), name=row["name"] or "", type="load", kind="injection")
        for id, row in net.load.iterrows()
    ]
    outaged_elements = [*branch_elements, *generators, *loads]

    basecase_contingency = [Contingency(id="BASECASE", name="BASECASE", elements=[])]
    single_contingencies = [
        Contingency(id=element.id, name=element.name or "", elements=[element]) for element in outaged_elements
    ]

    nminus1_definition = Nminus1Definition(
        contingencies=[*basecase_contingency, *single_contingencies],
        monitored_elements=monitored_elements,
        id_type="unique_pandapower",  # Default id type for Pandapower
        loadflow_parameters=LoadflowParameters(
            distributed_slack=False,  # This is the default for Powsybl
        ),
    )
    return nminus1_definition
