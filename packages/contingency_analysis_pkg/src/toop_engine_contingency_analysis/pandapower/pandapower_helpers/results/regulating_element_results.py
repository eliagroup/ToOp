# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


"""Utilities for extracting pandapower regulating elements simulation results per contingency."""

import pandera as pa
import pandera.typing as pat
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
    PandapowerMonitoredElementSchema,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import (
    RegulatingElementResultSchema,
    RegulatingElementType,
)


@pa.check_types
def get_regulating_element_results(
    timestep: int, monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema], contingency: PandapowerContingency
) -> pat.DataFrame[RegulatingElementResultSchema]:
    """Get the regulating element results for the given network and contingency.

    This currently only returns fake slack bus and generator results for the basecase

    Parameters
    ----------
    timestep : int
        The timestep of the results
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
        The dataframe containing the monitored elements
    contingency: PandapowerContingency
        The contingency to compute the regulating element results for

    Returns
    -------
    pat.DataFrame[RegulatingElementResultSchema]
        The regulating element results for the given network and contingency
    """
    if monitored_elements.empty:
        # If no elements are monitored, return an empty dataframe
        return get_empty_dataframe_from_model(RegulatingElementResultSchema)
    regulating_elements = get_empty_dataframe_from_model(RegulatingElementResultSchema)
    # TODO dont fake this
    if len(contingency.elements) == 0:
        regulating_elements.loc[(timestep, contingency.unique_id, monitored_elements.index[0]), "value"] = -9999.0
        regulating_elements.loc[
            (timestep, contingency.unique_id, monitored_elements.index[0]), "regulating_element_type"
        ] = RegulatingElementType.GENERATOR_Q.value
        regulating_elements.loc[(timestep, contingency.unique_id, monitored_elements.index[1]), "value"] = 9999.0
        regulating_elements.loc[
            (timestep, contingency.unique_id, monitored_elements.index[1]), "regulating_element_type"
        ] = RegulatingElementType.SLACK_P.value
    regulating_elements["element_name"] = ""
    regulating_elements["contingency_name"] = ""
    return regulating_elements
