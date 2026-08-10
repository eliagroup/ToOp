# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


"""Utilities for extracting pandapower regulating elements simulation results per contingency."""

import pandera as pa
import pandera.typing.polars as patpl
import polars as pl
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
)
from toop_engine_interfaces.interface_helpers import get_empty_polars_dataframe_from_model
from toop_engine_interfaces.loadflow_results import (
    RegulatingElementType,
)
from toop_engine_interfaces.loadflow_results_polars import RegulatingElementResultSchemaPolars


@pa.check_types
def get_regulating_element_results(
    timestep: int, monitored_element_ids: pl.Series, contingency: PandapowerContingency
) -> patpl.DataFrame[RegulatingElementResultSchemaPolars]:
    """Get the regulating element results for the given network and contingency.

    This currently only returns fake slack bus and generator results for the basecase.

    Parameters
    ----------
    timestep : int
        The timestep of the results
    monitored_element_ids : pl.Series
        Globally unique ids of the monitored elements (``ResultConstants.monitored_element_ids``).
    contingency : PandapowerContingency
        The contingency to compute the regulating element results for

    Returns
    -------
    patpl.DataFrame[RegulatingElementResultSchemaPolars]
        Flat polars frame with ``timestep``/``contingency``/``element`` as ordinary columns,
        following the RegulatingElementResultSchema layout.
    """
    # Only the base case (no outaged elements) produces (placeholder) results today.
    if monitored_element_ids.is_empty() or len(contingency.elements) != 0:
        return get_empty_polars_dataframe_from_model(RegulatingElementResultSchemaPolars)

    return pl.DataFrame(
        {
            "timestep": [timestep, timestep],
            "contingency": [contingency.unique_id, contingency.unique_id],
            "element": [monitored_element_ids[0], monitored_element_ids[1]],
            "value": [-9999.0, 9999.0],
            "regulating_element_type": [
                RegulatingElementType.GENERATOR_Q.value,
                RegulatingElementType.SLACK_P.value,
            ],
            "element_name": ["", ""],
            "contingency_name": ["", ""],
        },
        schema=get_empty_polars_dataframe_from_model(RegulatingElementResultSchemaPolars).schema,
    )
