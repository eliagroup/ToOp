# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Build cascade outage groups caused by current overloads."""

import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower.cascade.outage_groups.topology import (
    get_outage_group_for_elements,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR


def compute_current_overload_outage_group(
    net: pp.pandapowerNet,
    current_overloaded_element_df: pd.DataFrame,
) -> dict:
    """Build outage groups caused by current-overloaded branches.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network to inspect.
    current_overloaded_element_df : pd.DataFrame
        Branch rows above the loading threshold.

    Returns
    -------
    dict
        Mapping from overloaded element id to the elements that should be outaged.
    """
    if current_overloaded_element_df is None or current_overloaded_element_df.empty:
        return {}

    contingency_elements = {
        row.element: [row.element.split(SEPARATOR)] for row in current_overloaded_element_df.itertuples()
    }
    return get_outage_group_for_elements(net, contingency_elements)
