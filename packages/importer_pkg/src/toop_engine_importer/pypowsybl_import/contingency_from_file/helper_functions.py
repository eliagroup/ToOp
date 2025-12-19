# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Powsybl contingency from file helper functions.

Author: Benjamin Petrick
Created: 2025-05-13
"""

import pandas as pd
import pypowsybl
from toop_engine_importer.contingency_from_power_factory.power_factory_data_class import AllGridElementsSchema


# TODO: consider refactoring from grid_helpers.powsybl.powsybl_asset_topo import get_all_element_names
def get_all_element_names(net: pypowsybl.network.Network) -> AllGridElementsSchema:
    """Get all element names from the network.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to get the element names from.

    Returns
    -------
    AllGridElementsSchema
        A DataFrame containing the element names and their types.
    """
    attributes = ["name"]
    vl = net.get_voltage_levels(attributes=attributes)
    vl["element_type"] = "BUS"

    busbar_sections = net.get_busbar_sections(attributes=attributes)
    busbar_sections["element_type"] = "BUSBAR_SECTION"

    lines = net.get_lines(attributes=attributes)
    lines["element_type"] = "LINE"

    transformers_2w = net.get_2_windings_transformers(attributes=attributes)
    transformers_2w["element_type"] = "TWO_WINDINGS_TRANSFORMER"

    transformers_3w = net.get_3_windings_transformers(attributes=attributes)
    transformers_3w["element_type"] = "THREE_WINDINGS_TRANSFORMER"

    switches = net.get_switches(attributes=attributes)
    switches["element_type"] = "SWITCH"

    generators = net.get_generators(attributes=attributes)
    generators["element_type"] = "GENERATOR"

    loads = net.get_loads(attributes=attributes)
    loads["element_type"] = "LOAD"

    shunt_compensators = net.get_shunt_compensators(attributes=attributes)
    shunt_compensators["element_type"] = "SHUNT_COMPENSATOR"

    dangling = net.get_dangling_lines(attributes=attributes)
    dangling["element_type"] = "DANGLING_LINE"

    tie = net.get_tie_lines(attributes=attributes)
    tie["element_type"] = "TIE_LINE"

    all_elements = pd.concat(
        [
            vl,
            busbar_sections,
            lines,
            transformers_2w,
            transformers_3w,
            switches,
            generators,
            loads,
            shunt_compensators,
            dangling,
            tie,
        ],
        ignore_index=False,
    )
    all_elements = all_elements.reset_index(drop=False)
    all_elements.rename(columns={"id": "grid_model_id", "name": "grid_model_name"}, inplace=True)
    AllGridElementsSchema.validate(all_elements)

    return all_elements
