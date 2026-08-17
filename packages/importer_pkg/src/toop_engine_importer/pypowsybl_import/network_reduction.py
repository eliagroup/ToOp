# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Reduce the network to based on Voltagelevel and range."""

from beartype.typing import Union
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.cgmes.powsybl_masks_cgmes import (
    get_potentially_relevant_voltage_levels as get_potentially_relevant_voltage_levels_cgmes,
)
from toop_engine_importer.pypowsybl_import.ucte.powsybl_masks_ucte import (
    get_potentially_relevant_voltage_levels as get_potentially_relevant_voltage_levels_ucte,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    CgmesImporterParameters,
    UcteImporterParameters,
)


def reduce_network_to_view_area(
    net: Network, importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
) -> None:
    """Reduce the network to the view area based on the voltage level and range.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to be reduced.
        Note: The network is modified in place.
    importer_parameters : Union[UcteImporterParameters, CgmesImporterParameters]
        The importer parameters containing the view area and range.
    """
    if importer_parameters.data_type == "cgmes":
        voltage_level_ids = get_potentially_relevant_voltage_levels_cgmes(
            net=net,
            area_codes=importer_parameters.area_settings.control_area,
            cutoff_voltage=importer_parameters.area_settings.cutoff_voltage,
            # We need the full area, this short list may only have a few voltage levels
            select_by_voltage_level_id_list=None,
        )
    if importer_parameters.data_type == "ucte":
        voltage_level_ids = get_potentially_relevant_voltage_levels_ucte(
            net=net,
            area_codes=importer_parameters.area_settings.control_area,
            cutoff_voltage=importer_parameters.area_settings.cutoff_voltage,
            # We need the full area, this short list may only have a few voltage levels
            select_by_voltage_level_id_list=None,
        )

    # get vl_depths with importer setting
    vl_depths = [(vl_id, importer_parameters.network_reduction_voltage_level_range) for vl_id in voltage_level_ids]
    net.reduce_by_ids_and_depths(vl_depths=vl_depths, with_boundary_lines=False)
