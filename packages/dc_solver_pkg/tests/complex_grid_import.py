# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Import of the complex test grid with an input N-1 definition, shared by the DC end-to-end tests."""

from pathlib import Path

import pypowsybl
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_grid_helpers.powsybl.powsybl_helpers import save_lf_params_to_fs
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import AreaSettings, CgmesImporterParameters
from toop_engine_interfaces.nminus1_definition import Nminus1Definition, save_nminus1_definition


def import_complex_grid(folder: Path, nminus1_definition: Nminus1Definition) -> tuple[StaticInformation, NetworkData]:
    """Import the complex grid with ``nminus1_definition`` into ``folder`` and run DC preprocessing."""
    net = create_complex_grid_battery_hvdc_svc_3w_trafo(connect_line_out_of_service=True)
    pypowsybl.loadflow.run_dc(net, CGMES_DISTRIBUTED_SLACK)
    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)
    nminus1_definition_file = folder / "input_nminus1_definition.json"
    save_nminus1_definition(nminus1_definition_file, nminus1_definition)

    preprocessing.convert_file(
        importer_parameters=CgmesImporterParameters(
            grid_model_file=grid_file_path,
            data_folder=folder,
            nminus1_definition_file=nminus1_definition_file,
            fail_on_non_convergence=False,
            area_settings=AreaSettings(
                cutoff_voltage=1.0,
                control_area=["BE", "NL"],
                view_area=["BE", "NL"],
                nminus1_area=["BE", "NL"],
                dso_trafo_factors=None,
                dso_trafo_weight=1.0,
                border_line_factors=None,
                border_line_weight=1.0,
            ),
        )
    )
    _stats, static_information, network_data = load_grid(data_folder_dirfs=DirFileSystem(str(folder)), pandapower=False)
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(str(folder)), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )
    return static_information, network_data
