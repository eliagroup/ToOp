# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The importer's N-1 definition survives DC preprocessing, which writes its own DC N-1 definition."""

from pathlib import Path

import pytest
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_lf_params_from_fs
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import AreaSettings, CgmesImporterParameters
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition

INPUT_NMINUS1_DEFINITION_FILE = Path(__file__).parents[4] / "data/complex_grid/nminus1_definition_complex.json"


def _import_and_preprocess(tmp_path: Path, nminus1_definition_file: Path | None) -> tuple[Path, bytes]:
    """Run the importer and DC preprocessing on the complex grid.

    Returns the data folder and the importer's N-1 definition as written before DC preprocessing.
    """
    grid_path = tmp_path / "complex_grid.xiidm"
    create_complex_grid_battery_hvdc_svc_3w_trafo().save(grid_path)
    importer_parameters = CgmesImporterParameters(
        grid_model_file=grid_path,
        data_folder=tmp_path / "processed",
        fail_on_non_convergence=False,
        nminus1_definition_file=nminus1_definition_file,
        area_settings=AreaSettings(
            cutoff_voltage=220, control_area=["BE"], view_area=["BE", "NL"], nminus1_area=["BE", "NL"]
        ),
    )
    data_folder = preprocessing.convert_file(importer_parameters=importer_parameters).data_folder
    importer_definition_bytes = (data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"]).read_bytes()

    data_folder_dirfs = DirFileSystem(path=str(data_folder), fs=LocalFileSystem())
    lf_params = load_lf_params_from_fs(data_folder_dirfs, Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))
    load_grid(data_folder_dirfs=data_folder_dirfs, lf_params=lf_params)
    return data_folder, importer_definition_bytes


@pytest.mark.parametrize("with_input_definition", [True, False], ids=["input_definition", "mask_definition"])
def test_load_grid_keeps_importer_nminus1_definition(tmp_path: Path, with_input_definition: bool) -> None:
    data_folder, importer_definition_bytes = _import_and_preprocess(
        tmp_path, INPUT_NMINUS1_DEFINITION_FILE if with_input_definition else None
    )

    definition_path = data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"]
    assert definition_path.read_bytes() == importer_definition_bytes
    dc_definition = load_nminus1_definition(data_folder / PREPROCESSING_PATHS["dc_nminus1_definition_file_path"])
    assert dc_definition.contingencies[0].is_basecase()
    assert dc_definition.id_type == load_nminus1_definition(definition_path).id_type


def test_load_grid_keeps_input_contingencies_and_spps_rules(tmp_path: Path) -> None:
    """Journey A: the authoritative input definition reaches AC with all its cases and SPPS rules."""
    data_folder, _ = _import_and_preprocess(tmp_path, INPUT_NMINUS1_DEFINITION_FILE)

    definition = load_nminus1_definition(data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])
    input_definition = load_nminus1_definition(INPUT_NMINUS1_DEFINITION_FILE)
    assert [contingency.id for contingency in definition.contingencies] == [
        contingency.id for contingency in input_definition.contingencies
    ]
    assert definition.spps_rules is not None
    assert input_definition.spps_rules is not None
    assert [rule.scheme_name for rule in definition.spps_rules] == [rule.scheme_name for rule in input_definition.spps_rules]


def test_mask_definition_has_no_switch_only_contingencies(tmp_path: Path) -> None:
    """Journey B: switches are never outaged on their own, so no switch-only contingency reaches AC."""
    data_folder, _ = _import_and_preprocess(tmp_path, None)

    definition = load_nminus1_definition(data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])
    switch_only = [
        contingency.id
        for contingency in definition.contingencies
        if contingency.elements and all(element.type == "SWITCH" for element in contingency.elements)
    ]
    assert switch_only == []
