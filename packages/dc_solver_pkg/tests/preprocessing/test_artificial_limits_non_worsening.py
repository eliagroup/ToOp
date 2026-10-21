# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Importer and DC preprocessing of the complex grid with every monitored branch non-worsening.

A non-worsening branch gets an artificial limit at least as high as its worst flow, so nothing can be critical.
"""

from dataclasses import dataclass

import polars as pl
import pypowsybl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.aggregate_results import aggregate_to_metric_batched
from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
from toop_engine_dc_solver.jax.topology_computations import default_topology
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_grid_helpers.powsybl.example_grids import (
    create_complex_grid_battery_hvdc_svc_3w_trafo,
    create_complex_grid_nminus1_definition,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_grid_helpers.powsybl.powsybl_helpers import save_lf_params_to_fs
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_importer.pypowsybl_import.preprocessing import ImportedGridModel
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import AreaSettings, CgmesImporterParameters
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition, save_nminus1_definition


@dataclass(frozen=True)
class NonWorseningComplexGrid:
    """The complex grid after the import and the DC preprocessing."""

    imported: ImportedGridModel
    static_information: StaticInformation
    monitored_branch_ids: list[str]


@pytest.fixture(scope="module")
def non_worsening_complex_grid(tmp_path_factory: pytest.TempPathFactory) -> NonWorseningComplexGrid:
    """Import the complex grid with all monitored branches non-worsening and run the DC preprocessing."""
    folder = tmp_path_factory.mktemp("complex_grid_non_worsening")
    net = create_complex_grid_battery_hvdc_svc_3w_trafo(connect_line_out_of_service=True)
    pypowsybl.loadflow.run_dc(net, CGMES_DISTRIBUTED_SLACK)
    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)

    definition = create_complex_grid_nminus1_definition()
    definition = definition.model_copy(
        update={
            "monitored_elements": [
                element.model_copy(update={"non_worsening": True, "optimized": False})
                if element.kind == "branch"
                else element
                for element in definition.monitored_elements
            ]
        }
    )
    input_nminus1_definition_file = folder / "input_nminus1_definition.json"
    save_nminus1_definition(input_nminus1_definition_file, definition)
    # The importer's DC limits then use the same slack distribution as the solver flows
    lf_params_file = folder / "input_loadflow_parameters.json"
    save_lf_params_to_fs(CGMES_DISTRIBUTED_SLACK, LocalFileSystem(), lf_params_file)

    imported = preprocessing.import_grid_with_security_analysis(
        importer_parameters=CgmesImporterParameters(
            grid_model_file=grid_file_path,
            data_folder=folder,
            input_nminus1_definition_file=input_nminus1_definition_file,
            loadflow_parameters_file=lf_params_file,
            create_artificial_limit_when_missing_permanent=True,
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
    _stats, static_information, _network_data = load_grid(data_folder_dirfs=DirFileSystem(str(folder)), pandapower=False)

    imported_definition = load_nminus1_definition(folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])
    monitored_branch_ids = [element.id for element in imported_definition.monitored_elements if element.kind == "branch"]
    assert all(element.non_worsening for element in imported_definition.monitored_elements if element.kind == "branch"), (
        "The importer must keep the non-worsening flags of the input definition"
    )
    return NonWorseningComplexGrid(imported, static_information, monitored_branch_ids)


def test_all_monitored_branches_have_the_artificial_limit(non_worsening_complex_grid: NonWorseningComplexGrid) -> None:
    """Every monitored branch with a loadflow result has an artificial limit for N-0 and N-1.

    This includes the branches without a permanent limit. Branches without a current, as outside the main
    component, have nothing to base it on.
    """
    initial_loadflow = non_worsening_complex_grid.imported.initial_loadflow
    limits = initial_loadflow.branch_limits.collect()
    with_current = initial_loadflow.branch_results.filter(pl.col("i").is_not_null() & pl.col("i").is_not_nan())
    with_result = set(with_current.select("element").unique().collect()["element"].to_list())
    monitored_branch_ids = set(non_worsening_complex_grid.monitored_branch_ids) & with_result
    assert len(monitored_branch_ids) > 50

    for limit_name in ("artificial_limit_n_0", "artificial_limit_n_1"):
        elements_with_limit = set(limits.filter(limits["limit_name"] == limit_name)["element"].to_list())
        assert monitored_branch_ids - elements_with_limit == set(), limit_name


def test_no_critical_branches_in_importer_and_dc_solver(non_worsening_complex_grid: NonWorseningComplexGrid) -> None:
    """Neither the AC and DC metrics of the importer nor the DC solver find a critical branch."""
    imported = non_worsening_complex_grid.imported
    static_information = non_worsening_complex_grid.static_information
    dynamic_information = static_information.dynamic_information

    lf_res, success = compute_symmetric_batch(
        default_topology(static_information.solver_config),
        None,
        None,
        nodal_inj_start_options=None,
        dynamic_information=dynamic_information,
        solver_config=static_information.solver_config,
    )
    assert success.all()
    # A non-worsening limit equals the worst flow, so the solver flow may exceed it by float noise
    branch_limits = dynamic_information.branch_limits
    branch_limits = replace(
        branch_limits,
        max_mw_flow=branch_limits.max_mw_flow * (1 + 1e-6),
        max_mw_flow_n_1=branch_limits.max_mw_flow_n_1 * (1 + 1e-6),
    )
    solver_critical_branches = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        dynamic_information.n_sub_relevant,
        metric="critical_branch_count_artificial_limit_n_1",
    )

    importer_critical_branches = imported.initial_metrics["critical_branch_count_artificial_limit_n_1"]
    assert importer_critical_branches == 0
    assert imported.import_result.overload_energy_artificial_dc_limit_n_1 == 0
    assert float(solver_critical_branches[0]) == importer_critical_branches
