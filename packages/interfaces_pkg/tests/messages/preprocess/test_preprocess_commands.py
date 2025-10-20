from pathlib import Path

import pytest
from pydantic import ValidationError
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    BaseImporterParameters,
    CgmesImporterParameters,
    Command,
    LimitAdjustmentParameters,
    PreprocessParameters,
    ShutdownCommand,
    StartPreprocessingCommand,
    UcteImporterParameters,
)


def test_limit_adjustment_parameters():
    params = LimitAdjustmentParameters(
        n_0_factor=1.2,
        n_1_factor=1.4,
        n_0_min_increase=0.05,
        n_1_min_increase=0.05,
    )
    assert params.n_0_factor == 1.2
    assert params.n_1_factor == 1.4
    assert params.n_0_min_increase == 0.05
    assert params.n_1_min_increase == 0.05

    n0_params = params.get_parameters_for_case("n0")
    assert n0_params == (1.2, 0.05)

    n1_params = params.get_parameters_for_case("n1")
    assert n1_params == (1.4, 0.05)

    with pytest.raises(ValueError):
        params.get_parameters_for_case("invalid_case")

    with pytest.raises(ValidationError):
        LimitAdjustmentParameters(
            n_0_factor=-1.2,
            n_1_factor=1.4,
            n_0_min_increase=0.05,
            n_1_min_increase=0.05,
        )

    assert params.model_dump_json() is not None


def test_base_importer_parameters():
    params = BaseImporterParameters(
        area_settings=AreaSettings(
            control_area=["BE", "D2"],
            view_area=["LU", "D4"],
            nminus1_area=["FR", "D8"],
            cutoff_voltage=220,
        ),
        data_folder=Path("/some/path"),
        data_type="ucte",
        grid_model_file=Path("/some/grid/model.uct"),
    )
    assert params.area_settings.control_area == ["BE", "D2"]
    assert params.area_settings.view_area == ["LU", "D4"]
    assert params.area_settings.nminus1_area == ["FR", "D8"]
    assert params.area_settings.cutoff_voltage == 220
    assert params.data_folder == Path("/some/path")
    assert params.data_type == "ucte"
    assert params.grid_model_file == Path("/some/grid/model.uct")
    assert params.area_settings.dso_trafo_weight == 1.0
    assert params.area_settings.border_line_weight == 1.0

    assert params.model_dump_json() is not None

    params2 = BaseImporterParameters.model_validate_json(params.model_dump_json())
    assert params == params2
    assert params.area_settings == params2.area_settings

    with pytest.raises(ValidationError):
        params = BaseImporterParameters(
            area_settings=AreaSettings(
                control_area=["D1", "D2"],
                view_area=["D3", "D4"],
                nminus1_area=["D5", "D6"],
                cutoff_voltage=220,
            ),
            data_folder=Path("/some/path"),
            grid_model_file=Path("/some/grid/model.uct"),
            data_type="Not a valid type",
        )

    limit_params = LimitAdjustmentParameters(
        n_0_factor=1.2,
        n_1_factor=1.4,
        n_0_min_increase=0.05,
        n_1_min_increase=0.05,
    )

    params = BaseImporterParameters(
        area_settings=AreaSettings(
            control_area=["BE", "D2"],
            view_area=["LU", "D4"],
            nminus1_area=["FR", "D8"],
            cutoff_voltage=220,
            dso_trafo_weight=2.0,
            border_line_weight=2.0,
            dso_trafo_factors=limit_params,
            border_line_factors=limit_params,
        ),
        data_folder=Path("/some/path"),
        data_type="ucte",
        grid_model_file=Path("/some/grid/model.uct"),
    )
    assert params.area_settings.dso_trafo_weight == 2.0
    assert params.area_settings.border_line_weight == 2.0


def test_ucte_importer_parameters():
    params = UcteImporterParameters(data_folder=Path("/some/path"), grid_model_file=Path("/some/other/path"))
    assert params.area_settings.control_area == ["D8"]
    assert params.area_settings.view_area == ["D2", "D4", "D7", "D8"]
    assert params.area_settings.nminus1_area == ["D2", "D4", "D7", "D8"]
    assert params.area_settings.cutoff_voltage == 220
    assert params.data_folder == Path("/some/path")
    assert params.grid_model_file == Path("/some/other/path")
    assert params.data_type == "ucte"


def test_ucte_importer_parameters_missing_required():
    with pytest.raises(ValidationError):
        UcteImporterParameters(grid_model_file=Path("/some/other/path"))
    with pytest.raises(ValidationError):
        UcteImporterParameters(data_folder=Path("/some/path"))


def test_ucte_importer_parameters_optional():
    params = UcteImporterParameters(
        data_folder=Path("/some/path"),
        grid_model_file=Path("/some/grid/model.uct"),
        white_list_file=Path("/some/white/list"),
        black_list_file=Path("/some/black/list"),
    )
    assert params.white_list_file == Path("/some/white/list")
    assert params.black_list_file == Path("/some/black/list")

    assert params.model_dump_json() is not None


def test_cgmes_import_parameter():
    params = CgmesImporterParameters(
        data_folder=Path("/some/path"),
        grid_model_file=Path("/some/cgmes/model.zip"),
    )
    assert params.area_settings.control_area == ["BE"]
    assert params.area_settings.view_area == ["BE", "LU", "D4", "D2", "NL", "FR"]
    assert params.area_settings.nminus1_area == ["BE"]
    assert params.area_settings.cutoff_voltage == 220
    assert params.data_folder == Path("/some/path")
    assert params.grid_model_file == Path("/some/cgmes/model.zip")
    assert params.data_type == "cgmes"


def test_preprocess_parameters():
    params = PreprocessParameters(double_limit_n0=0.9, double_limit_n1=0.9)
    assert params.double_limit_n0 == 0.9
    assert params.double_limit_n1 == 0.9


def test_start_preprocessing_command():
    importer_params = UcteImporterParameters(data_folder=Path("/some/path"), grid_model_file=Path("/some/ucte/file.uct"))
    preprocess_params = PreprocessParameters(compute_branch_actions=True, double_limit_n0=0.9, double_limit_n1=0.9)
    command = StartPreprocessingCommand(
        importer_parameters=importer_params, preprocess_parameters=preprocess_params, preprocess_id="test_id"
    )
    assert command.importer_parameters == importer_params
    assert command.preprocess_parameters == preprocess_params
    assert command.preprocess_id == "test_id"


def test_shutdown_command():
    command = ShutdownCommand(exit_code=0)
    assert command.exit_code == 0

    command = ShutdownCommand()
    assert command.exit_code == 0


def test_command_wrapper():
    start_command = StartPreprocessingCommand(
        importer_parameters=UcteImporterParameters(
            data_folder=Path("/some/path"), grid_model_file=Path("/some/ucte/file.uct")
        ),
        preprocess_parameters=PreprocessParameters(),
        preprocess_id="test_id",
    )
    command = Command(command=start_command)
    assert command.command == start_command

    shutdown_command = ShutdownCommand(exit_code=1)
    command = Command(command=shutdown_command)
    assert command.command == shutdown_command
