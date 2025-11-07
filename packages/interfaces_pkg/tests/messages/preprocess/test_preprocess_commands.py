from pathlib import Path

import pytest
from toop_engine_interfaces.messages.preprocess.preprocess_commands_factory import (
    get_area_settings,
    get_command_wrapper,
    get_default_importer_params,
    get_importer_params,
    get_limit_adjustment_parameters,
    get_parameters_for_case,
    get_preprocess_parameters,
    get_shutdown_command,
    get_start_preprocessing_command,
)


def test_limit_adjustment_parameters():
    params = get_limit_adjustment_parameters(
        n_0_factor=1.2,
        n_1_factor=1.4,
        n_0_min_increase=0.05,
        n_1_min_increase=0.05,
    )
    assert params.n_0_factor == 1.2
    assert params.n_1_factor == 1.4
    assert params.n_0_min_increase == 0.05
    assert params.n_1_min_increase == 0.05

    n0_params = get_parameters_for_case(params, "n0")
    assert n0_params == (1.2, 0.05)

    n1_params = get_parameters_for_case(params, "n1")
    assert n1_params == (1.4, 0.05)

    with pytest.raises(ValueError):
        get_parameters_for_case(params, "n2")

    with pytest.raises(ValueError):
        get_limit_adjustment_parameters(
            n_0_factor=-1.2,
            n_1_factor=1.4,
            n_0_min_increase=0.05,
            n_1_min_increase=0.05,
        )


def test_base_importer_parameters():
    params = get_importer_params(
        area_settings=get_area_settings(
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
    assert params.data_folder == "/some/path"
    assert params.data_type == "ucte"
    assert params.grid_model_file == "/some/grid/model.uct"
    assert params.area_settings.dso_trafo_weight == 1.0
    assert params.area_settings.border_line_weight == 1.0

    # with pytest.raises(TypeError):
    #     params = get_importer_params(
    #         area_settings=get_area_settings(
    #             control_area=["D1", "D2"],
    #             view_area=["D3", "D4"],
    #             nminus1_area=["D5", "D6"],
    #             cutoff_voltage=220,
    #         ),
    #         data_folder=Path("/some/path"),
    #         grid_model_file=Path("/some/grid/model.uct"),
    #         data_type="Not a valid type",
    #     )

    limit_params = get_limit_adjustment_parameters(
        n_0_factor=1.2,
        n_1_factor=1.4,
        n_0_min_increase=0.05,
        n_1_min_increase=0.05,
    )

    params = get_importer_params(
        area_settings=get_area_settings(
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
    params = get_default_importer_params(data_folder=Path("/some/path"), grid_model_file=Path("/some/other/path"), ucte=True)
    assert params.area_settings.control_area == ["D8"]
    assert params.area_settings.view_area == ["D2", "D4", "D7", "D8"]
    assert params.area_settings.nminus1_area == ["D2", "D4", "D7", "D8"]
    assert params.area_settings.cutoff_voltage == 220
    assert params.data_folder == "/some/path"
    assert params.grid_model_file == "/some/other/path"
    assert params.data_type == "ucte"


def test_ucte_importer_parameters_missing_required():
    with pytest.raises(TypeError):
        get_default_importer_params(data_folder=Path("/some/path"), ucte=True)
    with pytest.raises(TypeError):
        get_default_importer_params(grid_model_file=Path("/some/other/path"), ucte=True)


def test_ucte_importer_parameters_optional():
    params = get_default_importer_params(
        data_folder=Path("/some/path"), grid_model_file=Path("/some/grid/model.uct"), ucte=True
    )
    params.white_list_file = "/some/white/list"
    params.black_list_file = "/some/black/list"
    assert params.white_list_file == "/some/white/list"
    assert params.black_list_file == "/some/black/list"


def test_cgmes_import_parameter():
    params = get_default_importer_params(
        data_folder=Path("/some/path"), grid_model_file=Path("/some/cgmes/model.zip"), cgmes=True
    )
    assert params.area_settings.control_area == ["BE"]
    assert params.area_settings.view_area == ["BE", "LU", "D4", "D2", "NL", "FR"]
    assert params.area_settings.nminus1_area == ["BE"]
    assert params.area_settings.cutoff_voltage == 220
    assert params.data_folder == "/some/path"
    assert params.grid_model_file == "/some/cgmes/model.zip"
    assert params.data_type == "cgmes"


def test_preprocess_parameters():
    params = get_preprocess_parameters(double_limit_n0=0.9, double_limit_n1=0.9)
    assert params.double_limit_n0 == 0.9
    assert params.double_limit_n1 == 0.9


def test_start_preprocessing_command():
    importer_params = get_default_importer_params(
        data_folder=Path("/some/path"), grid_model_file=Path("/some/ucte/file.uct"), ucte=True
    )
    preprocess_params = get_preprocess_parameters(double_limit_n0=0.9, double_limit_n1=0.9)
    command = get_start_preprocessing_command(
        importer_parameters=importer_params, preprocess_parameters=preprocess_params, preprocess_id="test_id"
    )
    assert command.importer_parameters == importer_params
    assert command.preprocess_parameters == preprocess_params
    assert command.preprocess_id == "test_id"


def test_shutdown_command():
    command = get_shutdown_command(exit_code=0)
    assert command.exit_code == 0

    command = get_shutdown_command()
    assert command.exit_code == 0


def test_command_wrapper():
    start_command = get_start_preprocessing_command(
        importer_parameters=get_default_importer_params(
            data_folder=Path("/some/path"), grid_model_file=Path("/some/ucte/file.uct"), ucte=True
        ),
        preprocess_parameters=get_preprocess_parameters(),
        preprocess_id="test_id",
    )
    command = get_command_wrapper(command=start_command)
    assert command.start_preprocessing == start_command

    shutdown_command = get_shutdown_command(exit_code=1)
    command = get_command_wrapper(command=shutdown_command)
    assert command.shutdown == shutdown_command
