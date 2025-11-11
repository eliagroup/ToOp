"""The factory functions to create messages for the preprocessing engine."""

import uuid
from datetime import datetime
from pathlib import Path

from beartype.typing import Literal, Optional, TypeAlias, Union
from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import PositiveFloat, PositiveInt
from toop_engine_interfaces.messages.protobuf_schema.preprocess.preprocess_commands_pb2 import (
    AreaSettings,
    BaseImporterParameters,
    Command,
    LimitAdjustmentParameters,
    PreprocessParameters,
    ShutdownCommand,
    StartPreprocessingCommand,
)

UCTERegionType: TypeAlias = Literal[
    "A",
    "B",
    "C",
    "D",
    "D1",
    "D2",
    "D4",
    "D6",
    "D7",
    "D8",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "2",
    "_",
]

# As defined by CGMES: ISO 3166
CGMESRegionType: TypeAlias = Literal[
    "AL",
    "AD",
    "AM",
    "AT",
    "AZ",
    "BY",
    "BE",
    "BA",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "GE",
    "DE",
    "GR",
    "HU",
    "IS",
    "IE",
    "IT",
    "LV",
    "LI",
    "LT",
    "LU",
    "MT",
    "MD",
    "MC",
    "ME",
    "NL",
    "MK",
    "NO",
    "PL",
    "PT",
    "RO",
    "RU",
    "SM",
    "RS",
    "SK",
    "SI",
    "ES",
    "SE",
    "CH",
    "TR",
    "UA",
    "GB",
    "VA",
]

GridModelType: TypeAlias = Literal["ucte", "cgmes"]

RealiseStationBusbarChoiceHeuristic: TypeAlias = Literal["first", "least_connected_busbar"]


# Use the Empty string to select all regions
AllCountriesRegionType: TypeAlias = Literal[""]

RegionType: TypeAlias = Union[UCTERegionType, CGMESRegionType, AllCountriesRegionType]
GridModelType: TypeAlias = Literal["ucte", "cgmes"]


def create_limit_adjustment_parameters(
    n_0_factor: float = 1.2, n_1_factor: float = 1.4, n_0_min_increase: float = 0.05, n_1_min_increase: float = 0.1
) -> LimitAdjustmentParameters:
    """
    Parameters for the adjustment of limits of special branches.

    The new operational limits will be created like this:
        1) Compute AC Loadflow
        2) new_limit = current_flow * n_0_factor (or n_1_factor)
        3) If new_limit > min_increase (=old_limit * n_0_min_increase)
            new_limit = min_increase
        4) If new_limit > old_limit
            new_limit = old_limit

    Parameters
    ----------
    n_0_factor : float, optional
        The factor for the N-0 current limit. Default is 1.2.
    n_1_factor : float, optional
        The factor for the N-1 current limit. Default is 1.4.
    n_0_min_increase : float, optional
        The minimal allowed load increase in percent for the n-0 case. This value multiplied with the current limit
        gives a lower border for the new limit. This makes sure, that lines that currently are barely loaded
        can be used at all. Default is 5%.
    n_1_min_increase : float, optional
        The minimal allowed load increase in percent for the n-1 case. This value multiplied with the current limit
        gives a lower border for the new limit. This makes sure, that lines that currently are barely loaded
        can be used at all. Default is 10%.

    Returns
    -------
    LimitAdjustmentParameters
        The limit adjustment parameters with the specified factors and minimum increases.
    """
    if n_0_factor < 0:
        raise ValueError("n_0_factor must be non-negative.")
    if n_1_factor < 0:
        raise ValueError("n_1_factor must be non-negative.")
    if n_0_min_increase < 0:
        raise ValueError("n_0_min_increase must be non-negative.")
    if n_1_min_increase < 0:
        raise ValueError("n_1_min_increase must be non-negative.")
    params = LimitAdjustmentParameters()
    params.n_0_factor = n_0_factor
    params.n_1_factor = n_1_factor
    params.n_0_min_increase = n_0_min_increase
    params.n_1_min_increase = n_1_min_increase
    return params


def create_parameters_for_case(
    params: LimitAdjustmentParameters,
    case: Literal["n0", "n1"],
) -> tuple[PositiveFloat, PositiveFloat]:
    """Get the factors for the specific case

    Parameters
    ----------
    params : LimitAdjustmentParameters
        The limit adjustment parameters
    case: Literal["n0", "n1"]
        Which case should be returned

    Returns
    -------
    tuple[PositiveFloat, PositiveFloat]
        The factor and the minimal increase
    """
    if case == "n0":
        return params.n_0_factor, params.n_0_min_increase
    if case == "n1":
        return params.n_1_factor, params.n_1_min_increase
    raise ValueError(f"Case {case} not defined")


def create_area_settings(
    control_area: list[RegionType],
    view_area: list[RegionType],
    nminus1_area: list[RegionType],
    cutoff_voltage: PositiveInt = 220,
    dso_trafo_factors: Optional[LimitAdjustmentParameters] = None,
    dso_trafo_weight: Optional[PositiveFloat] = 1.0,
    border_line_factors: Optional[LimitAdjustmentParameters] = None,
    border_line_weight: Optional[PositiveFloat] = 1.0,
) -> AreaSettings:
    """
    Get settings related to the areas that are imported.

    Parameters
    ----------
    control_area : list[RegionType]
        The area in which switching can take place. Substations from this area will automatically
        become relevant substations (switchable) except they are below the cutoff voltage. Also lines
        in this area will become disconnectable.
    view_area : list[RegionType]
        The areas in which branches shall be part of the overload computation, i.e. for which regions
        shall line flows be computed.
    nminus1_area : list[RegionType]
        The areas where elements shall be part of the N-1 computation, i.e. which elements to fail.
    cutoff_voltage : PositiveInt, optional
        The cutoff voltage under which to ignore equipment. Equipment that doesn't have at least one
        end equal or above this nominal voltage will not be part of the reward/nminus1 computation. Default is 220.
    dso_trafo_factors : Optional[LimitAdjustmentParameters], optional
        If given, the N-0 and N-1 flows across the dso trafos in the specified region will be limited
        to the current N-0 flows in the unsplit configuration. For each case (n0 or n1) a new operational limit
        with the name "border_limit_n0"/"border_limit_n1" is added.
    dso_trafo_weight : Optional[PositiveFloat], optional
        A weight that is used for trafos that leave the n-1 area, to underlying DSOs. Default is 1.0.
    border_line_factors : Optional[LimitAdjustmentParameters], optional
        If given, the N-0 and N-1 flows across the border lines leaving or entering the specified region will be limited
        to the current N-0 flows in the unsplit configuration. For each case (n0 or n1) a new operational limit
        with the name "border_limit_n0"/"border_limit_n1" is added.
    border_line_weight : Optional[PositiveFloat], optional
        A weight that is used for lines that leave the n-1 area, to neighbouring TSOs. Default is 1.0.

    Returns
    -------
    AreaSettings
        The area settings
        The factor for the N-0 current limit. Default is 1.2.
    n_1_factor : float, optional
        The factor for the N-1 current limit. Default is 1.4.
    n_0_min_increase : float, optional
        The minimal allowed load increase in percent for the n-0 case. This value multiplied with the current limit
        gives a lower border for the new limit. This makes sure, that lines that currently are barely loaded
        can be used at all. Default is 5%.
    n_1_min_increase : float, optional
        The minimal allowed load increase in percent for the n-1 case. This value multiplied with the current limit
        gives a lower border for the new limit. This makes sure, that lines that currently are barely loaded
        can be used at all. Default is 10%.

    Returns
    -------
    LimitAdjustmentParameters
        The limit adjustment parameters with the specified factors and minimum increases.
    """
    area_settings = AreaSettings()
    area_settings.control_area.extend(control_area)
    area_settings.view_area.extend(view_area)
    area_settings.nminus1_area.extend(nminus1_area)
    area_settings.cutoff_voltage = cutoff_voltage
    if dso_trafo_factors is not None:
        area_settings.dso_trafo_factors.CopyFrom(dso_trafo_factors)
    area_settings.dso_trafo_weight = dso_trafo_weight

    if border_line_factors is not None:
        area_settings.border_line_factors.CopyFrom(border_line_factors)
    area_settings.border_line_weight = border_line_weight
    return area_settings


def create_importer_params(
    area_settings: AreaSettings,
    data_folder: Path,
    grid_model_file: Path,
    data_type: GridModelType,
    white_list_file: Optional[Path] = None,
    black_list_file: Optional[Path] = None,
    ignore_list_file: Optional[Path] = None,
    ingress_id: Optional[str] = None,
    contingency_list_file: Optional[Path] = None,
    schema_format: Optional[Literal["ContingencyImportSchemaPowerFactory", "ContingencyImportSchema"]] = None,
) -> BaseImporterParameters:
    """
    Parameters that are required to import any data format.

    Parameters
    ----------
    area_settings : AreaSettings
        Which areas of the grid are to be imported and how to handle boundaries.
    data_folder : Path
        The path where the entry point where the timestep data folder structure starts.
        The folder structure is defined in dc_solver.interfaces.folder_structure. This folder is relative to the
        processed_grid_folder that is configured in the backend/importer. A typical default would be grid_model_file.stem.
    grid_model_file : Path
        The path to the input grid model file.
        This file should contain the grid model in the format defined by the data_type.
        For instance a .uct for UCTE data or a .zip for CGMES data.
    data_type : Final[GridModelType]
        The type of data that is being imported.
        This will determine the importer that is used to load the data.
    white_list_file : Optional[Path], optional
        The path to the white lists if present.
    black_list_file : Optional[Path], optional
        The path to the black lists if present.
    ignore_list_file : Optional[Path], optional
        The path to the ignore lists if present.
        A csv file with the following columns: grid_model_id, reason.
        The implementation is expected to ignore all elements that are in the ignore list.
    ingress_id : Optional[str], optional
        An optional id that is used to identify the source of the data.
        This can be used to track where the data came from, e.g. if it was imported from a specific
        database or a specific user.
    contingency_list_file : Optional[Path], optional
        The path to the contingency lists if present.
        Expected format see: importer/contingency_from_power_factory/PF_data_class.py.
    schema_format : Optional[Literal["ContingencyImportSchemaPowerFactory", "ContingencyImportSchema"]], optional
        The schema format of the contingency list file if present.
        This can be either "ContingencyImportSchemaPowerFactory" or "ContingencyImportSchema".
        Found in:
        - importer/contingency_from_power_factory/PF_data_class.py
        - importer/pypowsybl_import/contingency_from_file/contingency_file_models.py

    Returns
    -------
    BaseImporterParams
        The base importer parameters with all specified fields.
    """
    params = BaseImporterParameters()
    params.area_settings.CopyFrom(area_settings)
    params.data_folder = str(data_folder)
    params.grid_model_file = str(grid_model_file)
    params.data_type = data_type
    params.white_list_file = str(white_list_file)
    params.black_list_file = str(black_list_file)
    params.ignore_list_file = str(ignore_list_file)
    if ingress_id is not None:
        params.ingress_id = ingress_id
    if contingency_list_file is not None:
        params.contingency_list_file = str(contingency_list_file)
    if schema_format is not None:
        params.schema_format = schema_format
    return params


def create_default_importer_params(
    data_folder: Path, grid_model_file: Path, cgmes: bool = False, ucte: bool = False
) -> BaseImporterParameters:
    """
    Generate default importer parameters based on the specified data type (CGMES or UCTE).

    Parameters
    ----------
    cgmes : bool
        If True, use CGMES data type and related area settings.
    ucte : bool
        If True, use UCTE data type and related area settings.
    data_folder : Path
        Path to the folder containing input data files.
    grid_model_file : Path
        Path to the grid model file.

    Returns
    -------
    params : dict
        Dictionary containing importer parameters for the selected data type.

    """
    if cgmes:
        area_params = create_area_settings(
            control_area=["BE"],
            view_area=["BE", "LU", "D4", "D2", "NL", "FR"],
            nminus1_area=["BE"],
            cutoff_voltage=220,
        )
        params = create_importer_params(
            area_settings=area_params,
            data_folder=data_folder,
            grid_model_file=grid_model_file,
            data_type="cgmes",
        )

    elif ucte:
        area_params = create_area_settings(
            control_area=["D8"], view_area=["D2", "D4", "D7", "D8"], nminus1_area=["D2", "D4", "D7", "D8"]
        )
        params = create_importer_params(
            area_settings=area_params,
            data_folder=data_folder,
            grid_model_file=grid_model_file,
            data_type="ucte",
        )
    else:
        raise ValueError("Either cgmes or ucte must be True")
    return params


def create_preprocess_parameters(  # noqa: PLR0913
    filter_disconnectable_branches_processes: int = 1,
    action_set_filter_bridge_lookup: bool = True,
    action_set_filter_bsdf_lodf: bool = True,
    action_set_filter_bsdf_lodf_batch_size: int = 8,
    action_set_clip: int = 2**20,
    asset_topo_close_couplers: bool = False,
    realise_station_busbar_choice_heuristic: RealiseStationBusbarChoiceHeuristic = "least_connected_busbar",
    ac_dc_interpolation: float = 0.0,
    enable_n_2: bool = False,
    n_2_more_splits_penalty: float = 2000.0,
    enable_bb_outage: bool = False,
    bb_outage_as_nminus1: bool = True,
    bb_outage_more_splits_penalty: float = 50.0,
    clip_bb_outage_penalty: bool = False,
    double_limit_n0: float = 0.9,
    double_limit_n1: float = 0.9,
    initial_loadflow_processes: int = 8,
) -> PreprocessParameters:
    """
    Create and return a PreprocessParameters object with default values.

    Parameters
    ----------
    filter_disconnectable_branches_processes : int, optional
        Number of worker processes to use when checking for disconnectable branches,
        as this is a costly operation. Default is 1.
    action_set_filter_bridge_lookup : bool, optional
        Whether to filter the action set using bridge lookups, removing assignments
        with less than two non-bridges on every side. Default is True.
    action_set_filter_bsdf_lodf : bool, optional
        Whether to filter the action set using a consecutive BSDF/LODF application.
        Filters out actions that split the grid under N-1 branch outages.
        This is a costly process. Default is True.
    action_set_filter_bsdf_lodf_batch_size : int, optional
        Batch size to use when filtering with BSDF/LODF. Larger batch sizes use more
        memory but are faster. Default is 8.
    action_set_clip : int, optional
        Maximum number of actions at a substation before randomly subselecting.
        Prevents exponential explosion of the action space. Default is 2**20.
    asset_topo_close_couplers : bool, optional
        Whether to close open couplers in all stations in the asset topology.
        May cancel a maintenance accidentally. Default is False.
    realise_station_busbar_choice_heuristic : str, optional
        Heuristic for choosing a busbar when multiple are available for an asset.
        Options are "first" or "least_connected_busbar". Default is "least_connected_busbar".
    ac_dc_interpolation : float, optional
        Interpolation between DC loadflow (0) and AC loadflow (1).
        Can be any value in between. Default is 0.0.
    enable_n_2 : bool, optional
        Whether to enable N-2 analysis. Default is False.
    n_2_more_splits_penalty : float, optional
        Penalty for additional splits in N-2 that were not present in the unsplit grid.
        Added to the overload energy penalty. Default is 2000.0.
    enable_bb_outage : bool, optional
        Whether to enable busbar outage analysis. Default is False.
    bb_outage_as_nminus1 : bool, optional
        Whether to treat busbar outages as N-1 outages. If False, treated similar to N-2 outages.
        Used to compute busbar outage penalty. Default is True.
    bb_outage_more_splits_penalty : float, optional
        Penalty for additional splits in busbar outages not present in the unsplit grid.
        Added to the overload energy penalty. Default is 50.0.
    clip_bb_outage_penalty : bool, optional
        Whether to clip the lower bound of the busbar outage penalty to 0.
        Set to False to solve busbar outage problems, True to avoid exacerbating them.
        Default is False.
    double_limit_n0 : float, optional
        If set, double limits are computed for N-0 flows. Lines below this relative load
        have their capacity multiplied by this value to prevent full loading. Default is 0.9.
    double_limit_n1 : float, optional
        If set, double limits are computed for N-1 flows. Lines below this relative load
        have their capacity multiplied by this value to prevent full loading. Default is 0.9.
    initial_loadflow_processes : int, optional
        Number of processes to use for computing the initial AC loadflow. Default is 8.

    Returns
    -------
    PreprocessParameters
        The preprocessing parameters with default values.
    """
    return PreprocessParameters(
        filter_disconnectable_branches_processes=filter_disconnectable_branches_processes,
        action_set_filter_bridge_lookup=action_set_filter_bridge_lookup,
        action_set_filter_bsdf_lodf=action_set_filter_bsdf_lodf,
        action_set_filter_bsdf_lodf_batch_size=action_set_filter_bsdf_lodf_batch_size,
        action_set_clip=action_set_clip,
        asset_topo_close_couplers=asset_topo_close_couplers,
        realise_station_busbar_choice_heuristic=realise_station_busbar_choice_heuristic,
        ac_dc_interpolation=ac_dc_interpolation,
        enable_n_2=enable_n_2,
        n_2_more_splits_penalty=n_2_more_splits_penalty,
        enable_bb_outage=enable_bb_outage,
        bb_outage_as_nminus1=bb_outage_as_nminus1,
        bb_outage_more_splits_penalty=bb_outage_more_splits_penalty,
        clip_bb_outage_penalty=clip_bb_outage_penalty,
        double_limit_n0=double_limit_n0,
        double_limit_n1=double_limit_n1,
        initial_loadflow_processes=initial_loadflow_processes,
    )


def create_start_preprocessing_command(
    importer_parameters: BaseImporterParameters,
    preprocess_parameters: PreprocessParameters,
    preprocess_id: str,
) -> StartPreprocessingCommand:
    """
    Create a StartPreprocessingCommand with the given importer parameters and preprocessing parameters.

    Parameters
    ----------
    importer_parameters : BaseImporterParameters
        The parameters for the data importer.
    preprocess_parameters : PreprocessParameters
        The parameters for the preprocessing.
    preprocess_id : str
        The unique identifier for the preprocessing task.

    Returns
    -------
    StartPreprocessingCommand
        The command to start preprocessing with the specified parameters.
    """
    command = StartPreprocessingCommand()
    command.importer_parameters.CopyFrom(importer_parameters)
    command.preprocess_parameters.CopyFrom(preprocess_parameters)
    command.preprocess_id = preprocess_id
    return command


def create_shutdown_command(exit_code: Optional[int] = 0) -> ShutdownCommand:
    """
    Create a ShutdownCommand to signal the shutdown of the preprocessing engine.

    Returns
    -------
    ShutdownCommand
        The command to shut down the preprocessing engine.
    """
    command = ShutdownCommand()
    command.exit_code = exit_code
    return command


def create_command_wrapper(command: Union[StartPreprocessingCommand, ShutdownCommand]) -> Command:
    """
    Wrap a preprocessing command in a PreprocessCommandWrapper.

    Parameters
    ----------
    command : Union[StartPreprocessingCommand, ShutdownCommand]
        The preprocessing command to wrap.

    Returns
    -------
    PreprocessCommandWrapper
        The wrapped preprocessing command.
    """
    command_wrapper = Command()
    if isinstance(command, StartPreprocessingCommand):
        command_wrapper.start_preprocessing.CopyFrom(command)
    elif isinstance(command, ShutdownCommand):
        command_wrapper.shutdown.CopyFrom(command)
    else:
        raise ValueError("Unsupported command type for wrapping.")
    ts = Timestamp()
    ts.FromDatetime(datetime.now())
    command_wrapper.timestamp.CopyFrom(ts)
    command_wrapper.uuid = str(uuid.uuid4())
    return command_wrapper
