# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines the commands that can be sent to a preprocessing worker."""

import uuid
from datetime import datetime
from pathlib import Path

from beartype.typing import Final, Literal, Optional, TypeAlias, Union
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

# deactivate formatting for the region type definitions
# fmt: off
# As defined in
# https://eepublicdownloads.entsoe.eu/clean-documents/Publications/SOC/Continental_Europe/
# 150420_quality_of_datasets_and_calculations_3rd_edition.pdf
UCTERegionType: TypeAlias = Literal[
        "A","B","C","D","D1","D2","D4","D6","D7","D8","E","F","G","H","I","J","K","L",
        "M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","2","_"
    ]

# As defined by CGMES: ISO 3166
CGMESRegionType: TypeAlias = Literal[
    "AL","AD","AM","AT","AZ","BY","BE","BA","BG","HR","CY","CZ","DK","EE","FI","FR",
    "GE","DE","GR","HU","IS","IE","IT","LV","LI","LT","LU","MT","MD","MC","ME","NL",
    "MK","NO","PL","PT","RO","RU","SM","RS","SK","SI","ES","SE","CH","TR","UA","GB","VA"
    ]

# fmt: on

# Use the Empty string to select all regions
AllCountriesRegionType: TypeAlias = Literal[""]

RegionType: TypeAlias = Union[UCTERegionType, CGMESRegionType, AllCountriesRegionType]
GridModelType: TypeAlias = Literal["ucte", "cgmes"]


class LimitAdjustmentParameters(BaseModel):
    """Parameters for the adjustment of limits of special branches.

    The new operational limits will be created like this:
    1) Compute AC Loadflow
    2) new_limit = current_flow * n_0_factor (or n_1_factor)
    3) If new_limit > min_increase (=old_limit * n_0_min_increase)
        new_limit = min_increase
    4) If new_limit > old_limit
        new_limit = old_limit
    """

    n_0_factor: PositiveFloat = 1.2
    """The factor for the N-0 current limit. Default is 1.2"""

    n_1_factor: PositiveFloat = 1.4
    """The factor for the N-1 current limit. Default is 1.4"""

    n_0_min_increase: PositiveFloat = 0.05
    """The minimal allowed load increase in percent for the n-0 case. This value multiplied with the current limit
    gives a lower border for the new limit. This makes sure, that lines that currently are barely loaded
    can be used at all Default is 5%."""
    n_1_min_increase: PositiveFloat = 0.05
    """The minimal allowed load increase in percent for the n-1 case. This value multiplied with the current limit
    gives a lower border for the new limit. This makes sure, that lines that currently are barely loaded
    can be used at all Default is 5%."""

    def get_parameters_for_case(self, case: Literal["n0", "n1"]) -> tuple[PositiveFloat, PositiveFloat]:
        """Get the factors for the specific case

        Parameters
        ----------
        case: Literal["n0", "n1"]
            Which case should be returned

        Returns
        -------
        tuple[PositiveFloat, PositiveFloat]
            The factor and the minimal increase
        """
        if case == "n0":
            return self.n_0_factor, self.n_0_min_increase
        if case == "n1":
            return self.n_1_factor, self.n_1_min_increase
        raise ValueError(f"Case {case} not defined")


class AreaSettings(BaseModel):
    """Setting related to the areas that are imported"""

    control_area: list[RegionType]
    """The area in which switching can take place. Substations from this area will automatically
    become relevant substations (switchable) except they are below the cutoff voltage. Also lines
    in this area will become disconnectable."""

    view_area: list[RegionType]
    """The areas in which branches shall be part of the overload computation, i.e. for which regions
    shall line flows be computed."""

    nminus1_area: list[RegionType]
    """The areas where elements shall be part of the N-1 computation, i.e. which elements to fail."""

    cutoff_voltage: PositiveInt = 220
    """The cutoff voltage under which to ignore equipment. Equipment that doesn't have at least one
    end equal or above this nominal voltage will not be part of the reward/nminus1 computation"""

    dso_trafo_factors: Optional[LimitAdjustmentParameters] = None
    """If given, the N-0 and N-1 flows across the dso trafos in the specied region will be limited
    to the current N-0 flows in the unsplit configuration. For each case (n0 or n1) a new operational limit
    with the name "border_limit_n0"/"border_limit_n1" is added.
    """

    dso_trafo_weight: Optional[PositiveFloat] = 1.0
    """ A weight that is used for trafos that leave the n-1 area, to underlying DSOs"""

    border_line_factors: Optional[LimitAdjustmentParameters] = None
    """If given, the N-0 and N-1 flows across the border lines leaving or entering the specied region will be limited
    to the current N-0 flows in the unsplit configuration. For each case (n0 or n1) a new operational limit
    with the name "border_limit_n0"/"border_limit_n1" is added.
    """

    border_line_weight: Optional[PositiveFloat] = 1.0
    """ A weight that is used for lines that leave the n-1 area, to neighbouring TSOs"""


class RelevantStationRules(BaseModel):
    """Rules to determine whether a substation is relevant or not."""

    min_busbars: PositiveInt = 2
    """The minimum number of busbars a substation must have to be considered relevant."""

    min_connected_branches: PositiveInt = 4
    """The minimum number of connected branches a substation must have to be considered relevant.
    This only counts branches (lines, transformers, tie-lines), not injections (generators, loads, shunts, etc.)."""

    min_connected_elements: PositiveInt = 4
    """The minimum number of connected elements a substation must have to be considered relevant.
    This includes branches and injections (generators, loads, shunts, etc.)."""


class BaseImporterParameters(BaseModel):
    """Parameters that are required to import any data format."""

    area_settings: AreaSettings
    """Which areas of the grid are to be imported and how to handle boundaries"""

    data_folder: Path
    """The path where the entry point where the timestep data folder structure starts.

    The folder structure is defined in interfaces.folder_structure. This folder is relative to the
    processed_grid_folder that is configured in the backend/importer. A typical default would be grid_model_file.stem"""

    grid_model_file: Path
    """The path to the input grid model file.

    This file should contain the grid model in the format defined by the data_type.
    For instance a .uct for UCTE data or a .zip for CGMES data.
    """

    data_type: Final[GridModelType]
    """The type of data that is being imported.

    This will determine the importer that is used to load the data."""

    white_list_file: Optional[Path] = None
    """The path to the white lists if present"""

    black_list_file: Optional[Path] = None
    """The path to the balck lists if present"""

    ignore_list_file: Optional[Path] = None
    """The path to the ignore lists if present

    A csv file with the following columns:
    grid_model_id, reason

    The implementation is expected to ignore all elements that are in the ignore list.
    """

    select_by_voltage_level_id_list: Optional[list[str]] = None
    """If given, only the voltage levels in this list will be imported.
    Note: not all voltage levels in this list might be considered relevant after preprocessing.
    This can happen if the requirements for relevant substations are not met.
    E.g. minimum number of busbars, connected branches or missing busbar couplers.
    """

    ingress_id: Optional[str] = None
    """An optional id that is used to identify the source of the data.
    This can be used to track where the data came from, e.g. if it was imported from a specific
    database or a specific user.
    """

    contingency_list_file: Optional[Path] = None
    """The path to the contingency lists if present
    expected format see:
    importer/contingency_from_power_factory/PF_data_class.py
    """

    schema_format: Optional[Literal["ContingencyImportSchemaPowerFactory", "ContingencyImportSchema"]] = None
    """The schema format of the contingency list file if present.
    This can be either "ContingencyImportSchemaPowerFactory" or "ContingencyImportSchema".
    found in:
    - importer/contingency_from_power_factory/PF_data_class.py
    - importer/pypowsybl_import/contingency_from_file/contingency_file_models.py
    """

    relevant_station_rules: RelevantStationRules = RelevantStationRules()
    """Rules to determine whether a substation is relevant or not."""


class UcteImporterParameters(BaseImporterParameters):
    """Parameters that are required to import the data from a UCTE file.

    This will utilize powsybl and the powsybl backend to the loadflow solver
    """

    area_settings: AreaSettings = AreaSettings(
        control_area=["D8"],
        view_area=["D2", "D4", "D7", "D8"],
        nminus1_area=["D2", "D4", "D7", "D8"],
    )
    """By default the D8 is controllable and the german grid is viewable"""

    grid_model_file: Path
    """The path to the UCTE file to load. Note that only a single timestep, i.e. only a single UCTE
    file will be loaded in one import/preprocessing run. For multiple timesteps, the preprocessing
    is triggered multiple times."""

    data_type: Literal["ucte"] = "ucte"
    """A constant field to indicate that this is a UCTE importer"""


class CgmesImporterParameters(BaseImporterParameters):
    """Parameters to start an import data from a CGMES file.

    This will utilize powsybl and the powsybl backend to the loadflow solver.
    """

    area_settings: AreaSettings = AreaSettings(
        control_area=["BE"],
        view_area=["BE", "LU", "D4", "D2", "NL", "FR"],
        nminus1_area=["BE"],
        cutoff_voltage=220,
    )
    """The area settings for the CGMES importer"""

    grid_model_file: Path
    """The path to the CGMES .zip file to load.

    Note that only a single timestep, i.e. only a single
    CGMES .zip file will be loaded in one import/preprocessing run. For multiple timesteps, the
    preprocessing is triggered multiple times.
    Note: the .zip file must contain all xml files in the same root folder, i.e. the following files:
    - EQ.xml
    - SSH.xml
    - SV.xml
    - TP.xml
    - EQBD.xml
    - TPBD.xml
    """

    data_type: Literal["cgmes"] = "cgmes"
    """A constant field to indicate that this is a CGMES importer"""


class PreprocessParameters(BaseModel):
    """Parameters for the preprocessing procedure which is independent of the data source"""

    # ---- Parameters for preprocess() -----
    filter_disconnectable_branches_processes: int = 1
    """When checking for disconnectable branches, multiple worker processes can be used as it is a costly operation."""

    action_set_filter_bridge_lookup: bool = True
    """Whether to filter the action set using bridge lookups. This will remove all assignments that have less than
    two non-bridges on every side"""

    action_set_filter_bsdf_lodf: bool = True
    """Whether to filter the action set using a consecutive BSDF/LODF application. This will filter out all actions that
    are also filtered by bridge lookup and additionally all actions that split the grid under N-1 branch outages, i.e. all
    assignments that created a new bridge in the graph. This is a relatively costly process to run, only set to true
    if you can afford the extra preprocessing time."""

    action_set_filter_bsdf_lodf_batch_size: int = 8
    """If filtering with bsdf/lodf - which batch size to use. Larger will use more memory but be faster."""

    action_set_clip: int = 2**20
    """After which size to randomly subselect actions at a substation. If a substations has a lot of branches, the action
    space will explode exponentially and a safe-guard is to clip after a certain number of actions."""

    asset_topo_close_couplers: bool = False
    """Whether to close open couplers in all stations in the asset topology. This might accidentally cancel a maintenance"""

    realise_station_busbar_choice_heuristic: Literal["first", "least_connected_busbar"] = "least_connected_busbar"
    """The heuristic to use when there are multiple physical busbars available for an asset. The options are:
    - "first": Use the first busbar in the list of busbars (fastest preprocessing)
    - "least_connected_busbar": Use the busbar with the least number of connections to other assets (best results)

    The "least_connected_busbar" heuristic is the default and is recommended for most cases, trying to spread the assets
    evenly across the busbars in a station.
    """

    # ---- Parameters for convert_to_jax ------

    ac_dc_interpolation: float = 0.0
    """Whether to use the DC loadflow as the base loadflow (0) or the AC loadflow (1). Can also be anything in between."""

    enable_n_2: bool = False
    """Whether to enable N-2 analysis"""

    n_2_more_splits_penalty: float = 2000.0
    """How to penalize additional splits in N-2 that were not there in the unsplit grid. Will be
    added to the overload energy penalty."""

    enable_bb_outage: bool = False
    """Whether to enable busbar outage analysis"""

    bb_outage_as_nminus1: bool = True
    """Whether to treat busbar outages as N-1 outages. If set to False, the busbar outage will be treated similar to
    N-2 outages. This will be used to compute the busbar outage penalty."""

    bb_outage_more_splits_penalty: float = 50.0
    """How to penalize additional splits in busbar outages that were not there in the unsplit grid. Will be
    added to the overload energy penalty."""

    # TODO: MOve this parameter to optimiser configs
    clip_bb_outage_penalty: bool = False
    """
    Whether to clip the lower bound of the busbar outage penalty to 0.
    We set this parameter to False, if we want the optimiser to solve busbar outage problems in the grid. However,
    when we just want to ensure that the busbar outage problems are not exacerbated due to the optimiser, we set
    this to True."""

    # ---- Parameters for the initial loadflow -----
    double_limit_n0: Optional[PositiveFloat] = 0.9
    """If passed, then double limits will be computed for the N-0 flows. Lines that are below
    double_limit_n0 relative load in the unsplit configuration will have their capacity multiplied
    by double_limit_n0 to prevent loading them up to their maximum capacity."""

    double_limit_n1: Optional[PositiveFloat] = 0.9
    """If passed, then double limits will be computed for the N-1 flows. Lines that are below
    double_limit_n1 relative load in the unsplit configuration will have their capacities multiplied
    by double_limit_n1 to prevent loading them up to their maximum capacity."""

    initial_loadflow_processes: int = 8
    """How many processes to use to compute the initial AC loadflow"""


class StartPreprocessingCommand(BaseModel):
    """A command to launch a preprocessing run of a timestep upon reception."""

    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters] = Field(discriminator="data_type")
    """The parameters to the importer, depending which input source was chosen"""

    preprocess_parameters: PreprocessParameters = PreprocessParameters()
    """Parameters required for preprocessing independent of the data source"""

    preprocess_id: str
    """The id of the preprocessing run, should be included in all responses to identify where
    the data came from"""


class ShutdownCommand(BaseModel):
    """A command to shut down the preprocessing worker"""

    exit_code: Optional[int] = 0
    """The exit code to return"""


class Command(BaseModel):
    """A wrapper to aid deserialization"""

    command: Union[StartPreprocessingCommand, ShutdownCommand]
    """The actual command posted"""

    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
    """When the command was sent"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for this command message, used to avoid duplicates during processing"""
