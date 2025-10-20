"""Module contains functions for the pypowsybl preprocessing for the grid export into the loadflow solver.

File: preprocessing.py
Author:  Benjamin Petrick
Created: 2024-09-04
"""

import json
import shutil
from pathlib import Path
from typing import (
    Any,  # noqa: F401
    Callable,
    Optional,
    Union,
)

import logbook
import pypowsybl
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.loadflow_parameters import (
    DISTRIBUTED_SLACK,
)
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import get_topology
from toop_engine_importer.network_graph import powsybl_station_to_graph
from toop_engine_importer.pypowsybl_import import network_analysis, powsybl_masks
from toop_engine_importer.pypowsybl_import.data_classes import PreProcessingStatistics
from toop_engine_importer.pypowsybl_import.loadflow_based_current_limits import (
    create_new_border_limits,
)
from toop_engine_importer.pypowsybl_import.powsybl_masks import NetworkMasks
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    CgmesImporterParameters,
    UcteImporterParameters,
)
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import (
    PreprocessStage,
    empty_status_update_fn,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    UcteImportResult,
)
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition, save_nminus1_definition

logger = logbook.Logger(__name__)


def save_preprocessing_statistics(statistics: PreProcessingStatistics, output_file: Path) -> None:
    """Save the preprocessing statistics to the output folder.

    Parameters
    ----------
    statistics: PreprocessingStatistics
        The statistics to save.
    output_file: Path
        The file to save the preprocessing statistics to.


    """
    with open(output_file, "w") as f:
        f.write(statistics.model_dump_json(indent=4))


def load_preprocessing_statistics(file: Path) -> PreProcessingStatistics:
    """Load the preprocessing statistics from the file.

    Parameters
    ----------
    file: Path
        The file to load the preprocessing statistics from.

    Returns
    -------
    statistics: PreprocessingStatistics
        The loaded statistics.

    """
    with open(file, "r") as f:
        statistics = json.load(f)
    import_result = PreProcessingStatistics(**statistics)
    return import_result


def create_nminus1_definition_from_masks(network: Network, network_masks: NetworkMasks) -> Nminus1Definition:
    """Create the N-1 definition from the network masks.

    Parameters
    ----------
    network: Network
        The network to create the N-1 definition for.
    network_masks: NetworkMasks
        The network masks to create the N-1 definition from.

    Returns
    -------
    Nminus1Definition
        The created N-1 definition.
    """
    contingencies = [Contingency(id="BASECASE", name="BASECASE", elements=[])]

    lines = network.get_lines(attributes=["name"])
    monitored_lines = [
        GridElement(id=idx, name=row["name"], type="LINE", kind="branch")
        for idx, row in lines[network_masks.line_for_reward].iterrows()
    ]
    outaged_lines = [
        Contingency(id=idx, name=row["name"], elements=[GridElement(id=idx, name=row["name"], type="LINE", kind="branch")])
        for idx, row in lines[network_masks.line_for_nminus1].iterrows()
    ]

    trafos = network.get_2_windings_transformers(attributes=["name"])
    is_trafo2w = ~trafos.index.str.contains("-Leg[123]$")
    monitored_trafos = [
        GridElement(id=idx, name=row["name"], type="TWO_WINDINGS_TRANSFORMER", kind="branch")
        for idx, row in trafos[is_trafo2w & network_masks.trafo_for_reward].iterrows()
    ]
    outaged_trafos = [
        Contingency(
            id=idx,
            name=row["name"],
            elements=[GridElement(id=idx, name=row["name"], type="TWO_WINDINGS_TRANSFORMER", kind="branch")],
        )
        for idx, row in trafos[is_trafo2w & network_masks.trafo_for_nminus1].iterrows()
    ]

    is_trafo3w = trafos.index.str.contains("-Leg[123]$")
    trafos.index = trafos.index.str.replace("-Leg[123]$", "", regex=True)
    if not trafos.empty:
        trafos.name = trafos.name.str.replace("-Leg[123]$", "", regex=True)

    monitored_trafo3w = [
        GridElement(id=idx, name=row["name"], type="THREE_WINDINGS_TRANSFORMER", kind="branch")
        for idx, row in trafos[is_trafo3w & network_masks.trafo_for_reward].drop_duplicates().iterrows()
    ]
    outaged_trafo3w = [
        Contingency(
            id=idx,
            name=row["name"],
            elements=[GridElement(id=idx, name=row["name"], type="THREE_WINDINGS_TRANSFORMER", kind="branch")],
        )
        for idx, row in trafos[is_trafo3w & network_masks.trafo_for_nminus1].drop_duplicates().iterrows()
    ]

    tie_lines = network.get_tie_lines(attributes=["name"])
    monitored_tie_lines = [
        GridElement(id=idx, name=row["name"], type="TIE_LINE", kind="branch")
        for idx, row in tie_lines[network_masks.tie_line_for_reward].iterrows()
    ]
    outaged_tie_lines = [
        Contingency(
            id=idx, name=row["name"], elements=[GridElement(id=idx, name=row["name"], type="TIE_LINE", kind="branch")]
        )
        for idx, row in tie_lines[network_masks.tie_line_for_nminus1].iterrows()
    ]

    dangling_lines = network.get_dangling_lines(attributes=["name", "paired"])
    outaged_dangling = [
        Contingency(
            id=idx,
            name=row["name"],
            elements=[
                GridElement(id=idx, name=row["name"], type="DANGLING_LINE", kind="injection"),
            ],
        )
        for idx, row in dangling_lines[network_masks.dangling_line_for_nminus1 & ~dangling_lines["paired"]].iterrows()
    ]

    generators = network.get_generators(attributes=["name"])
    outaged_generators = [
        Contingency(
            id=idx, name=row["name"], elements=[GridElement(id=idx, name=row["name"], type="GENERATOR", kind="injection")]
        )
        for idx, row in generators[network_masks.generator_for_nminus1].iterrows()
    ]

    loads = network.get_loads(attributes=["name"])
    outaged_loads = [
        Contingency(
            id=idx, name=row["name"], elements=[GridElement(id=idx, name=row["name"], type="LOAD", kind="injection")]
        )
        for idx, row in loads[network_masks.load_for_nminus1].iterrows()
    ]

    switches = network.get_switches(attributes=["name"])
    monitored_switches = [
        GridElement(id=idx, name=row["name"], type="SWITCH", kind="branch")
        for idx, row in switches[network_masks.switch_for_reward].iterrows()
    ]
    outaged_switches = [
        Contingency(id=idx, name=row["name"], elements=[GridElement(id=idx, name=row["name"], type="SWITCH", kind="branch")])
        for idx, row in switches[network_masks.switch_for_nminus1].iterrows()
    ]

    buses = network.get_buses()
    relevant_bus_ids = buses.index[network_masks.relevant_subs].to_list()
    busbar_sections = network.get_busbar_sections(attributes=["name", "bus_id"])
    monitored_busbars = [
        GridElement(id=idx, name=row["name"], type="BUSBAR_SECTION", kind="bus")
        for idx, row in busbar_sections[busbar_sections.index.isin(relevant_bus_ids)].iterrows()
    ]
    busbreaker_buses = network.get_bus_breaker_view_buses(attributes=["name", "bus_id"])
    monitored_busbreakers = [
        GridElement(id=idx, name=row["name"], type="BUS_BREAKER_BUS", kind="bus")
        for idx, row in busbreaker_buses[busbreaker_buses.index.isin(relevant_bus_ids)].iterrows()
    ]

    nminus1_definition = Nminus1Definition(
        monitored_elements=(
            monitored_lines
            + monitored_trafos
            + monitored_trafo3w
            + monitored_tie_lines
            + monitored_switches
            + monitored_busbars
            + monitored_busbreakers
        ),
        contingencies=(
            contingencies
            + outaged_lines
            + outaged_trafos
            + outaged_trafo3w
            + outaged_tie_lines
            + outaged_dangling
            + outaged_generators
            + outaged_loads
            + outaged_switches
        ),
    )
    return nminus1_definition


def convert_file(
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
    status_update_fn: Callable[[PreprocessStage, Optional[str]], None] = empty_status_update_fn,
) -> UcteImportResult:
    """Convert the UCTE file to a format that can be used by the RL agent.

    Saves data and network to the output folder.

    Parameters
    ----------
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        Parameters that are required to import the data from a UCTE file. This will utilize
        powsybl and the powsybl backend to the loadflow solver
    status_update_fn: Callable[[PreprocessStage, Optional[str]]
        A function to call to signal progress in the preprocessing pipeline. Takes a stage and an
        optional message as parameters

    Returns
    -------
    UcteImportResult
        The result of the import process.

    """
    # Copy original grid file
    (importer_parameters.data_folder / PREPROCESSING_PATHS["original_gridfile_path"]).mkdir(exist_ok=True, parents=True)
    original_gridfile_path = importer_parameters.data_folder / PREPROCESSING_PATHS["original_gridfile_path"]
    # Copy original grid file
    shutil.copy(importer_parameters.grid_model_file, original_gridfile_path / importer_parameters.grid_model_file.name)

    # load network
    status_update_fn("load_ucte", "start loading grid file")
    network = pypowsybl.network.load(
        importer_parameters.grid_model_file, {"iidm.import.cgmes.post-processors": "cgmesGLImport"}
    )
    # Save original grid file as powsybl grid file
    network.save(original_gridfile_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    network_analysis.remove_branches_with_same_bus(network)
    status_update_fn("load_ucte", "done loading grid file")

    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(network)
    if pypowsybl.__version__ <= "1.12.0":
        # Fix the bug, where the operational limits of the 2winding transformers are not set correctly
        op_lim = network.get_operational_limits(all_attributes=True, show_inactive_sets=True)
        trafo3w_lims = op_lim[op_lim.index.str.contains("-Leg")][["group_name"]].rename(
            columns={"group_name": "selected_limits_group_1"}
        )
        trafo3w_lims.index.name = "id"
        network.update_2_windings_transformers(trafo3w_lims)
    statistics = PreProcessingStatistics(
        import_result=UcteImportResult(data_folder=importer_parameters.data_folder),
        import_parameter=importer_parameters,
    )
    status_update_fn("apply_cb_list", "Applying Whitelists")
    if importer_parameters.data_type == "ucte":
        # TODO: move to UCTE Toolset after all PRs are merged
        apply_preprocessing_changes_to_network(
            network=network,
            statistics=statistics,
            status_update_fn=status_update_fn,
        )

        # apply black and white list
        statistics = network_analysis.apply_cb_lists(
            network=network,
            statistics=statistics,
            ucte_importer_parameters=importer_parameters,
        )
    elif importer_parameters.data_type == "cgmes":
        statistics.id_lists["white_list"] = []
        statistics.id_lists["black_list"] = []
        logger.warning("CGMES of white_list and black_list not yet implemented")
    grid_file_path = importer_parameters.data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(exist_ok=True, parents=True)
    network.save(grid_file_path)
    # Reload Network because powsybl likes to change order during save
    network = pypowsybl.network.load(grid_file_path)
    # get N-1 masks
    status_update_fn("get_masks", "Creating Network Masks")
    network_masks = create_network_masks(network, importer_parameters, statistics)
    nminus1_definition = create_nminus1_definition_from_masks(network, network_masks)
    save_nminus1_definition(original_gridfile_path / PREPROCESSING_PATHS["nminus1_definition_file_path"], nminus1_definition)
    if (
        importer_parameters.area_settings.dso_trafo_factors is not None
        or importer_parameters.area_settings.border_line_factors is not None
    ):
        status_update_fn("cross_border_current", "Setting cross border current limit")
        lf_result = pypowsybl.loadflow.run_ac(network, parameters=DISTRIBUTED_SLACK)
        if lf_result[0].status != pypowsybl.loadflow.ComponentStatus.CONVERGED:
            pypowsybl.loadflow.run_dc(network, parameters=DISTRIBUTED_SLACK)
        create_new_border_limits(network, network_masks, importer_parameters)

    save_preprocessing_statistics(
        statistics,
        importer_parameters.data_folder / PREPROCESSING_PATHS["importer_auxiliary_file_path"],
    )

    status_update_fn("get_topology_model", "Creating Pydantic Topology Model")
    create_topology_model(network, network_masks, importer_parameters)

    return statistics.import_result


def create_network_masks(
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
    statistics: PreProcessingStatistics,
) -> NetworkMasks:
    """Create network masks and save them.

    Parameters
    ----------
    network: Network
        The network to create the asset topology for
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        import parameters that include the datafolder
    statistics: PreProcessingStatistics
        preprocessing statistics to fill with information

    Returns
    -------
    NetworkMasks
        The created network masks
    """
    network_masks = powsybl_masks.make_masks(
        network=network,
        importer_parameters=importer_parameters,
        blacklisted_ids=statistics.id_lists["black_list"],
    )
    fill_statistics_for_network_masks(network=network, statistics=statistics, network_masks=network_masks)
    powsybl_masks.save_masks_to_files(data_folder=importer_parameters.data_folder, network_masks=network_masks)
    return network_masks


def create_topology_model(
    network: Network,
    network_masks: NetworkMasks,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
) -> None:
    """Create the initial asset topology.

    Parameters
    ----------
    network: Network
        The network to create the asset topology for
    network_masks: NetworkMasks
        The network masks giving info which elements are relevant
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        import parameters that include the datafolder

    Returns
    -------
    None
    """
    if importer_parameters.data_type == "ucte":
        topology_model = get_topology(
            network,
            relevant_stations=network_masks.relevant_subs,
            topology_id=importer_parameters.grid_model_file.name,
            grid_model_file=str(importer_parameters.grid_model_file),
        )
    elif importer_parameters.data_type == "cgmes":
        topology_model = powsybl_station_to_graph.get_topology(network, network_masks, importer_parameters)
    topology_path = importer_parameters.data_folder / PREPROCESSING_PATHS["asset_topology_file_path"]
    topology_path.parent.mkdir(exist_ok=True, parents=True)
    with open(topology_path, "w") as f:
        f.write(topology_model.model_dump_json(indent=4))


def apply_preprocessing_changes_to_network(
    network: Network,
    statistics: PreProcessingStatistics,
    status_update_fn: Optional[Callable[[PreprocessStage, Optional[str]], None]] = None,
) -> None:
    """Apply the default changes to the network.

    These changes include:
    - removing low impedance lines
    - removing branches across switches

    Parameters
    ----------
    network: Network
        The network to apply the changes to.
        Note: This function modifies the network in place.
    statistics: PreprocessingStatistics
        The statistics of the preprocessing.
        Note: This function modifies the statistics in place.
    status_update_fn: Optional[Callable[[PreprocessStage, Optional[str]], None]]
        A function to call to signal progress in the preprocessing pipeline. Takes a stage and an
        optional message as parameters

    """
    if status_update_fn is None:
        status_update_fn = empty_status_update_fn
    status_update_fn("modify_low_impedance_lines", "Converting low impedance lines to breakers")
    low_impedance_lines = network_analysis.convert_low_impedance_lines(network, "D8")
    statistics.import_result.n_low_impedance_lines = len(low_impedance_lines)
    statistics.network_changes["low_impedance_lines"] = low_impedance_lines.index.to_list()

    status_update_fn("modify_branches_over_switches", "Removing branches across switches")
    branches_across_switch = network_analysis.remove_branches_across_switch(network)
    statistics.import_result.n_branch_across_switch = len(branches_across_switch)
    statistics.network_changes["branches_across_switch"] = branches_across_switch.index.to_list()


def fill_statistics_for_network_masks(
    network: Network, statistics: PreProcessingStatistics, network_masks: NetworkMasks
) -> None:
    """Fill the statistics with the network masks.

    Parameters
    ----------
    network: Network
        The network to get the id lists from.
    statistics: PreprocessingStatistics
        The statistics to fill.
        Note: This function modifies the statistics in place.
    network_masks: NetworkMasks
        The masks for the network.

    """
    statistics.id_lists["relevant_subs"] = network.get_buses(attributes=[])[network_masks.relevant_subs].index.to_list()
    statistics.id_lists["line_for_nminus1"] = network.get_lines(attributes=[])[
        network_masks.line_for_nminus1
    ].index.to_list()
    statistics.id_lists["trafo_for_nminus1"] = network.get_2_windings_transformers(attributes=[])[
        network_masks.trafo_for_nminus1
    ].index.to_list()
    statistics.id_lists["tie_line_for_nminus1"] = network.get_tie_lines(attributes=[])[
        network_masks.tie_line_for_nminus1
    ].index.to_list()
    statistics.id_lists["dangling_line_for_nminus1"] = network.get_dangling_lines(attributes=[])[
        network_masks.dangling_line_for_nminus1
    ].index.to_list()
    statistics.id_lists["generator_for_nminus1"] = network.get_generators(attributes=[])[
        network_masks.generator_for_nminus1
    ].index.to_list()
    statistics.id_lists["load_for_nminus1"] = network.get_loads(attributes=[])[
        network_masks.load_for_nminus1
    ].index.to_list()
    statistics.id_lists["switch_for_nminus1"] = network.get_switches(attributes=[])[
        network_masks.switch_for_nminus1
    ].index.to_list()
    statistics.id_lists["line_disconnectable"] = network.get_lines(attributes=[])[
        network_masks.line_disconnectable
    ].index.to_list()

    statistics.import_result.n_relevant_subs = int(network_masks.relevant_subs.sum())
    statistics.import_result.n_line_for_nminus1 = int(network_masks.line_for_nminus1.sum())
    statistics.import_result.n_line_for_reward = int(network_masks.line_for_reward.sum())
    statistics.import_result.n_line_disconnectable = int(network_masks.line_disconnectable.sum())

    statistics.import_result.n_trafo_for_nminus1 = int(network_masks.trafo_for_nminus1.sum())
    statistics.import_result.n_trafo_for_reward = int(network_masks.trafo_for_reward.sum())
    statistics.import_result.n_tie_line_disconnectable = int(network_masks.tie_line_disconnectable.sum())
    statistics.import_result.n_tie_line_for_nminus1 = int(network_masks.tie_line_for_nminus1.sum())
    statistics.import_result.n_tie_line_for_reward = int(network_masks.tie_line_for_reward.sum())
    statistics.import_result.n_tie_line_disconnectable = int(network_masks.tie_line_disconnectable.sum())
    statistics.import_result.n_dangling_line_for_nminus1 = int(network_masks.dangling_line_for_nminus1.sum())
    statistics.import_result.n_generator_for_nminus1 = int(network_masks.generator_for_nminus1.sum())
    statistics.import_result.n_load_for_nminus1 = int(network_masks.load_for_nminus1.sum())
    statistics.import_result.n_switch_for_nminus1 = int(network_masks.switch_for_nminus1.sum())
    statistics.import_result.n_switch_for_reward = int(network_masks.switch_for_reward.sum())
