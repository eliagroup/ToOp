# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides general helper functions for pypowsybl networks.

These functions are used to extract and manipulate data from pypowsybl networks,
such as loadflow results, branch limits, and monitored elements.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandapower
import pandas as pd
import pypowsybl
from beartype.typing import Dict, List, Literal, Optional
from fsspec import AbstractFileSystem
from pypowsybl.network import Network
from pypowsybl.report import ReportNode
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK


def extract_single_injection_loadflow_result(injections: pd.DataFrame, injection_id: str) -> tuple[float, float]:
    """Extract the loadflow results for a single injection.

    Parameters
    ----------
    injections : pd.DataFrame
        The injections dataframe with the loadflow results
    injection_id : str
        The id of the injection to extract

    Returns
    -------
    tuple[float, float]
        The active and reactive power of the injection
    """
    p = injections.loc[injection_id, "p"]
    q = injections.loc[injection_id, "q"]
    return p, q


def extract_single_branch_loadflow_result(
    branches: pd.DataFrame, branch_id: str, from_side: bool = True
) -> tuple[float, float]:
    """Extract the loadflow results for a single branch

    Parameters
    ----------
    branches : pd.DataFrame
        The branches dataframe with the loadflow results
    branch_id : str
        The id of the branch to extract
    from_side : bool, optional
        If True, the from side is used, by default True
        If False, the to side is used

    Returns
    -------
    tuple[float, float]
        The active and reactive power of the branch
    """
    p_mapper = "p1" if from_side else "p2"
    q_mapper = "q1" if from_side else "q2"

    p = branches.loc[branch_id, p_mapper]
    q = branches.loc[branch_id, q_mapper]
    return p, q


def get_branches_with_i(branches: pd.DataFrame, net: Network) -> pd.DataFrame:
    """Get the get_branches results with the i values filled

    In DC, they are Nan and will be computed through p = i * v, in AC they are already computed

    Parameters
    ----------
    branches : pd.DataFrame
        The result of net.get_branches() with or without i values
    net : Network
        The powsybl network to load voltage information from

    Returns
    -------
    pd.DataFrame
        The result of net.get_branches(), but with the i values filled
    """
    if np.any(branches["i1"].notna()):
        return branches
    # DC loadflow, compute i = p / v
    # # P[kW] = sqrt(3) * U[kV] * I[A]
    # P[MW] = P[kW] / 1000
    # I[A] = P[MW] / (1000 * sqrt(3) * U[kV])
    branches = pd.merge(
        left=branches,
        right=net.get_voltage_levels()["nominal_v"].rename("nominal_v_1"),
        left_on="voltage_level1_id",
        right_index=True,
    )
    branches = pd.merge(
        left=branches,
        right=net.get_voltage_levels()["nominal_v"].rename("nominal_v_2"),
        left_on="voltage_level2_id",
        right_index=True,
    )
    branches["i1"] = (branches["p1"] / (branches["nominal_v_1"] * math.sqrt(3) * 1e-3)).abs()
    branches["i2"] = (branches["p2"] / (branches["nominal_v_2"] * math.sqrt(3) * 1e-3)).abs()
    del branches["nominal_v_1"]
    del branches["nominal_v_2"]
    return branches


def get_injections_with_i(injections: pd.DataFrame, net: Network) -> pd.DataFrame:
    """Get the get_injections results with the i values filled

    In DC, they are Nan and will be computed through p = i * v, in AC they are already computed

    Parameters
    ----------
    injections : pd.DataFrame
        The result of net.get_injections() with or without i values
    net : Network
        The powsybl network to load voltage information from

    Returns
    -------
    pd.DataFrame
        The result of net.get_injections(), but with the i values filled
    """
    if np.any(injections["i"].notna()):
        return injections
    # DC loadflow, compute i = p / v
    # # P[kW] = sqrt(3) * U[kV] * I[A]
    # P[MW] = P[kW] / 1000
    # I[A] = P[MW] / (1000 * sqrt(3) * U[kV])
    injections = pd.merge(
        left=injections,
        right=net.get_voltage_levels()["nominal_v"],
        left_on="voltage_level_id",
        right_index=True,
    )
    injections["i"] = (injections["p"] / (injections["nominal_v"] * math.sqrt(3) * 1e-3)).abs()
    del injections["nominal_v"]
    return injections


def get_branches_with_i_max(branches: pd.DataFrame, net: Network, limit_name: str = "permanent_limit") -> pd.DataFrame:
    """Get the get_branches results with the i_max values

    These will be pulled from the current limits table. If one side is missing, the other side will
    be taken. If both sides are missing, the value will be NaN

    Parameters
    ----------
    branches : pd.DataFrame
        The result of net.get_branches() without i_max values
    net : Network
        The powsybl network with current limits
    limit_name : str, optional
        The name of the limit to pull, by default "permanent_limit"

    Returns
    -------
    pd.DataFrame
        The result of net.get_branches(), but with i1_max and i2_max columns
    """
    current_limits = net.get_operational_limits()
    current_limits = current_limits[current_limits["name"] == limit_name]
    current_limits = current_limits.groupby(level=["element_id", "side"]).min().reset_index("side")
    current_limits1 = current_limits[current_limits["side"] == "ONE"]["value"].reindex(current_limits.index)
    current_limits2 = current_limits[current_limits["side"] == "TWO"]["value"].reindex(current_limits.index)

    # Take the other side if one side is missing
    current_limits1 = current_limits1.fillna(current_limits2)
    current_limits2 = current_limits2.fillna(current_limits1)

    branches = pd.merge(
        left=branches,
        right=current_limits1.rename("i1_max"),
        left_index=True,
        right_index=True,
        how="left",
    )
    branches = pd.merge(
        left=branches,
        right=current_limits2.rename("i2_max"),
        left_index=True,
        right_index=True,
        how="left",
    )

    return branches


def get_voltage_level_with_region(
    network: Network, attributes: Optional[list[str]] = None, all_attributes: Optional[Literal[True, False]] = None
) -> pd.DataFrame:
    """Get the region for each voltage level in the network.

    This function is an extension to the network.get_voltage_levels() function.
    It retrieves the region for each voltage level using the substation region.

    Parameters
    ----------
    network: Network
        The network for which the regions should be retrieved.
    attributes: Optional[list[str]]
        The attributes that should be retrieved for the voltage levels.
        Behaves like the attributes parameter in network.get_voltage_levels().
    all_attributes: Optional[Union[True,False]]
        If True, all attributes are retrieved for the voltage levels.
        Behaves like the all_attributes parameter in network.get_voltage_levels().

    Returns
    -------
    pd.DataFrame
        A DataFrame with the voltage levels and their regions.
    """
    substation_region = network.get_substations(attributes=["country"])
    substation_region.rename(columns={"country": "region"}, inplace=True)
    if attributes is not None and all_attributes is not None:
        raise ValueError("Only one of 'attributes' and 'all_attributes' can be specified")
    if ((attributes is None) and (not all_attributes)) or attributes == ["region"]:
        voltage_level = network.get_voltage_levels()
    elif all_attributes:
        voltage_level = network.get_voltage_levels(all_attributes=True)
    elif attributes is not None:
        if "region" in attributes:
            attributes = [attr for attr in attributes if attr != "region"]
        voltage_level = network.get_voltage_levels(attributes=attributes)
    voltage_level = voltage_level.merge(
        substation_region, left_on="substation_id", right_on="id", how="left", suffixes=("", "_substation")
    ).set_index(voltage_level.index)
    if ["region"] == attributes:
        voltage_level = voltage_level[["region"]]
    return voltage_level


def change_dangling_to_tie(dangling_lines: pd.DataFrame, station_elements: pd.DataFrame) -> pd.DataFrame:
    """Change the type of the dangling lines to TIE_LINE.

    Changes dangling lines if a tie line is present.
    Keep Dangling lines if no tie line is present.

    Parameters
    ----------
    dangling_lines: pd.DataFrame
        Dataframe of all dangling lines in the network with column "tie_line_id"
    station_elements: pd.DataFrame
        DataFrame with the injections and branches of the bus

    Returns
    -------
    pd.DataFrame
        A copy of the station_elements dataframe with all dangling lines that are actually part of a tie line set to
        TIE_LINE.
    """
    dangling = station_elements[station_elements["type"] == "DANGLING_LINE"].copy()
    dangling_index = dangling.index
    # if dangling lines are present -> check for tie lines
    if not dangling.empty:
        dangling = dangling.merge(
            dangling_lines,
            left_index=True,
            right_index=True,
        )
        condition_empty = dangling["tie_line_id"] == ""
        dangling.loc[~condition_empty, "type"] = "TIE_LINE"
        dangling.loc[condition_empty, "tie_line_id"] = dangling[condition_empty].index
        dangling.rename(columns={"tie_line_id": "id"}, inplace=True)
        dangling.set_index("id", inplace=True)

        station_elements = station_elements.drop(dangling_index)
        station_elements = pd.concat([station_elements, dangling])

    return station_elements


def load_powsybl_from_fs(
    filesystem: AbstractFileSystem,
    file_path: Path,
    parameters: Optional[Dict[str, str]] = None,
    post_processors: Optional[List[str]] = None,
    report_node: Optional[ReportNode] = None,
    allow_variant_multi_thread_access: bool = False,
) -> pypowsybl.network.Network:
    """Load any supported Powsybl network format from a filesystem.

    Supported standard Powsybl formats like CGMES (.zip), powsybl nativa (.xiddm), UCTE (.uct), Matpower (.mat).
    For all supported formats, see pypowsybl documentation for `pypowsybl.network.load`:
    https://powsybl.readthedocs.io/projects/powsybl-core/en/stable/grid_exchange_formats/index.html

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The filesystem to load the Powsybl network from.
    file_path : Path
        The path to the Powsybl network in the filesystem.
    parameters : Optional[Dict[str, str]], optional
        Additional parameters to pass to the pypowsybl.network.load function, by default None
    post_processors : Optional[List[str]], optional
        a list of import post processors (will be added to the ones defined by the platform config), by default None
    report_node : Optional[ReportNode], optional
        the reporter to be used to create an execution report
    allow_variant_multi_thread_access : bool, optional
        allow multi-thread access to variant, by default False

    Returns
    -------
    pypowsybl.network.Network
        The loaded Powsybl network.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_grid_path = Path(temp_dir) / file_path.name
        filesystem.download(
            str(file_path),
            str(tmp_grid_path),
        )
        return pypowsybl.network.load(
            file=str(tmp_grid_path),
            parameters=parameters,
            post_processors=post_processors,
            report_node=report_node,
            allow_variant_multi_thread_access=allow_variant_multi_thread_access,
        )


def save_lf_params_to_fs(
    lf_params: pypowsybl.loadflow.Parameters | dict, filesystem: AbstractFileSystem, file_path: Path, make_dir: bool = True
) -> None:
    """Save the loadflow parameters to a filesystem.

    Parameters
    ----------
    lf_params : pypowsybl.loadflow.Parameters
        The loadflow parameters to save.
    filesystem : AbstractFileSystem
        The filesystem to save the loadflow parameters to.
    file_path : Path
        The path to save the loadflow parameters in the filesystem.
    make_dir: bool
        create parent folder if not exists.
    """
    if isinstance(lf_params, pypowsybl.loadflow.Parameters):
        json_str = lf_params.to_json()
        # This is a hack, as long as the saving/loading bug is not fixed in pypowsybl:
        # https://github.com/powsybl/pypowsybl/issues/1153
        _json = json.loads(lf_params.to_json())
        _json["provider_parameters"] = lf_params.provider_parameters
        json_str = json.dumps(_json)
    else:
        json_str = json.dumps(lf_params)

    if make_dir:
        filesystem.makedirs(Path(file_path).parent.as_posix(), exist_ok=True)
    # save json string to filesystem
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_grid_path = Path(temp_dir) / file_path.name
        with open(tmp_grid_path, "w") as f:
            f.write(json_str)
        filesystem.upload(
            str(tmp_grid_path),
            str(file_path),
        )


def load_lf_params_from_fs(
    filesystem: AbstractFileSystem,
    file_path: Path,
) -> pypowsybl.loadflow.Parameters | dict:
    """Load the loadflow parameters from a filesystem.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The filesystem to load the loadflow parameters from.
    file_path : Path
        The path to the loadflow parameters in the filesystem.

    Returns
    -------
    pypowsybl.loadflow.Parameters
        The loaded loadflow parameters.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_grid_path = Path(temp_dir) / file_path.name
        filesystem.download(
            str(file_path),
            str(tmp_grid_path),
        )
        with open(tmp_grid_path, "r") as f:
            json_str = f.read()
            _json = json.loads(json_str)
            provider_params = _json.pop("provider_parameters", {})
            json_str = json.dumps(_json)
        params = pypowsybl.loadflow.Parameters.from_json(json_str)
        params.provider_parameters = provider_params
        return params


def save_powsybl_to_fs(
    net: pypowsybl.network.Network,
    filesystem: AbstractFileSystem,
    file_path: Path,
    format: Optional[Literal["CGMES", "XIIDM", "UCTE", "MATPOWER"]] = "XIIDM",
    make_dir: bool = True,
) -> None:
    """Save any supported Powsybl network format to a filesystem.

    Supported standard Powsybl formats like CGMES (.zip), powsybl nativa (.xiddm), UCTE (.uct), Matpower (.mat).
    For all supported formats, see pypowsybl documentation for `pypowsybl.network.save`:
    https://powsybl.readthedocs.io/projects/powsybl-core/en/stable/grid_exchange_formats/index.html

    Parameters
    ----------
    net : pypowsybl.network.Network
        The Powsybl network to save.
    filesystem : AbstractFileSystem
        The filesystem to save the Powsybl network to.
    file_path : Path
        The path to save the Powsybl network in the filesystem.
    format : Optional[Literal["CGMES", "XIIDM", "UCTE", "MATPOWER"]], optional
        The format to save the Powsybl network in, by default "CGMES".
    make_dir: bool
        create parent folder if not exists.
    """
    if make_dir:
        filesystem.makedirs(Path(file_path).parent.as_posix(), exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_grid_path = Path(temp_dir) / file_path.name
        net.save(str(tmp_grid_path), format=format)
        filesystem.upload(
            str(tmp_grid_path),
            str(file_path),
        )


def select_a_generator_as_slack_and_run_loadflow(network: Network) -> None:
    """Select a generator as slack and run loadflow.

    Powsybl tends to set the reference bus and slack as two different buses.
    Additionally, in some cases the slack is not a generator bus.
    This function selects a generator as slack bus and runs the loadflow again.

    Parameters
    ----------
    network : Network
        The Powsybl network to modify and run loadflow on.

    Raises
    ------
    ValueError
        If the loadflow does not converge after setting the slack.
        If the slack bus is not a generator.
    """
    try:
        # try to get slack from CGMES data
        b = network.get_buses()
        ref_bus = b[(b["v_angle"] == 0) & (b["connected_component"] == 0)]

        slack_voltage_id = ref_bus["voltage_level_id"].values[0]
        slack_bus_id = ref_bus.index.values[0]
    except Exception:
        # if not found, set first largest generator as slack
        generators = network.get_generators(attributes=["bus_id", "voltage_level_id", "max_p"])
        generators = generators.sort_values(by="max_p", ascending=False)
        first = 1
        slack_bus_id = generators["bus_id"].values[first]
        slack_voltage_id = generators["voltage_level_id"].values[first]

    dict_slack = {"voltage_level_id": slack_voltage_id, "bus_id": slack_bus_id}
    pypowsybl.network.Network.create_extensions(network, extension_name="slackTerminal", **dict_slack)
    network.get_extensions("slackTerminal")

    powsybl_loadflow_parameter = pypowsybl.loadflow.Parameters(
        voltage_init_mode=pypowsybl.loadflow.VoltageInitMode.DC_VALUES,
        read_slack_bus=True,
        distributed_slack=True,
        use_reactive_limits=True,
        component_mode=pypowsybl.loadflow.ComponentMode.MAIN_CONNECTED,  # ConnectedComponentMode
    )

    loadflow_res = pypowsybl.loadflow.run_ac(network, parameters=powsybl_loadflow_parameter)[0]
    if loadflow_res.status != pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED:
        raise ValueError(
            f"Load flow did not converge. Status: {loadflow_res.status}, "
            f"Status text: {loadflow_res.status_text}, "
            f"Reference bus ID: {loadflow_res.reference_bus_id}"
        )

    slack_bus_id = loadflow_res.slack_bus_results[0].id
    generators = network.get_generators(attributes=["bus_id"])
    if slack_bus_id not in generators["bus_id"].values:
        raise ValueError("The slack bus must be a generator.")


def load_pandapower_net_for_powsybl(
    net: pandapower.pandapowerNet, check_trafo_resistance: bool = True, set_slack_generator: bool = True
) -> pypowsybl.network.Network:
    """Load a pandapower network and convert it to a pypowsybl network.

    Known pandapower test grids that fail to convert:
    (This list is a logical AND of convert_from_pandapower and grid2opt conversion methods
    + the logical OR of check_powsybl_import errors)
    'example_multivoltage'      -> Generator minimum reactive power is not set
    'simple_four_bus_system'    -> Generator minimum reactive power is not set
    'simple_mv_open_ring_net'   -> 2 windings transformer '0_1_6': b is invalid
    'create_cigre_network_hv'   -> Generator minimum reactive power is not set
    'create_cigre_network_lv'   -> No Slack Generator found
    'case145'                   -> Transformer with negative resistance
    'case_illinois200'          -> No Slack Generator found
    'case300'                   -> No Slack Generator found

    Parameters
    ----------
    net : pandapower.pandapowerNet
        The pandapower network to convert.
    check_trafo_resistance : bool, optional
        If True, check for negative transformer resistance after conversion, by default True
    set_slack_generator : bool, optional
        If True, select a generator as slack and run loadflow after conversion, by default True

    Returns
    -------
    pypowsybl.network.Network
        The converted pypowsybl network.

    """
    try:
        pypowsybl_network = load_pandapower_net_for_powsybl_with_convert_from_pandapower(net)
        check_powsybl_import(pypowsybl_network, check_trafo_resistance=check_trafo_resistance)
    except (pypowsybl.PyPowsyblError, ValueError) as e:
        try:
            pypowsybl_network = load_pandapower_net_via_grid2opt_for_powsybl(net)
            check_powsybl_import(pypowsybl_network, check_trafo_resistance=check_trafo_resistance)
        except Exception as e2:
            raise ValueError(
                f"Failed to convert pandapower net to pypowsybl network. "
                f"pypowsybl.network.convert_from_pandapower: {e}. Conversion via grid2opt failed with error: {e2}"
            ) from e2
    if set_slack_generator:
        try:
            select_a_generator_as_slack_and_run_loadflow(pypowsybl_network)
        except Exception as e:
            raise ValueError(f"Slack selection failed after conversion from pandapower to powsybl: {e}") from e

    return pypowsybl_network


def load_pandapower_net_for_powsybl_with_convert_from_pandapower(net: pandapower.pandapowerNet) -> pypowsybl.network.Network:
    """Load a pandapower network and convert it to a pypowsybl network using convert_from_pandapower.

    Known pandapower test grids that fail to convert:
    'example_simple'            -> Generator minimum reactive power is not set
    'example_multivoltage'      -> Generator minimum reactive power is not set
    'simple_four_bus_system'    -> Generator minimum reactive power is not set
    'simple_mv_open_ring_net'   -> 2 windings transformer '0_1_6': b is invalid
    'create_cigre_network_hv'   -> Generator minimum reactive power is not set

    Parameters
    ----------
    net : pandapower.pandapowerNet
        The pandapower network to convert.

    Returns
    -------
    pypowsybl.network.Network
        The converted pypowsybl network.
    """
    pypowsybl_network = pypowsybl.network.convert_from_pandapower(net)
    return pypowsybl_network


def load_pandapower_net_via_grid2opt_for_powsybl(
    net: pandapower.pandapowerNet,
) -> pypowsybl.network.Network:
    """Load a pandapower network and convert it to a pypowsybl network using grid2opt.

    Known pandapower test grids that fail to convert:
    'example_multivoltage'      -> Transformer with negative resistance
    'create_cigre_network_hv'   -> Line with different voltage levels -> failed transformer conversion
    'case14'                    -> Line with different voltage levels -> failed transformer conversion
    'case_ieee30'               -> Line with different voltage levels -> failed transformer conversion
    'case57'                    -> Line with different voltage levels -> failed transformer conversion
    'case89pegase'              -> Line with different voltage levels -> failed transformer conversion
    'case118'                   -> Line with different voltage levels -> failed transformer conversion
    'case145'                   -> Transformer with negative resistance
    'case_illinois200'          -> Line with different voltage levels -> failed transformer conversion
    'case300'                   -> Line with different voltage levels -> failed transformer conversion

    Parameters
    ----------
    net : pandapower.pandapowerNet
        The pandapower network to convert.

    Returns
    -------
    pypowsybl.network.Network
        The converted pypowsybl network.
    """
    pandapower.runpp(net)
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=True) as tmpfile:
        _ = pandapower.converter.to_mpc(net, tmpfile.name)
        loading_params = {
            "matpower.import.ignore-base-voltage": "false",  # change the voltage from per unit to Kv
        }
        pypowsybl_network = pypowsybl.network.load(tmpfile.name, loading_params)
    return pypowsybl_network


def check_powsybl_import(pypowsybl_network: pypowsybl.network.Network, check_trafo_resistance: bool = True) -> None:
    """Check the import of a pypowsybl network.

    Parameters
    ----------
    pypowsybl_network : pypowsybl.network.Network
        The pypowsybl network to test.
    check_trafo_resistance : bool, optional
        If True, check for negative transformer resistance, by default True

    Raises
    ------
    ValueError
        If a transformer with negative resistance is found in the converted network.
        If a line with different voltage levels is found in the converted network.
        If the load flow does not converge.
    """
    # importing pn.example_multivoltage -> one transformer has negative resistance
    if check_trafo_resistance:
        transformers = pypowsybl_network.get_2_windings_transformers()
        if len(transformers[transformers["r"] < 0]) > 0:
            raise ValueError("A Transformer in the converted pandapower net has a negative resistance")

    # test if lines have the same voltage level
    line_voltage = pypowsybl_network.get_lines(attributes=["voltage_level1_id", "voltage_level2_id"])
    line_voltage = line_voltage.merge(
        pypowsybl_network.get_voltage_levels(attributes=["nominal_v"]), left_on="voltage_level1_id", right_index=True
    )
    line_voltage = line_voltage.merge(
        pypowsybl_network.get_voltage_levels(attributes=["nominal_v"]),
        left_on="voltage_level2_id",
        right_index=True,
        suffixes=("_vl1", "_vl2"),
    )
    if not all(line_voltage["nominal_v_vl1"] == line_voltage["nominal_v_vl2"]):
        raise ValueError("A Line in the converted pandapower net has two different voltage levels")

    loadflow_res = pypowsybl.loadflow.run_ac(pypowsybl_network, DISTRIBUTED_SLACK)[0]
    if loadflow_res.status != pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED:
        raise ValueError(f"Load flow failed: {loadflow_res.status_text}")
