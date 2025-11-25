"""Provides general helper functions for pypowsybl networks.

These functions are used to extract and manipulate data from pypowsybl networks,
such as loadflow results, branch limits, and monitored elements.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pypowsybl
from beartype.typing import Literal, Optional
from fsspec import AbstractFileSystem
from pypowsybl.network import Network


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


def load_powsybl_from_fs(filesystem: AbstractFileSystem, file_path: Path) -> pypowsybl.network.Network:
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
        return pypowsybl.network.load(str(tmp_grid_path))


def save_powsybl_to_fs(
    net: pypowsybl.network.Network,
    filesystem: AbstractFileSystem,
    file_path: Path,
    format: Optional[Literal["CGMES", "XIIDM", "UCTE", "MATPOWER"]] = "XIIDM",
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
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_grid_path = Path(temp_dir) / file_path.name
        net.save(str(tmp_grid_path), format=format)
        filesystem.upload(
            str(tmp_grid_path),
            str(file_path),
        )
