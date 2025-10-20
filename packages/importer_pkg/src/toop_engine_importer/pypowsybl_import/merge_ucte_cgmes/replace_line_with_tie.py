"""Logic to replace a dangling node with tie lines."""

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
from pypowsybl.network import BusBreakerTopology
from pypowsybl.network.impl.network import Network


class DanglingLineCreationSchema(pa.DataFrameModel):
    """A Schema for network.create_dangling_lines."""

    name: pat.Series[str] = pa.Field(coerce=True)
    """The name of the line."""

    pairing_key: pat.Series[str] = pa.Field(coerce=True)
    """The dangling pairing key of the line."""

    bus_id: pat.Series[str] = pa.Field(coerce=True)
    """The bus id of the line.
       Note: depending on the connected attribute this can be a "connectable_bus" id or a "bus_id"."""

    voltage_level_id: pat.Series[str] = pa.Field(coerce=True)
    """The voltage level id of the line."""

    r: pat.Series[float] = pa.Field(coerce=True)
    """The resistance of the line."""

    x: pat.Series[float] = pa.Field(coerce=True)
    """The reactance of the line."""

    g: pat.Series[float] = pa.Field(coerce=True)
    """The conductance of the line."""

    b: pat.Series[float] = pa.Field(coerce=True)
    """The susceptance of the line."""

    p0: pat.Series[float] = pa.Field(coerce=True)
    """The active power of the line."""

    q0: pat.Series[float] = pa.Field(coerce=True)
    """The reactive power of the line."""

    connected: pat.Series[bool] = pa.Field(coerce=True)
    """The connected attribute of the line.
         Note: This is a boolean value that indicates if the line is connected or not.
         This determines if the bus_id or connectable_bus_id is used in network.create_dangling_lines.
     """


class DanglingGeneratorSchema(pa.DataFrameModel):
    """A Schema for network.create_dangling_lines."""

    id: pat.Index[str] = pa.Field(coerce=True)
    """The id of the generator."""

    min_p: pat.Series[float] = pa.Field(coerce=True)
    """The minimum active power output of the generator."""

    max_p: pat.Series[float] = pa.Field(coerce=True)
    """The maximum active power output of the generator."""

    target_p: pat.Series[float] = pa.Field(coerce=True)
    """The active power target of the generator."""

    target_q: pat.Series[float] = pa.Field(coerce=True)
    """The reactive power target of the generator."""

    target_v: pat.Series[float] = pa.Field(coerce=True)
    """The voltage target of the generator."""

    voltage_regulator_on: pat.Series[bool] = pa.Field(coerce=True)
    """True if the generator regulates voltage."""

    bus_id: pat.Series[str] = pa.Field(coerce=True)
    """The bus id of the generator.
       Used to identify the dangling line id."""

    powsybl_gen_id: pat.Series[str] = pa.Field(coerce=True)
    """The powsybl id of the generator."""

    class Config:
        """Config for the schema."""

        strict = True
        """Do not allow extra columns."""


def check_dangling_node(bus_breaker_topo: BusBreakerTopology) -> None:
    """
    Check if the bus breaker topology has a dangling node.

    A dangling node is a node that has no switches and only contains elements of type LINE.
    Dangling nodes where GENERATORs are present are supported.

    Parameters
    ----------
    bus_breaker_topo : BusBreakerTopology
        The bus breaker topology to check.
        from network.get_bus_breaker_topology(voltage_level)

    Raises
    ------
        ValueError: If the bus breaker topology has switches or no elements of type LINE.
        ValueError: If the bus breaker topology has no elements.
        ValueError: If the bus breaker topology has any elements other than LINE or LINE and GENERATOR.
        ValueError: If the bus breaker topology has more than 2 lines connected to one bus.
    """
    if len(bus_breaker_topo.switches) != 0:
        raise ValueError("The dangling Node contains switches, wrong voltage_level?")
    if len(bus_breaker_topo.elements) == 0:
        raise ValueError("The dangling Node contains no elements, wrong voltage_level?")

    if not (
        set(bus_breaker_topo.elements["type"].unique()) == {"LINE"}
        or set(bus_breaker_topo.elements["type"].unique()) == {"LINE", "GENERATOR"}
    ):
        raise ValueError(
            "The dangling Node contains elements of type"
            f" {list(bus_breaker_topo.elements['type'].unique())}, wrong voltage_level?"
            f"bus_ids: {bus_breaker_topo.elements['bus_id'].unique()}"
        )

    # check if only two lines are connected to one bus
    # maximum 2 lines are allowed
    # one line is also possible, if the dangling node is a generator
    max_lines_dangling_node = 2
    for bus_id in bus_breaker_topo.elements["bus_id"].unique():
        cond_bus = bus_breaker_topo.elements["bus_id"] == bus_id
        cond_line = bus_breaker_topo.elements["type"] == "LINE"
        if len(bus_breaker_topo.elements[cond_bus & cond_line]) > max_lines_dangling_node:
            raise ValueError(
                f"The dangling Node contains more than 2 lines connected to one bus: {bus_id}, "
                f"wrong voltage_level? bus_ids: {bus_breaker_topo.elements['bus_id'].unique()}"
            )


def get_dangling_creation_schema(
    network: Network, dangling_voltage_level: str, name_col: str = "elementName"
) -> tuple[DanglingLineCreationSchema, DanglingGeneratorSchema]:
    """Get the dangling lines and generator schema for a given voltage level.

    This expects that the voltage level is a dangling node.
    All lines that are connected will be converted to a dangling line dataframe.

    Parameters
    ----------
    network : Network
        The network to get the dangling lines from.
    dangling_voltage_level : str
        The id of the voltage level of the dangling node.
    name_col : str
        The column name to use for the name of the line. Default is "elementName".

    Returns
    -------
    dangling_line_creation_df : DanglingLineCreationSchema
        Contains lines that can be converted to dangling lines.
    dangling_generator_df : DanglingGeneratorSchema
        Contains generators that can be converted to dangling generator.
    """
    bus_breaker_topo = network.get_bus_breaker_topology(dangling_voltage_level)
    check_dangling_node(bus_breaker_topo)
    elements = bus_breaker_topo.elements
    lines = elements[elements["type"] == "LINE"]
    generators = elements[elements["type"] == "GENERATOR"]
    dangling_line_creation_df = get_dangling_lines_creation_schema(
        network=network, bus_breaker_topo_lines=lines, dangling_voltage_level=dangling_voltage_level, name_col=name_col
    )
    dangling_generator_df = get_dangling_generator_creation_schema(network=network, generators=generators)
    set_dangling_generator_ids(
        dangling_line_creation_df=dangling_line_creation_df, dangling_generator_df=dangling_generator_df
    )

    return dangling_line_creation_df, dangling_generator_df


def get_dangling_generator_creation_schema(network: Network, generators: pd.DataFrame) -> DanglingGeneratorSchema:
    """Get the dangling generator schema for a given BusBreakerTopology.elements.

    Note: This expects that the BusBreakerTopology.elements are generators.
    All generators that are connected will be converted to a dangling generator dataframe.

    Parameters
    ----------
    network : Network
        The network to get the dangling lines from.
    generators : pd.DataFrame
        The dataframe with the generators to convert to dangling generator.
        This is the BusBreakerTopology.elements dataframe filtered for generators.

    Returns
    -------
    dangling_generator_df : DanglingGeneratorSchema
        Contains generators that can be converted to dangling generator.
        Note: the correct ids have not been set yet.

    Raises
    ------
        ValueError: If the dangling generator dataframe contains active or reactive power.
        This case is not supported. There is likely no tie line to connect to.
        Likely a Dangling Node that ends.
    """
    columns = ["min_p", "max_p", "target_p", "target_q", "target_v", "voltage_regulator_on"]
    dangling_generator_df = generators.merge(
        network.get_generators(attributes=[*columns, "p", "q"]),
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("", "_gen"),
    )
    # change inner german name to X-node name
    dangling_generator_df["bus_id"] = dangling_generator_df["bus_id"].apply(lambda x: "X" + x[1:] if len(x) > 0 else x)
    dangling_generator_df["powsybl_gen_id"] = dangling_generator_df.index
    if not (dangling_generator_df["p"].isnull().all() or dangling_generator_df["q"].isnull().all()):
        raise ValueError(
            "The dangling generator dataframe contains active or reactive power. "
            "This case is not supported. There is likely no tie line to connect to."
        )

    columns = ["bus_id", "powsybl_gen_id", *columns]
    dangling_generator_df = dangling_generator_df[columns]
    # check schema
    dangling_generator_df = DanglingGeneratorSchema.validate(dangling_generator_df)
    return dangling_generator_df


def set_dangling_generator_ids(
    dangling_line_creation_df: DanglingLineCreationSchema, dangling_generator_df: DanglingGeneratorSchema
) -> None:
    """Set the ids of the dangling_generator_df.

    This will set the ids of the dangling_generator_df based on the dangling_line_creation_df.
    A generator id must match a dangling line id to create a dangling line with generator

    Parameters
    ----------
    dangling_line_creation_df : DanglingLineCreationSchema
        The dataframe with the dangling lines to set the ids for.
    dangling_generator_df : DanglingGeneratorSchema
        The dataframe with the generators to set the ids for.
        Note: the ids are set in place: id, bus_id

    Raises
    ------
        ValueError: If the generator id does not match a dangling line id.
        Error in the selected node. Not a dangling node?
    """
    dangling_generator_df.reset_index(inplace=True)
    single_line = 1
    double_line = 2
    for index, row in dangling_generator_df.iterrows():
        dangling_found = dangling_line_creation_df[dangling_line_creation_df["pairing_key"] == row["bus_id"]]
        if len(dangling_found) == single_line:
            dangling_generator_df.loc[index, "id"] = dangling_found.index[0]
            dangling_generator_df.loc[index, "bus_id"] = dangling_found["bus_id"].iloc[0]
        elif len(dangling_found) == double_line:
            # generator will be created for each dangling line
            dangling_generator_df.loc[index, "id"] = dangling_line_creation_df[
                dangling_line_creation_df["pairing_key"] == row["bus_id"]
            ].index[0]
            dangling_generator_df.loc[index, "bus_id"] = dangling_line_creation_df[
                dangling_line_creation_df["pairing_key"] == row["bus_id"]
            ]["bus_id"].iloc[0]
            # add a new row to the dangling_generator_df with the same values as the first one
            new_row = dangling_generator_df.iloc[index].copy()
            new_row["id"] = dangling_line_creation_df[dangling_line_creation_df["pairing_key"] == row["bus_id"]].index[1]
            new_row["bus_id"] = dangling_line_creation_df[dangling_line_creation_df["pairing_key"] == row["bus_id"]][
                "bus_id"
            ].iloc[1]
            dangling_generator_df.loc[len(dangling_generator_df)] = new_row
        else:
            raise ValueError(
                f"Generator {row['id']} has no matching line in "
                f"dangling_line_creation_df or multiple matches found: {dangling_found.index}."
            )
    # check schema
    dangling_generator_df.set_index("id", inplace=True)
    dangling_generator_df = DanglingGeneratorSchema.validate(dangling_generator_df)


def get_dangling_lines_creation_schema(
    network: Network, bus_breaker_topo_lines: pd.DataFrame, dangling_voltage_level: str, name_col: str = "elementName"
) -> DanglingLineCreationSchema:
    """Get the dangling lines dataframe for a given voltage level.

    This expects that the voltage level is a dangling node.
    All lines that are connected will be converted to a dangling line dataframe.

    Parameters
    ----------
    network : Network
        The network to get the dangling lines from.
    bus_breaker_topo_lines : pd.DataFrame
        expects network.get_bus_breaker_topology(voltage_level).elements filtered for lines
    dangling_voltage_level : str
        The id of the voltage level of the dangling node.
    name_col : str
        The column name to use for the name of the line. Default is "elementName".

    Returns
    -------
    new_dangling_df : DanglingLineCreationSchema
        Contains lines that can be converted to dangling lines.
    """
    # get the lines from the dangling voltage level
    new_dangling_df = bus_breaker_topo_lines.merge(
        network.get_lines(all_attributes=True), how="left", left_index=True, right_index=True
    )
    # define name
    if name_col != "name" and name_col in new_dangling_df.columns:
        new_dangling_df.drop(columns=["name"], inplace=True)
        new_dangling_df.rename(columns={name_col: "name"}, inplace=True)
    # define pairing_key
    new_dangling_df.rename(columns={"bus_id": "pairing_key"}, inplace=True)
    # change inner german name to X-node name
    new_dangling_df["pairing_key"] = new_dangling_df["pairing_key"].apply(lambda x: "X" + x[1:] if len(x) > 0 else x)

    # define empty columns
    new_dangling_df["bus_id"] = ""
    new_dangling_df["voltage_level_id"] = ""
    new_dangling_df["g"] = 0.0
    new_dangling_df["b"] = 0.0
    new_dangling_df["p0"] = 0.0
    new_dangling_df["q0"] = 0.0
    new_dangling_df["connected"] = True

    for index, row in new_dangling_df.iterrows():
        # get only lines
        if row["type"] != "LINE":
            continue
        # define dangling information depending on the voltage level side
        if row["voltage_level1_id"] == dangling_voltage_level:
            new_dangling_df.loc[index, "voltage_level_id"] = row["voltage_level2_id"]
            new_dangling_df.loc[index, "bus_id"] = row["bus_breaker_bus2_id"]
            if row["pairing_key"] == "":
                new_dangling_df.loc[index, "pairing_key"] = "X" + row["bus_breaker_bus1_id"][1:]
        else:
            new_dangling_df.loc[index, "voltage_level_id"] = row["voltage_level1_id"]
            new_dangling_df.loc[index, "bus_id"] = row["bus_breaker_bus1_id"]
            if row["pairing_key"] == "":
                new_dangling_df.loc[index, "pairing_key"] = "X" + row["bus_breaker_bus2_id"][1:]
        new_dangling_df.loc[index, "b"] = row["b1"] + row["b2"]
        new_dangling_df.loc[index, "g"] = row["g1"] + row["g2"]
        # double check if connected is consistent
        if row["connected1"] and row["connected2"]:
            new_dangling_df.loc[index, "connected"] = True
        elif not row["connected1"] and not row["connected2"]:
            new_dangling_df.loc[index, "connected"] = False
        else:
            raise ValueError(
                f"Connected at {index} is not consistent. connected1: "
                f"{row['connected1']} and connected2: {row['connected2']}"
                "This is likely a data quality issue."
            )

    dangling_columns = ["name", "pairing_key", "bus_id", "voltage_level_id", "r", "x", "g", "b", "p0", "q0", "connected"]
    new_dangling_df = new_dangling_df[dangling_columns]
    return new_dangling_df


def replace_line_with_dangling_line(
    network: Network,
    dangling_line_creation_df: DanglingLineCreationSchema,
    dangling_gen_creation_df: DanglingGeneratorSchema,
) -> None:
    """Replace the lines in the network with dangling lines.

    This will remove the lines from the network and create dangling lines.
    The lines are replaced in place.

    Parameters
    ----------
    network : Network
        The network to modify. Note: The network is modified in place.
        removes the lines and generators from the network and creates dangling lines
    dangling_line_creation_df : DanglingLineCreationSchema
        The dataframe with the lines to replace.
    dangling_gen_creation_df : DanglingGeneratorSchema
        The dataframe with the generators to replace.
    """
    network.remove_elements(dangling_line_creation_df.index)
    disconnected_lines = dangling_line_creation_df[~dangling_line_creation_df["connected"]]
    dangling_line_creation_df.drop(columns=["connected"], inplace=True)
    if len(dangling_gen_creation_df) > 0:
        network.remove_elements(list((dangling_gen_creation_df["powsybl_gen_id"].unique())))
        dangling_gen_creation_df.drop(columns=["bus_id"], inplace=True)
        dangling_gen_creation_df.drop(columns=["powsybl_gen_id"], inplace=True)
        network.create_dangling_lines(df=dangling_line_creation_df, generator_df=dangling_gen_creation_df)
    else:
        network.create_dangling_lines(df=dangling_line_creation_df)

    for disconnected_line in disconnected_lines.index:
        network.disconnect(disconnected_line)


def reconnect_dangling_as_tie_line(network: Network, new_dangling_df: DanglingLineCreationSchema) -> None:
    """Reconnect the dangling lines as tie lines.

    The new dangling lines are created as tie lines.

    Parameters
    ----------
    network : Network
        The network to modify. Note: The network is modified in place.
        Creates tie lines from the dangling lines.
    new_dangling_df : DanglingLineCreationSchema
        The dataframe with the lines to replace.

    Raises
    ------
        ValueError: If the pairing key does not have 2 dangling lines.
    """
    pair = 2
    for pairing_key in new_dangling_df["pairing_key"].unique():
        if len(new_dangling_df[new_dangling_df["pairing_key"] == pairing_key]) != pair:
            raise ValueError(
                f"Pairing key {pairing_key} has not 2 dangling lines. "
                f"Found lines: {new_dangling_df[new_dangling_df['pairing_key']].index}"
            )
        dangling_line1_id = new_dangling_df[new_dangling_df["pairing_key"] == pairing_key].index[0]
        dangling_line2_id = new_dangling_df[new_dangling_df["pairing_key"] == pairing_key].index[1]
        tie_id = f"{dangling_line1_id} + {dangling_line2_id}"
        network.create_tie_lines(id=tie_id, dangling_line1_id=dangling_line1_id, dangling_line2_id=dangling_line2_id)


def replace_voltage_level_with_tie_line(network: Network, voltage_level_id: str, name_col: str = "elementName") -> None:
    """Replace the voltage level with a tie line.

    This will remove the voltage level from the network and create a tie line.
    The voltage level is replaced in place.

    Parameters
    ----------
    network : Network
        The network to modify. Note: The network is modified in place.
        removes the voltage level from the network and creates a tie line
    voltage_level_id : str
        The id of the voltage level to replace.
    name_col : str
        The column name to use for the name of the line. Default is "elementName".
    """
    dangling_line_creation_df, dangling_gen_creation_df = get_dangling_creation_schema(network, voltage_level_id, name_col)
    replace_line_with_dangling_line(network, dangling_line_creation_df, dangling_gen_creation_df)
    reconnect_dangling_as_tie_line(network, dangling_line_creation_df)
    network.remove_elements(voltage_level_id)


def get_dangling_voltage_levels(network: Network, external_border_mask: np.ndarray, area_codes: list[str]) -> list[str]:
    """Get the dangling voltage levels from the network.

    Get the dangling voltage levels from the network that are connected to the external border.

    Parameters
    ----------
    network : Network
        The network to get the dangling lines from.
    external_border_mask: np.ndarray
        A boolean array over all lines, that depicts outgoing border lines
    area_codes: list[str]
        A list of area codes to check for. The area codes are used to check if the voltage level is a dangling voltage level.

    Returns
    -------
    dangling_voltage_levels : list[str]
        A list of dangling voltage levels that are connected to the external border.
    """
    dangling_df = network.get_lines()[external_border_mask]
    dangling_voltage_levels = []

    for index, row in dangling_df.iterrows():
        for area_code in area_codes:
            if row["voltage_level1_id"][: len(area_code)] == area_code:
                dangling_voltage_levels.append(row["voltage_level2_id"])
            elif row["voltage_level2_id"][: len(area_code)] == area_code:
                dangling_voltage_levels.append(row["voltage_level1_id"])
            else:
                raise ValueError(
                    f"Area code {area_code} not found in bus_breaker_bus1_id or bus_breaker_bus2_id in row {index}"
                )
    return list(set(dangling_voltage_levels))
