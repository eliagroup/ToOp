"""Module provides tools for parsing UCTE files."""

import re
from dataclasses import dataclass
from functools import partial
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Spec:
    """Dataclass for a specification of a column in a fixed-width file."""

    start: int
    end: int
    name: str
    dtype: type
    default: Optional[Any] = None

    def __post_init__(self) -> None:
        """Check if start and and value are valid."""
        if self.start >= self.end:
            raise ValueError("Start must be smaller than end")


# For parsing the UCTE format we need those colspecs, which are taken from
# https://eepublicdownloads.entsoe.eu/clean-documents/pre2015/publications/ce/otherreports/UCTE-format.pdf
# (Copied by hand)
SpecsT = Dict[str, List[Spec]]
specs: SpecsT = {
    "node": [
        Spec(0, 8, "code", str),
        Spec(9, 21, "name", str),
        Spec(22, 23, "status", int),
        Spec(24, 25, "type", str),
        Spec(26, 32, "voltage", float, np.nan),
        Spec(33, 40, "p_load", float, 0),
        Spec(41, 48, "q_load", float, 0),
        Spec(49, 56, "p_gen", float, 0),
        Spec(57, 64, "q_gen", float, 0),
        Spec(65, 72, "min_p_gen", float, -np.inf),
        Spec(73, 80, "max_p_gen", float, np.inf),
        Spec(81, 88, "min_q_gen", float, -np.inf),
        Spec(89, 96, "max_q_gen", float, np.inf),
        Spec(97, 102, "static_primary_control", float, np.nan),
        Spec(103, 110, "p_primary_control", float, np.nan),
        Spec(111, 118, "3ph_short_circuit_power", float, np.nan),
        Spec(119, 126, "x_r_ratio", float, np.nan),
        Spec(127, 128, "node_type", str),
    ],
    "line": [
        Spec(0, 8, "from", str),
        Spec(9, 17, "to", str),
        Spec(18, 19, "order", str),
        Spec(20, 21, "status", int),
        Spec(22, 28, "r", float),
        Spec(29, 35, "x", float),
        Spec(36, 44, "b", float),
        Spec(45, 51, "i", float, np.inf),
        Spec(52, 64, "name", str),
    ],
    "trafo": [
        Spec(0, 8, "from", str),
        Spec(9, 17, "to", str),
        Spec(18, 19, "order", str),
        Spec(20, 21, "status", int),
        Spec(22, 27, "voltage1", float),
        Spec(28, 33, "voltage2", float),
        Spec(34, 39, "p", float),
        Spec(40, 46, "r", float),
        Spec(47, 53, "x", float),
        Spec(54, 62, "b", float),
        Spec(63, 69, "g", float),
        Spec(70, 76, "i", float, np.inf),
        Spec(77, 89, "name", str),
    ],
    "trafo_reg": [
        Spec(0, 8, "from", str),
        Spec(9, 17, "to", str),
        Spec(18, 19, "order", str),
        Spec(20, 25, "phase_reg_delta_u", float, np.nan),
        Spec(26, 28, "phase_reg_n", float, np.nan),
        Spec(29, 32, "phase_reg_n2", float, np.nan),
        Spec(33, 38, "phase_reg_u", float, np.nan),
        Spec(39, 44, "angle_reg_delta_u", float, np.nan),
        Spec(45, 50, "angle_reg_theta", float, np.nan),
        Spec(51, 53, "angle_reg_n", float, np.nan),
        Spec(54, 57, "angle_reg_n2", float, np.nan),
        Spec(58, 63, "angle_reg_p", float, np.nan),
        Spec(64, 68, "type", str),
    ],
}

# colnames_node = ["code", "name", "status", "type", "voltage", "p_load", "q_load", "p_gen", "q_gen", "min_p_gen", "max_p_gen", "min_q_gen", "max_q_gen", "static_primary_control", "p_primary_control", "3ph_short_circuit_power", "x_r_ratio", "node_type"]   # noqa: E501
# colspecs_node = [(0, 8), (9, 21), (22, 23), (24, 25), (26, 32), (33, 40), (41, 48), (49, 56), (57, 64), (65, 72), (73, 80), (81, 88), (89, 96), (97, 102), (103, 110), (111, 118), (119, 126), (127, 128)]   # noqa: E501
# colnames_line = ["from", "to", "order", "status", "r", "x", "b", "i", "name"]
# colspecs_line = [(0, 8), (9, 17), (18, 19), (20, 21), (22, 28), (29, 35), (36, 44), (45, 51), (52, 64)]
# colnames_trafo = ["from", "to", "order", "status", "voltage1", "voltage2", "p", "r", "x", "b", "g", "i", "name"]
# colspecs_trafo = [(0, 8), (9, 17), (18, 19), (20, 21), (22, 27), (28, 33), (34, 39), (40, 46), (47, 53), (54, 62), (63, 69), (70, 76), (77, 89)]   # noqa: E501
# colnames_trafo_reg = ["from", "to", "order", "phase_reg_delta_u", "phase_reg_n", "phase_reg_n2", "phase_reg_u", "angle_reg_delta_u", "angle_reg_theta", "angle_reg_n", "angle_reg_n2", "angle_reg_p", "type"]   # noqa: E501
# colspecs_trafo_reg = [(0, 8), (9, 17), (18, 19), (20, 25), (26, 28), (29, 32), (33, 38), (39, 44), (45, 50), (51, 53), (54, 57), (58, 63), (64, 68)]   # noqa: E501

# Furthermore prepare a regex for matching sections
section_re = re.compile(r"\#\#[^\n]*\n")


def split_ucte(contents: str) -> Tuple[str, str, str, str, str, str]:
    """Split a UCTE file into its individual sections.

    Parameters
    ----------
    contents : str
        The contents of the ucte file as a big string

    Returns
    -------
    str
        The preamble of the file, unparsed, containing the comments sections ##C
    str
        The nodes section
    str
        The lines section
    str
        The transformers section
    str
        The transformer regulation section
    str
        The postamble of the file, unparsed, containing the ##TT and ##E sections
    """
    nodes_begin = contents.find("##N")
    lines_begin = contents.find("##L", nodes_begin)
    trafos_begin = contents.find("##T", lines_begin)
    trafo_reg_begin = contents.find("##R", trafos_begin)
    trafo_special_begin = contents.find("##TT", trafo_reg_begin)

    # For the nodes we don't want to replace hashtags just yet, we'll separate by sections later
    preamble = contents[:nodes_begin]
    nodes = contents[nodes_begin:lines_begin]
    lines = contents[lines_begin:trafos_begin]
    trafos = contents[trafos_begin:trafo_reg_begin]
    trafo_reg = contents[trafo_reg_begin:trafo_special_begin]
    postamble = contents[trafo_special_begin:]

    return preamble, nodes, lines, trafos, trafo_reg, postamble


def parse_dataframe(text: str, specs: List[Spec]) -> pd.DataFrame:
    """Parse a dataframe from a fixed-width text file and column specifications.

    Parameters
    ----------
    text : str
        The text to parse from the ucte file, should be a single section (split using split_ucte)
    specs : List[Spec]
        The specifications for the section

    Returns
    -------
    df : pd.DataFrame
        The parsed dataframe
    """
    colspecs = [(s.start, s.end) for s in specs]
    colnames = [s.name for s in specs]
    df = pd.read_fwf(
        StringIO(text),
        colspecs=colspecs,
        header=None,
        dtype=str,
        na_filter=False,
        delimiter="\n",
    )
    df.columns = colnames
    return df


def parse_ucte(
    contents: str,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Parse a ucte file in the form of a string, returning individual sections as dataframes.

    Parameters
    ----------
    contents : str
        The contents of the ucte file

    Returns
    -------
    preamble : str
        The preamble of the file, unparsed, containing the comments sections ##C
    nodes_df : pd.DataFrame
        The parsed node section
    lines_df : pd.DataFrame
        The parsed line section
    trafos_df : pd.DataFrame
        The parsed transformer section
    trafo_reg_df : pd.DataFrame
        The parsed transformer regulation section
    postamble : str
        The postamble of the file, unparsed, containing the ##TT and ##E sections
    """
    preamble, nodes, lines, trafos, trafo_reg, postamble = split_ucte(contents)

    # Lines, trafos and trafo_regs can be replaced directly
    lines = re.sub(section_re, "", lines)
    trafos = re.sub(section_re, "", trafos)
    trafo_reg = re.sub(section_re, "", trafo_reg)

    # Match each node section individually
    last_match = None
    node_sections = []
    for cur_match in re.finditer(section_re, nodes):
        if last_match is not None:
            node_sections.append(
                (
                    last_match.group(0).strip(),
                    nodes[last_match.end() : cur_match.start()],
                )
            )
        last_match = cur_match
    assert last_match is not None
    node_sections.append((last_match.group(0).strip(), nodes[last_match.end() :]))

    node_sections_dict = {name: parse_dataframe(text, specs=specs["node"]) for (name, text) in node_sections if len(text)}
    for s_key, _ in node_sections_dict.items():
        node_sections_dict[s_key]["country"] = s_key
    nodes_df = pd.concat(node_sections_dict.values())
    nodes_df = nodes_df.reset_index(drop=True)
    lines_df = parse_dataframe(lines, specs=specs["line"])
    trafos_df = parse_dataframe(trafos, specs=specs["trafo"])
    trafo_reg_df = parse_dataframe(trafo_reg, specs=specs["trafo_reg"])

    return preamble.strip(), nodes_df, lines_df, trafos_df, trafo_reg_df, postamble


def apply_types(df: pd.DataFrame, specs: List[Spec]) -> None:
    """Apply types to a dataframe according to a specification.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to apply types to, modified in place
    specs : List[Spec]
        The specification for the columns
    """
    for s in specs:
        if s.default is not None:
            empty_string = " " * (s.end - s.start)
            df[s.name] = df[s.name].replace(empty_string, s.default).replace(np.nan, s.default)
        try:
            df[s.name] = df[s.name].astype(s.dtype)
        except ValueError as e:
            raise ValueError(f"Error converting column {s.name} to type {s.dtype}") from e


def interpret_ucte(
    nodes: pd.DataFrame,
    lines: pd.DataFrame,
    trafos: pd.DataFrame,
    trafo_reg: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Interpret the parsed UCTE data.

    Converts to the correct type and fixes missing values

    Parameters
    ----------
    nodes : pd.DataFrame
        The parsed node section
    lines : pd.DataFrame
        The parsed line section
    trafos : pd.DataFrame
        The parsed transformer section
    trafo_reg : pd.DataFrame
        The parsed transformer regulation section

    Returns
    -------
    pd.DataFrame
        The interpreted node section
    pd.DataFrame
        The interpreted line section
    pd.DataFrame
        The interpreted transformer section with trafo_reg joined
    """
    nodes = nodes.copy()
    apply_types(nodes, specs["node"])

    lines = lines.copy()
    apply_types(lines, specs["line"])

    # Join trafo and trafo-reg
    trafos = pd.merge(left=trafos, right=trafo_reg, how="left", on=["from", "to", "order"])
    apply_types(trafos, specs["trafo"])
    apply_types(trafos, specs["trafo_reg"])

    return nodes, lines, trafos


def convert_row_to_str(row: pd.Series, spec: List[Spec]) -> str:
    """Print a row of a dataframe as a string with fixed-width columns according to a specification.

    Parameters
    ----------
    row : pd.Series
        The dataframe row to convert
    spec : List[Spec]
        The specification for the fixed-width columns

    Returns
    -------
    str
        The row as a string with fixed-width columns
    """
    retval = " " * (spec[-1].end + 1)
    for i, s in enumerate(spec):
        retval = retval[: s.start] + row[i].rjust(s.end - s.start, " ")[: s.end - s.start] + retval[s.end :]
    return retval


def create_section(data: pd.DataFrame, spec: List[Spec], section_header: str) -> str:
    """Print a section of the UCTE file to string.

    Parameters
    ----------
    data : pd.DataFrame
        The data of the section
    spec : List[Spec]
        The specification for the fixed-width columns
    section_header : str
        The header of the section, e.g. ##N

    Returns
    -------
    str
        The section as a string
    """
    return section_header + "\n" + "\n".join(data.apply(partial(convert_row_to_str, spec=spec), axis=1))


def convert_data(
    nodes: pd.DataFrame,
    lines: pd.DataFrame,
    trafos: pd.DataFrame,
    trafo_reg: pd.DataFrame,
) -> str:
    """Convert the parsed data back to a UCTE file.

    Excludes the preamble and postamble, only the data sections are converted

    Parameters
    ----------
    nodes : pd.DataFrame
        The parsed node section
    lines : pd.DataFrame
        The parsed line section
    trafos : pd.DataFrame
        The parsed transformer section
    trafo_reg : pd.DataFrame
        The parsed transformer regulation section

    Returns
    -------
    str
        The data sections as a string
    """
    countries = sorted(nodes["country"].unique().tolist())

    parsed = ["##N"] + [create_section(nodes[nodes["country"] == country], specs["node"], country) for country in countries]
    parsed.append(create_section(lines, specs["line"], "##L"))
    parsed.append(create_section(trafos, specs["trafo"], "##T"))
    parsed.append(create_section(trafo_reg, specs["trafo_reg"], "##R"))
    return "\n".join(parsed)


def make_ucte(
    preamble: str,
    nodes: pd.DataFrame,
    lines: pd.DataFrame,
    trafos: pd.DataFrame,
    trafo_reg: pd.DataFrame,
    postamble: str,
) -> str:
    """Create a UCTE file from the parsed data.

    Parameters
    ----------
    preamble : str
        The preamble of the file, unparsed, containing the comments sections ##C
    nodes : pd.DataFrame
        The parsed node section
    lines : pd.DataFrame
        The parsed line section
    trafos : pd.DataFrame
        The parsed transformer section
    trafo_reg : pd.DataFrame
        The parsed transformer regulation section
    postamble : str
        The postamble of the file, unparsed, containing the ##TT and ##E sections

    Returns
    -------
    str
        The full UCTE file that can be written to disk directly
    """
    return "\n".join([preamble, convert_data(nodes, lines, trafos, trafo_reg), postamble])
