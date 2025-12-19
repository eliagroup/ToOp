# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to handle pandapower ids

Pandapower indices are usually not globally unique, so we provide some helper functions to make
them globally unique.
"""

import pandas as pd
from beartype.typing import Optional, Union

SEPARATOR = "%%"


def get_globally_unique_id(elem_id: Union[str, int], elem_type: Optional[str]) -> str:
    """Get a globally unique id for an element

    Unfortunately, pandapowers ids are only unique within their type, so we need to add the type
    to the id to make it globally unique.

    Parameters
    ----------
    elem_id : Union[str, int]
        The id of the element
    elem_type : str
        The type of the element

    Returns
    -------
    str
        The globally unique id of the element
    """
    if elem_type is not None:
        # Sometimes, the elem_type comes with postfixes to separate
        elem_type = str(elem_type).replace(SEPARATOR, "&&")
    else:
        elem_type = ""

    return f"{elem_id}{SEPARATOR}{elem_type}"


def parse_globally_unique_id(globally_unique_id: str) -> tuple[int, str]:
    """Parse a globally unique id into its components

    Parameters
    ----------
    globally_unique_id : str
        The globally unique id

    Returns
    -------
    int
        The id of the element
    str
        The type of the element
    """
    elem_id, elem_type = globally_unique_id.split(SEPARATOR)
    return int(elem_id), elem_type


def get_globally_unique_id_from_index(element_idx: pd.Index | pd.Series, element_type: str) -> pd.Index | pd.Series:
    """Parse a series of globally unique ids into a dataframe

    Parameters
    ----------
    element_idx : pd.Index | pd.Series
        The index of the element table
    element_type : str
        The type of the element table (e.g. "bus", "line", etc.)

    Returns
    -------
    pd.Index
        The index with added table name as prefix to make it globally unique
    """
    globally_unique_ids = element_idx.astype(str) + SEPARATOR + element_type
    return globally_unique_ids


def parse_globally_unique_id_series(globally_unique_ids: pd.Series) -> pd.DataFrame:
    """Parse a series of globally unique ids into a dataframe

    Parameters
    ----------
    globally_unique_ids : pd.Series
        The series of globally unique ids

    Returns
    -------
    pd.DataFrame
        A dataframe with the id, type and name of the elements
    """
    parsed_ids = globally_unique_ids.str.split(SEPARATOR, expand=True)
    parsed_ids.columns = ["id", "type"]
    parsed_ids["id"] = parsed_ids["id"].astype(int)
    return parsed_ids


def table_id(globally_unique_id: str) -> Union[int, str]:
    """Get the id in the pandapower table from a globally unique id

    Parameters
    ----------
    globally_unique_id : str
        The globally unique id

    Returns
    -------
    Union[int, str]
        The id in the pandapower table
    """
    elem_id, _ = parse_globally_unique_id(globally_unique_id)
    return elem_id


def table_ids(list_of_globally_unique_ids: list[str]) -> list[int]:
    """Get the ids in the pandapower table from a list of globally unique ids

    Parameters
    ----------
    list_of_globally_unique_ids : list[str]
        The list of globally unique ids

    Returns
    -------
    list[int]
        The ids in the pandapower table
    """
    return [table_id(globally_unique_id) for globally_unique_id in list_of_globally_unique_ids]
