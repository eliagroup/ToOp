"""Import contingencies from a file.

This module contains functions to import contingencies from a file and match them with the grid model.

Author: Benjamin Petrick
Created: 2025-05-13
"""

from pathlib import Path

import logbook
import pandas as pd
from toop_engine_importer.contingency_from_power_factory.power_factory_data_class import (
    AllGridElementsSchema,
    ContingencyImportSchemaPowerFactory,
    ContingencyMatchSchema,
)

logger = logbook.Logger(__name__)


def get_contingencies_from_file(n1_file: Path, delimiter: str = ";") -> ContingencyImportSchemaPowerFactory:
    """Get the contingencies from the file.

    This function reads the contingencies from the file and returns a DataFrame in the
    ContingencyImportSchema format.

    Parameters
    ----------
    n1_file : Path
        The path to the file.
    delimiter : str
        The delimiter of the file. Default is ";".

    Returns
    -------
    ContingencyImportSchema
        A DataFrame containing the contingencies.
    """
    n1_definition = pd.read_csv(n1_file, delimiter=delimiter)
    cond = n1_definition["power_factory_grid_model_name"].isna()
    n1_definition.loc[cond, "power_factory_grid_model_name"] = n1_definition.loc[cond, "contingency_name"]
    n1_definition["contingency_id"] = n1_definition["contingency_id"].astype(int)
    n1_definition = ContingencyImportSchemaPowerFactory.validate(n1_definition)
    return n1_definition


def match_contingencies(
    n1_definition: ContingencyImportSchemaPowerFactory,
    all_element_names: AllGridElementsSchema,
    match_by_name: bool = True,
) -> ContingencyMatchSchema:
    """Match the contingencies from the file with the elements in the grid model.

    This function matches the contingencies from the file with the elements in the grid model.
    It first tries to match by index, then by name.

    Parameters
    ----------
    n1_definition : ContingencyImportSchema
        The contingencies from the file.
    all_element_names : AllGridElementsSchema
        The elements in the grid model.
    match_by_name : bool
        If True, match by name. Default is True.
        If False, only match by index.

    Returns
    -------
    ContingencyMatchSchema
        A DataFrame containing the matched contingencies.
    """
    processed_n1_definition = match_contingencies_by_index(n1_definition, all_element_names)
    if match_by_name:
        processed_n1_definition = match_contingencies_by_name(processed_n1_definition, all_element_names)
    return processed_n1_definition


def match_contingencies_by_index(
    n1_definition: ContingencyImportSchemaPowerFactory, all_element_names: AllGridElementsSchema
) -> ContingencyMatchSchema:
    """Match the contingencies from the file with the elements in the grid model.

    Parameters
    ----------
    n1_definition : ContingencyImportSchema
        The contingencies from the file.
    all_element_names : AllGridElementsSchema
        The elements in the grid model.

    Returns
    -------
    ContingencyMatchSchema
        A DataFrame containing the matched contingencies.
    """
    # match grid_model_ids directly
    processed_n1_definition = n1_definition.merge(
        all_element_names, how="left", left_on="power_factory_grid_model_rdf_id", right_on="grid_model_id"
    )
    if (~processed_n1_definition["grid_model_name"].isna()).sum() == 0:
        # no elements found, try to remove underscore from rdf_id
        n1_definition["power_factory_grid_model_rdf_id"] = n1_definition["power_factory_grid_model_rdf_id"].str[1:]
        processed_n1_definition = n1_definition.merge(
            all_element_names, how="left", left_on="power_factory_grid_model_rdf_id", right_on="grid_model_id"
        )
    if (~processed_n1_definition["grid_model_name"].isna()).sum() == 0:
        logger.warning("No elements found in the grid model via CIM id. Check the grid model and the contingency file.")

    processed_n1_definition = ContingencyMatchSchema.validate(processed_n1_definition)
    return processed_n1_definition


def match_contingencies_by_name(
    processed_n1_definition: ContingencyMatchSchema,
    all_element_names: AllGridElementsSchema,
) -> ContingencyMatchSchema:
    """Match the contingencies from the file with the elements in the grid model by name.

    Matches by name and replaces the grid_model_name, element_type and grid_model_id.
    First tries to match 100% of the name.
    Second tries to match by removing spaces and replacing "+" with "##_##".

    Parameters
    ----------
    processed_n1_definition : ContingencyMatchSchema
        The contingencies from the file.
    all_element_names : AllGridElementsSchema
        The elements in the grid model.

    Returns
    -------
    ContingencyMatchSchema
        A DataFrame containing the matched contingencies.
    """
    processed_n1_definition = match_contingencies_column(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
        n1_column="power_factory_grid_model_name",
        element_column="grid_model_name",
    )

    # a "+" sometimes makes problems in the name matching
    all_element_names["grid_model_name_no_space"] = (
        all_element_names["grid_model_name"].str.replace(" ", "").str.replace("+", "##_##")
    )
    processed_n1_definition["power_factory_grid_model_name_no_space"] = (
        processed_n1_definition["power_factory_grid_model_name"].str.replace(" ", "").str.replace("+", "##_##")
    )
    processed_n1_definition = match_contingencies_column(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
        n1_column="power_factory_grid_model_name_no_space",
        element_column="grid_model_name_no_space",
    )

    processed_n1_definition.drop(
        columns=[
            "power_factory_grid_model_name_no_space",
        ],
        inplace=True,
    )
    all_element_names.drop(columns=["grid_model_name_no_space"], inplace=True)

    return processed_n1_definition


def match_contingencies_with_suffix(
    processed_n1_definition: ContingencyMatchSchema,
    all_element_names: AllGridElementsSchema,
    grid_model_suffix: list[str],
) -> ContingencyMatchSchema:
    """Match the contingencies from the file with the elements in the grid model by name.

    Matches by name and replaces the grid_model_name with power_factory_grid_model_name.
    Removes suffix from the grid_model_name.

    Parameters
    ----------
    processed_n1_definition : ContingencyMatchSchema
        The contingencies from the file.
    all_element_names : AllGridElementsSchema
        The elements in the grid model.
    grid_model_suffix : list[str]
        The suffixes to match the grid model names.

    Returns
    -------
    ContingencyMatchSchema
        A DataFrame containing the matched contingencies.
    """
    all_element_names["grid_model_name_suffix"] = all_element_names["grid_model_name"]
    for suffix in grid_model_suffix:
        cond_suffix = all_element_names["grid_model_name_suffix"].str.endswith(suffix)
        all_element_names.loc[cond_suffix, "grid_model_name_suffix"] = all_element_names.loc[
            cond_suffix, "grid_model_name_suffix"
        ].str[: -len(suffix)]

    processed_n1_definition = match_contingencies_column(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
        n1_column="power_factory_grid_model_name",
        element_column="grid_model_name_suffix",
    )

    all_element_names.drop(columns=["grid_model_name_suffix"], inplace=True)

    return processed_n1_definition


def match_contingencies_column(
    processed_n1_definition: ContingencyMatchSchema,
    all_element_names: AllGridElementsSchema,
    n1_column: str,
    element_column: str,
) -> ContingencyMatchSchema:
    """Match a column processed_n1_definition with a column from all_element_names.

    This functions matches based on 100% name match and replaces the grid_model_name, element_type and grid_model_id

    Parameters
    ----------
    processed_n1_definition : ContingencyMatchSchema
        The contingencies from the file.
    all_element_names : AllGridElementsSchema
        The elements in the grid model.
    n1_column : str
        The column name in processed_n1_definition.
    element_column : str
        The column name in all_element_names.

    Returns
    -------
    ContingencyMatchSchema
        A DataFrame containing the matched contingencies.
    """
    # merge the n1_definition with all_element_names
    processed_n1_definition = processed_n1_definition.merge(
        all_element_names,
        how="left",
        left_on=n1_column,
        right_on=element_column,
        suffixes=("", "_2"),
    )
    # get new matched elements
    cond_not_matched_elements = processed_n1_definition["grid_model_name"].isna()
    cond_name_found = ~processed_n1_definition["grid_model_name_2"].isna()
    cond_replace = cond_not_matched_elements & cond_name_found
    # replace new matched elements
    processed_n1_definition.loc[cond_replace, "grid_model_name"] = processed_n1_definition.loc[
        cond_replace, "grid_model_name_2"
    ]
    processed_n1_definition.loc[cond_replace, "element_type"] = processed_n1_definition.loc[cond_replace, "element_type_2"]
    processed_n1_definition.loc[cond_replace, "grid_model_id"] = processed_n1_definition.loc[cond_replace, "grid_model_id_2"]
    processed_n1_definition.drop(columns=["grid_model_name_2", "element_type_2", "grid_model_id_2"], inplace=True)
    processed_n1_definition = ContingencyMatchSchema.validate(processed_n1_definition)
    return processed_n1_definition
