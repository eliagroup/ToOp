# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import logbook
import numpy as np
import pandas as pd
import pandera
import pandera.errors
import pytest
from toop_engine_importer.contingency_from_power_factory.contingency_from_file import (
    get_contingencies_from_file,
    match_contingencies,
    match_contingencies_by_index,
    match_contingencies_by_name,
    match_contingencies_column,
    match_contingencies_with_suffix,
)
from toop_engine_importer.contingency_from_power_factory.power_factory_data_class import (
    AllGridElementsSchema,
    ContingencyImportSchemaPowerFactory,
    ContingencyMatchSchema,
)


def get_example_contingency_data() -> pd.DataFrame:
    """Get the example contingency data.

    Returns
    -------
    pd.DataFrame
        The example contingency data.
    """
    df = pd.DataFrame(
        [
            {
                "contingency_name": "single_contingency_no_additional_info",
                "contingency_id": 1.0,
                "power_factory_grid_model_name": "single_contingency_no_additional_info",
                "power_factory_grid_model_fid": "",
                "power_factory_grid_model_rdf_id": "",
                "contingency_element_type": "ElmLne",
            },
            {
                "contingency_name": "single_contingency",
                "contingency_id": 2.0,
                "power_factory_grid_model_name": "Line1",
                "power_factory_grid_model_fid": "Line1   with multi    spacing",
                "power_factory_grid_model_rdf_id": "_aa111a1-2b22-3c33-d44d-e555ee5ee555",
                "contingency_element_type": "ElmLne",
            },
            {
                "contingency_name": "multi_contingency",
                "contingency_id": 3.0,
                "power_factory_grid_model_name": "Line1",
                "power_factory_grid_model_fid": "Line1   with multi    spacing",
                "power_factory_grid_model_rdf_id": "_aa111a1-2b22-3c33-d44d-e555ee5ee555",
                "contingency_element_type": "ElmLne",
            },
            {
                "contingency_name": "multi_contingency",
                "contingency_id": 3.0,
                "power_factory_grid_model_name": "Line2",
                "power_factory_grid_model_fid": "Line2   with multi    spacing",
                "power_factory_grid_model_rdf_id": "_aa111a1-2b22-3c33-d44d-e666ee6ee666",
                "contingency_element_type": "ElmLne",
            },
        ]
    )

    return df


def get_example_all_element_names() -> pd.DataFrame:
    """Get the example grid elements data.

    Returns
    -------
    pd.DataFrame
        The example grid elements data.
    """
    df = pd.DataFrame(
        [
            {"element_type": "LINE", "grid_model_id": "_aa111a1-2b22-3c33-d44d-e555ee5ee555", "grid_model_name": "Line1"},
            {"element_type": "LINE", "grid_model_id": "id2", "grid_model_name": "Line2"},
            {"element_type": "LINE", "grid_model_id": "id3", "grid_model_name": "single_contingency_no+additional_info"},
        ]
    )

    return df


def get_contingency_test_data() -> tuple[ContingencyImportSchemaPowerFactory, AllGridElementsSchema, ContingencyMatchSchema]:
    """Get the example contingency data and grid elements data.

    Returns
    -------
    tuple[ContingencyImportSchema, AllGridElementsSchema, ContingencyMatchSchema]
        A tuple containing the example contingency data, grid elements data, and the processed contingency data.
    """
    # Example contingency data
    contingency_data = get_example_contingency_data()
    all_element_names = get_example_all_element_names()

    # Validate schemas
    contingency_data = ContingencyImportSchemaPowerFactory.validate(contingency_data)
    all_element_names = AllGridElementsSchema.validate(all_element_names)

    processed_n1_definition = contingency_data.merge(
        all_element_names, how="left", left_on="power_factory_grid_model_rdf_id", right_on="grid_model_id"
    )
    processed_n1_definition = ContingencyMatchSchema.validate(processed_n1_definition)

    return contingency_data, all_element_names, processed_n1_definition


def test_get_contingencies_from_file(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test the get_contingencies_from_file function."""
    contingency_data = get_example_contingency_data()
    # get a temporary file
    tmp_path = tmp_path_factory.mktemp("get_contingencies_from_file")
    file_path = tmp_path / "test_contingency.csv"
    contingency_data.to_csv(file_path, index=False, sep=";")

    # Call the function to test
    result = get_contingencies_from_file(file_path, delimiter=";")
    expected_df = contingency_data.copy()
    expected_df.loc[expected_df["contingency_id"] == 1, "power_factory_grid_model_name"] = expected_df.loc[
        expected_df["contingency_id"] == 1, "contingency_name"
    ]
    expected_df.loc[expected_df["contingency_id"] == 1, "power_factory_grid_model_fid"] = np.nan
    expected_df.loc[expected_df["contingency_id"] == 1, "power_factory_grid_model_rdf_id"] = np.nan
    expected_df["contingency_id"] = expected_df["contingency_id"].astype(int)
    # Check if the result is as expected
    assert result.equals(expected_df), "The contingency data does not match the expected data."

    contingency_data.loc[contingency_data["contingency_id"] == 1, "contingency_name"] = 111
    contingency_data.to_csv(file_path, index=False, sep=";")
    result = get_contingencies_from_file(file_path, delimiter=";")
    expected_df.loc[expected_df["contingency_id"] == 1, "contingency_name"] = "111"
    assert result.equals(expected_df), "The contingency data does not match the expected data."

    # Test missing columns needed for the function to work
    contingency_data = get_example_contingency_data()
    contingency_data.drop(columns=["contingency_name"], inplace=True)
    contingency_data.to_csv(file_path, index=False, sep=";")
    with pytest.raises(KeyError):
        get_contingencies_from_file(file_path, delimiter=";")

    contingency_data = get_example_contingency_data()
    contingency_data.drop(columns=["power_factory_grid_model_name"], inplace=True)
    contingency_data.to_csv(file_path, index=False, sep=";")
    with pytest.raises(KeyError):
        get_contingencies_from_file(file_path, delimiter=";")

    contingency_data = get_example_contingency_data()
    contingency_data.drop(columns=["contingency_id"], inplace=True)
    contingency_data.to_csv(file_path, index=False, sep=";")
    with pytest.raises(KeyError):
        get_contingencies_from_file(file_path, delimiter=";")

    # test missing random missing column for schmema
    contingency_data = get_example_contingency_data()
    contingency_data.drop(columns=["power_factory_grid_model_rdf_id"], inplace=True)
    contingency_data.to_csv(file_path, index=False, sep=";")
    with pytest.raises(pandera.errors.SchemaError):
        get_contingencies_from_file(file_path, delimiter=";")


def test_match_contingencies() -> None:
    """Test the match_contingencies function."""
    # Example contingency data
    contingency_data, all_element_names, _ = get_contingency_test_data()

    res = match_contingencies(
        n1_definition=contingency_data,
        all_element_names=all_element_names,
        match_by_name=False,
    )
    assert isinstance(res, pd.DataFrame)
    assert list(res["grid_model_name"].values) == [np.nan, "Line1", "Line1", np.nan]
    res = match_contingencies(
        n1_definition=contingency_data,
        all_element_names=all_element_names,
        match_by_name=True,
    )
    # the first one does not match
    assert list(res["grid_model_name"].values) == [np.nan, "Line1", "Line1", "Line2"]


def test_match_contingencies_by_index() -> None:
    """Test the match_contingencies_by_index function."""
    with logbook.handlers.TestHandler() as caplog:
        # Example contingency data
        contingency_data, all_element_names, _ = get_contingency_test_data()

        # Test matching by index
        res = match_contingencies_by_index(
            n1_definition=contingency_data,
            all_element_names=all_element_names,
        )
        assert list(res["grid_model_name"].values) == [np.nan, "Line1", "Line1", np.nan]

        contingency_data = get_example_contingency_data()
        all_element_names = get_example_all_element_names()
        # test underscore in rfd_id
        all_element_names["grid_model_id"] = all_element_names["grid_model_id"].str[1:]
        res = match_contingencies_by_index(
            n1_definition=contingency_data,
            all_element_names=all_element_names,
        )
        assert list(res["grid_model_name"].values) == [np.nan, "Line1", "Line1", np.nan]

        contingency_data = get_example_contingency_data()
        all_element_names = get_example_all_element_names()
        # no ids found -> warning
        all_element_names["grid_model_id"] = all_element_names["grid_model_id"].str[2:]
        res = match_contingencies_by_index(
            n1_definition=contingency_data,
            all_element_names=all_element_names,
        )
        assert list(res["grid_model_name"].values) == [np.nan, np.nan, np.nan, np.nan]
        assert "No elements found in the grid model via CIM id. Check the grid model and the contingency file." in "".join(
            caplog.formatted_records
        )


def test_match_contingencies_by_name() -> None:
    """Test the match_contingencies_by_name function."""
    # Example contingency data
    _, all_element_names, processed_n1_definition = get_contingency_test_data()

    # Test matching by name
    res = match_contingencies_by_name(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
    )
    assert list(res["grid_model_name"].values) == [np.nan, "Line1", "Line1", "Line2"]
    processed_n1_definition.loc[processed_n1_definition["contingency_id"] == 1, "power_factory_grid_model_name"] = (
        "single_contingency_no+additional_info"
    )
    res = match_contingencies_by_name(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
    )
    assert list(res["grid_model_name"].values) == ["single_contingency_no+additional_info", "Line1", "Line1", "Line2"]

    # test that the df has not changed
    _, all_element_names2, processed_n1_definition2 = get_contingency_test_data()
    assert all_element_names2.equals(all_element_names)


def test_match_contingencies_column():
    """Test the match_contingencies function with a column that is not in the schema."""
    _, all_element_names, processed_n1_definition = get_contingency_test_data()

    processed_n1_definition = match_contingencies_column(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
        n1_column="power_factory_grid_model_name",
        element_column="grid_model_name",
    )

    assert list(processed_n1_definition["grid_model_name"].values) == [np.nan, "Line1", "Line1", "Line2"]

    _, all_element_names2, _ = get_contingency_test_data()
    assert all_element_names2.equals(all_element_names)


def test_match_contingencies_with_suffix() -> None:
    # Example contingency data
    contingency_data, all_element_names, processed_n1_definition = get_contingency_test_data()

    processed_n1_definition = ContingencyMatchSchema.validate(processed_n1_definition)
    suffix = ["RANDOM"]
    # Test matching by name
    res = match_contingencies_with_suffix(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
        grid_model_suffix=suffix,
    )
    assert list(res["grid_model_name"].values) == [np.nan, "Line1", "Line1", "Line2"]
    _, all_element_names2, processed_n1_definition2 = get_contingency_test_data()
    assert all_element_names2.equals(all_element_names)

    suffix = ["-3_w_trafo_leg_1", "-3_w_trafo_leg_2", "-3_w_trafo_leg_3"]
    all_element_names["grid_model_name"] = all_element_names["grid_model_name"].astype(str) + suffix[0]
    # get new processed_n1_definition with no match
    processed_n1_definition = contingency_data.merge(
        all_element_names, how="left", left_on="power_factory_grid_model_name", right_on="grid_model_name"
    )
    assert list(processed_n1_definition["grid_model_name"].values) == [np.nan, np.nan, np.nan, np.nan]
    # test that no match is found if suffix is wrong
    res = match_contingencies_with_suffix(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
        grid_model_suffix=["NOT_FOUND"],
    )
    assert list(res["grid_model_name"].values) == [np.nan, np.nan, np.nan, np.nan]
    res = match_contingencies_with_suffix(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
        grid_model_suffix=suffix,
    )
    assert list(res["grid_model_name"].values) == [
        np.nan,
        "Line1-3_w_trafo_leg_1",
        "Line1-3_w_trafo_leg_1",
        "Line2-3_w_trafo_leg_1",
    ]
