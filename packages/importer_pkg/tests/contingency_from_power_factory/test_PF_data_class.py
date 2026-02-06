# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandas as pd
from toop_engine_importer.contingency_from_power_factory.power_factory_data_class import (
    AllGridElementsSchema,
    ContingencyImportSchemaPowerFactory,
    ContingencyMatchSchema,
)


# Test data
def get_example_contingency_data() -> pd.DataFrame:
    contingency_data = pd.DataFrame(
        {
            "index": [1, 2],
            "contingency_name": ["contingency1", "contingency2"],
            "contingency_id": [101, 102],
            "power_factory_grid_model_name": ["ElmLne", "ElmGenstat"],
            "power_factory_grid_model_fid": ["fid1", "fid2"],
            "power_factory_grid_model_rdf_id": ["rdf1", "rdf2"],
            "comment": ["comment1", None],
            "power_factory_element_type": ["LINE", "GENERATOR"],
        }
    )
    return contingency_data


def get_example_grid_elements_data() -> pd.DataFrame:
    grid_elements_data = pd.DataFrame(
        {
            "element_type": ["LINE", "GENERATOR"],
            "grid_model_id": ["rdf1", "rdf2"],
            "grid_model_name": ["line1", "generator1"],
        }
    )
    return grid_elements_data


# Validate the schemas
def test_contingency_import_schema():
    contingency_data = get_example_contingency_data()
    validated_df = ContingencyImportSchemaPowerFactory.validate(contingency_data)
    assert not validated_df.empty
    assert validated_df.equals(contingency_data)


def test_all_grid_elements_schema():
    grid_elements_data = get_example_grid_elements_data()
    validated_df = AllGridElementsSchema.validate(grid_elements_data)
    assert not validated_df.empty
    assert validated_df.equals(grid_elements_data)


def test_contingency_match_schema():
    contingency_data = get_example_contingency_data()
    grid_elements_data = get_example_grid_elements_data()
    merged_data = pd.merge(
        contingency_data, grid_elements_data, how="left", left_on="power_factory_grid_model_rdf_id", right_on="grid_model_id"
    )
    validated_df = ContingencyMatchSchema.validate(merged_data)
    assert not validated_df.empty
    assert validated_df.equals(merged_data)
