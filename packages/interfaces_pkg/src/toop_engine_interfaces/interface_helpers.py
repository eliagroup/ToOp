# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module containing helper functions to interact with interfaces."""

from copy import deepcopy
from functools import lru_cache

import pandas as pd
import pandera as pa
from pandera import DataFrameModel, Index
from pandera import typing as pat


@lru_cache(maxsize=None)
def _get_empty_dataframe_from_model_cached(model: DataFrameModel) -> pat.DataFrame[DataFrameModel]:
    """Create an empty DataFrame based on the provided DataFrameModel and cache the result.

    DO NOT CALL THIS FUNCTION DIRECTLY
        Use get_empty_dataframe_from_model instead.
        Otherwise calls to loc may override the cache.

    Parameters
    ----------
    model : DataFrameModel
        The DataFrameModel from which to create the empty DataFrame.

    Returns
    -------
    pd.DataFrame
        An empty DataFrame with the correct index and columns as defined in the model.

    Raises
    ------
    ValueError
        If no index is found in the DataFrameModel, but it has been defined.
    """
    schema = model.to_schema()
    index = schema.index
    if index is None:
        indizes = {}
    elif isinstance(index, Index):
        for field, (annotation, _) in model.__fields__.items():
            if annotation.origin in pat.get_index_types():
                indizes = {field: index._dtype}
                break
        else:
            raise ValueError("No index found in the DataFrameModel. Please ensure that at least one field is an index type.")
    else:
        indizes = schema.index.dtypes
    columns = schema.dtypes
    all_columns = {**indizes, **columns}

    df = pd.DataFrame(columns=all_columns.keys()).astype({col: str(dtype) for col, dtype in all_columns.items()})
    if indizes:
        df = df.set_index(list(indizes.keys()))
    return df


@pa.check_types
def get_empty_dataframe_from_model(model: DataFrameModel) -> pat.DataFrame[DataFrameModel]:
    """Create an empty DataFrame based on the provided DataFrameModel.

    Parameters
    ----------
    model : DataFrameModel
        The DataFrameModel from which to create the empty DataFrame.

    Returns
    -------
    pd.DataFrame
        An empty DataFrame with the correct index and columns as defined in the model.

    Raises
    ------
    ValueError
        If no index is found in the DataFrameModel, but it has been defined.
    """
    df = _get_empty_dataframe_from_model_cached(model)
    return deepcopy(df)
