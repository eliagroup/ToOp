import pandas as pd
import pandera.typing as pat
import pytest
from pandera import DataFrameModel, Index
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model


# Minimal DataFrameModel with columns only
class SimpleModel(DataFrameModel):
    a: pat.Series[int]
    b: pat.Series[float]


# DataFrameModel with a single index
class IndexModel(DataFrameModel):
    idx: pat.Index[str]
    val: pat.Series[int]


# DataFrameModel with multiple indices
class MultiIndexModel(DataFrameModel):
    idx1: pat.Index[str]
    idx2: pat.Index[int]
    val: pat.Series[int]


def test_empty_dataframe_from_simple_model():
    df = get_empty_dataframe_from_model(SimpleModel)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df.empty
    assert df.index.names == [None]
    assert df.dtypes["a"] == int
    assert df.dtypes["b"] == float


def test_empty_dataframe_from_index_model():
    df = get_empty_dataframe_from_model(IndexModel)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["val"]
    assert df.empty
    assert df.index.names == ["idx"]
    assert df.dtypes["val"] == int
    assert df.index.dtype == "object"  # Index of type str


def test_empty_dataframe_from_multiindex_model():
    df = get_empty_dataframe_from_model(MultiIndexModel)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["val"]
    assert df.empty
    assert df.index.names == ["idx1", "idx2"]
    assert df.dtypes["val"] == int
    assert df.index.get_level_values(0).dtype == "object"  # First index of type str
    assert df.index.get_level_values(1).dtype == int  # Second index of type int


def test_raises_value_error_when_index_field_missing(monkeypatch):
    # Patch __fields__ to not include any index type
    class BrokenModel(DataFrameModel):
        a: pat.Series[int]

    schema = BrokenModel.to_schema()
    monkeypatch.setattr(BrokenModel, "__fields__", {"a": (type("Fake", (), {"origin": None})(), None)})
    # Patch schema.index to be an Index instance
    monkeypatch.setattr(schema, "index", Index(int))
    with pytest.raises(ValueError, match="No index found in the DataFrameModel"):
        get_empty_dataframe_from_model(BrokenModel)
