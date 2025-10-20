"""Models for contingency data files."""

from pandera import DataFrameModel
from pandera.typing import Index, Series


class ContingencyImportSchema(DataFrameModel):
    """Model representing the contingency data format.

    Currently this only contains data for branch elements.
    """

    mrid: Index[str]
    """The unique identifier of the element (GUID)"""
    heo_relevant: Series[bool]
    """Whether the element is relevant for the HEO"""
    contingency_case: Series[bool]
    """Whether the element should be outaged in the contingency analysis"""
    observe_std: Series[bool]
    """Whether the element is observed in the standard security analysis"""
    observe_ntc: Series[bool]
    """Whether the element is observed in the NTC analysis"""
    observe_vor: Series[bool]
    """Whether the element is observed in the VOR analysis"""
