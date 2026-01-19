# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Classes for the contingency list import from PowerFactory.

This importing has the focus on CIM bases grid models.
UCTE has not been tested.

Author:  Benjamin Petrick
Created: 2025-05-13
"""

from typing import Literal, Optional, TypeAlias

import pandera as pa
import pandera.typing as pat

GridModelTypePowerFactory: TypeAlias = Literal[
    "ElmTr2",
    "ElmLne",
    "ElmGenstat",
    "ElmLod",
    "ElmSym",
    "ElmNec",
    "ElmZpu",
    "ElmTr3",
    "ElmSind",
    "ElmTerm",
    "ElmShnt",
    "ElmVac",
]
GridElementType: TypeAlias = Literal[
    "BUS",
    "BUSBAR_SECTION",
    "LINE",
    "SWITCH",
    "TWO_WINDINGS_TRANSFORMER",
    "THREE_WINDINGS_TRANSFORMER",
    "GENERATOR",
    "LOAD",
    "SHUNT_COMPENSATOR",
    "DANGLING_LINE",
    "TIE_LINE",
]


class ContingencyImportSchemaPowerFactory(pa.DataFrameModel):
    """A ContingencyImportSchemaPowerFactory is a DataFrameModel defining the expected data of the contingency import.

    From PowerFactory:
    You may find the list of contingencies in the PowerFactory GUI under
    "Calculation > Contingency Analysis > Show Contingencies...".
    """

    index: pat.Index[pa.Int] = pa.Field(coerce=True, nullable=False)
    """The unique index of the DataFrame.
    This index is used as a unique id for the dataframe.
    """

    contingency_name: pat.Series[str] = pa.Field(coerce=True, nullable=False)
    """The id of contingency found in the contingency table.
    Attribute: "loc_name" of contingency table
    May be a multi index to group the contingencies together.
    """

    contingency_id: pat.Series[int] = pa.Field(coerce=True, nullable=False)
    """A id for the contingency.
    This id is used to group the contingencies together.
    Attribute: "number" of contingency table.
    """

    power_factory_grid_model_name: pat.Series[str] = pa.Field(coerce=True, nullable=False)
    """The name of the grid model element
    Attribute: "loc_name" of grid model element
    """

    power_factory_grid_model_fid: pat.Series[str] = pa.Field(coerce=True, nullable=True)
    """The foreign Key of the grid model element
    Attribute: "for_name" of grid model element
    Note: True spacing of FID must be kept in the string.
    """

    power_factory_grid_model_rdf_id: pat.Series[str] = pa.Field(coerce=True, nullable=True)
    """The rdf id (CIM) of the grid model element
    Attribute: "cimRdfId" of grid model element
    """

    comment: Optional[pat.Series[str]] = pa.Field(nullable=True)
    """May contain information about the contingency.
    Leave empty if not needed.
    Fill if comments or descriptions exist in the contingency table.
    """

    power_factory_element_type: Optional[pat.Series[str]] = pa.Field(
        coerce=True, nullable=True, isin=GridElementType.__args__
    )
    """The type of the contingency based on the PowerFactory type.
    Gives a hint where to look for the contingency.
    """


class AllGridElementsSchema(pa.DataFrameModel):
    """A AllGridElementsSchema is a DataFrameModel for all grid elements in the grid model.

    The grid model is loaded from the CGMES file in either PyPowsybl or Pandapower.
    """

    element_type: pat.Series[str] = pa.Field(coerce=True, nullable=True, isin=GridElementType.__args__)
    """The grid model type of the contingency. e.g. LINE, SWITCH, BUS, etc."""

    grid_model_id: pat.Series[str] = pa.Field(coerce=True, nullable=True)
    """The grid model id of the contingency. e.g. a CGMES id (cryptic number)"""

    grid_model_name: pat.Series[str] = pa.Field(coerce=True, nullable=True)
    """The grid model name of the contingency. e.g. a CGMES name (human readable name)"""


class ContingencyMatchSchema(ContingencyImportSchemaPowerFactory, AllGridElementsSchema):
    """A ContingencyMatchSchema is a DataFrameModel for matching the ContingencyImportSchema with the grid model.

    ContingencyMatchSchema is a merge of:
    ContingencyImportSchema.merge(
        AllGridElementsSchema, how="left", left_on="power_factory_grid_model_rdf_id", right_on="grid_model_id"
    )
    Note: the power_factory_grid_model_rdf_id has a leading underscore and may need modification.
    """

    # this is simply a merge of the two schemas
    # no additional fields are needed
    pass
