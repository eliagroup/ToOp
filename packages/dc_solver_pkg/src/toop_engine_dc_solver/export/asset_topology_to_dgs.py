# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module containing functions to translate asset topology model to the DGS format.

File: asset_topology_to_dgs.py
Author:  Benjamin Petrick
Created: 2024-11-12

Note: this module currently ignores the asset_setpoints.
"""

import io
from copy import deepcopy

import pandas as pd
import pandera as pa
import pandera.typing as pat
from beartype.typing import Any, Optional, cast
from toop_engine_dc_solver.export.dgs_v7_definitions import (
    DGS_GENERAL_SHEET_CONTENT_FID,
    DGS_GENERAL_SHEET_CONTENT_FID_CIM,
    DGS_SHEETS,
    DgsElmCoupSchema,
    DgsGeneralSchema,
)
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


class ForeignIdSchema(pa.DataFrameModel):
    """Schema for the switch update DataFrame."""

    grid_model_id: pat.Series[str] = pa.Field(coerce=True)
    """ The grid_model_id to be updated."""

    foreign_id: pat.Series[str] = pa.Field(coerce=True)
    """ The foreign id e.g. from PowerFactory."""


@pa.check_types
def switch_update_schema_to_dgs(
    switch_update_schema: pat.DataFrame[SwitchUpdateSchema],
    foreign_ids: Optional[pat.DataFrame[ForeignIdSchema]] = None,
    cim: bool = True,
) -> pat.DataFrame[DgsElmCoupSchema]:
    """Translate the switch update schema to the DGS format.

    Provide a ForeignIdSchema to update translate the grid_model_id to a foreign id.

    Parameters
    ----------
    switch_update_schema : pat.DataFrame[SwitchUpdateSchema]
        The switch update schema to be translated
    foreign_ids: Optional[pat.DataFrame[ForeignIdSchema]]
        The foreign ids to be used for the translation
    cim: bool, default True
        If True, the ids are expected to be in the CIM format -> a missing underscore at the beginning of the id.
        and RDF_ID is used as the foreign key.
        If False, the ids are expected to be in the DGS FID format.

    Returns
    -------
    switch_update_schema: pat.DataFrame[DgsElmCoupSchema]
        The switch update schema in the DGS format
    """
    dgs_df = deepcopy(switch_update_schema)
    if foreign_ids is not None:
        dgs_df = dgs_df.merge(
            foreign_ids,
            left_on="grid_model_id",
            right_on="grid_model_id",
            how="left",
        )
        assert dgs_df["foreign_id"].notna().all(), "Not all grid_model_ids have a foreign id"
        dgs_df.rename(columns={"foreign_id": "FID(a:40)", "open": "on_off"}, inplace=True)
        dgs_df.drop(columns=["grid_model_id"], inplace=True)
    else:
        # if no foreign ids are given, we use the grid_model_id as the FID(a:40)
        dgs_df.rename(columns={"grid_model_id": "FID(a:40)", "open": "on_off"}, inplace=True)

    dgs_df["OP"] = "U"
    # dgs on_off is inverted to the powsybl model
    # dgs: 0 for open, 1 for closed
    dgs_df["on_off"] = ~dgs_df["on_off"]
    dgs_df = dgs_df.astype({"FID(a:40)": str, "OP": str, "on_off": int})
    if cim:
        dgs_df["FID(a:40)"] = "_" + dgs_df["FID(a:40)"]
    return cast(pat.DataFrame[DgsElmCoupSchema], dgs_df)


@pa.check_types
def get_dgs_general_schema(
    cim: bool = True,
    general_info: Optional[list[dict[str, str]]] = None,
) -> pat.DataFrame[DgsGeneralSchema]:
    """Get the DGS General schema."""
    if general_info is None and cim:
        general_info = DGS_GENERAL_SHEET_CONTENT_FID_CIM
    elif general_info is None and not cim:
        general_info = DGS_GENERAL_SHEET_CONTENT_FID

    df_general = pd.DataFrame(general_info)
    return cast(pat.DataFrame[DgsGeneralSchema], df_general)


@pa.check_types
def switch_dgs_schema_to_xlsx(
    switch_dgs_schema: pat.DataFrame[DgsElmCoupSchema],
    df_general: pat.DataFrame[DgsGeneralSchema],
    file_name: str,
    sheet_name: DGS_SHEETS = "ElmCoup",
) -> None:
    """Write the DGS information to an xlsx file.

    This is the DGS dump function for the switch update schema.
    Consider only dumping changes to the network, not the whole network.
    use get_changing_switches_from_topology() to get the changes.

    Parameters
    ----------
    switch_dgs_schema : pat.DataFrame[DgsElmCoupSchema]
        The switch update schema in the DGS format
    df_general : pat.DataFrame[DgsGeneralSchema]
        The general information for the DGS format
    file_name : str
        Name of the xlsx file
    sheet_name : DGS_SHEETS
        Name of the sheet in the xlsx file
        The DGS format uses predefined sheet names defined in DGS_SHEETS

    Returns
    -------
    None
    """
    with pd.ExcelWriter(file_name) as writer:
        df_general.to_excel(writer, index=False, sheet_name="General")
        switch_dgs_schema.to_excel(writer, index=False, sheet_name=sheet_name)


@pa.check_types
def switch_dgs_schema_to_bytes_io(
    switch_dgs_schema: pat.DataFrame[DgsElmCoupSchema],
    df_general: pat.DataFrame[DgsGeneralSchema],
    sheet_name: DGS_SHEETS = "ElmCoup",
) -> io.BytesIO:
    """Write the DGS information to a BytesIO object.

    This is the DGS dump function for the switch update schema.
    Consider only dumping changes to the network, not the whole network.
    use get_changing_switches_from_topology() to get the changes.

    Parameters
    ----------
    switch_dgs_schema : pat.DataFrame[DgsElmCoupSchema]
        The switch update schema in the DGS format
    df_general : pat.DataFrame[DgsGeneralSchema]
        The general information for the DGS format
    sheet_name : DGS_SHEETS
        Name of the sheet in the xlsx file
        The DGS format uses predefined sheet names defined in DGS_SHEETS

    Returns
    -------
    bytes_io: io.BytesIO
        BytesIO object containing the xlsx file with the DGS information
    """
    bytes_io = io.BytesIO()
    with pd.ExcelWriter(cast(Any, bytes_io), engine="openpyxl") as writer:
        df_general.to_excel(writer, index=False, sheet_name="General")
        switch_dgs_schema.to_excel(writer, index=False, sheet_name=sheet_name)
    bytes_io.seek(0)
    return bytes_io
