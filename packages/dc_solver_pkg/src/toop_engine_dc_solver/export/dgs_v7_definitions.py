# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines constants and structures for DGS V7 export.

This module only contains needed definitions, currently it focuses on switches.
"""

import pandas as pd
import pandera
import pandera.pandas as pa
import pandera.typing as pat
from beartype.typing import Literal

DGS_SHEETS = Literal[
    "General",
    "ElmAsm",
    "ElmCoup",
    "ElmLne",
    "ElmLnesec",
    "ElmLod",
    "ElmNet",
    "ElmShnt",
    "ElmSym",
    "ElmTerm",
    "ElmTr2",
    "ElmTr3",
    "ElmXnet",
    "IntGrf",
    "IntGrfcon",
    "IntGrfnet",
    "StaCubic",
    "StaSwitch",
]

DGS_SWITCH_OPEN = 0
DGS_SWITCH_CLOSED = 1

# The setup of the DGS General Sheet is defined in the DGS 7.0 documentation.
# this is a basic setup for using FIDs (foreign ids) as an identifier for the in the DGS_SHEETS
ID_KEY = "ID(a:40)"
VAL_KEY = "Val(a:40)"
DESC_KEY = "Descr(a:40)"
DGS_GENERAL_SHEET_CONTENT_FID = [{ID_KEY: "1", DESC_KEY: "Version", VAL_KEY: "7.0"}]

# The DGS General Sheet with where the FID is pointed to the CIM RDF ID.
DGS_GENERAL_SHEET_CONTENT_FID_CIM = [
    *DGS_GENERAL_SHEET_CONTENT_FID,
    {ID_KEY: "2", DESC_KEY: "Id1", VAL_KEY: "cimRdfId:0"},
    {ID_KEY: "3", DESC_KEY: "IdColumn", VAL_KEY: "FID"},
]


def _register_missing_builtin_pandera_checks() -> None:  # noqa: C901
    """Register missing pandas backends for builtin Pandera checks used in this module."""
    version_tuple = tuple(map(int, pandera.__version__.split(".")[:2]))

    try:
        equal_to_dispatcher = pa.Check.get_builtin_check_fn("equal_to")
    except Exception:
        equal_to_dispatcher = None

    if pd.Series not in getattr(equal_to_dispatcher, "_function_registry", {}):
        assert version_tuple < (0, 27), (
            "Remove the temporary Pandera builtin-check registration once Pandera >= 0.27 is supported."
        )

        @pa.Check.register_builtin_check_fn
        def equal_to(data: pd.Series, value: object) -> pd.Series:
            return data.isna() | (data == value)

    try:
        isin_dispatcher = pa.Check.get_builtin_check_fn("isin")
    except Exception:
        isin_dispatcher = None

    if pd.Series not in getattr(isin_dispatcher, "_function_registry", {}):
        assert version_tuple < (0, 27), (
            "Remove the temporary Pandera builtin-check registration once Pandera >= 0.27 is supported."
        )

        @pa.Check.register_builtin_check_fn
        def isin(data: pd.Series, allowed_values: list[object] | tuple[object, ...]) -> pd.Series:
            return data.isna() | data.isin(allowed_values)

    try:
        str_length_dispatcher = pa.Check.get_builtin_check_fn("str_length")
    except Exception:
        str_length_dispatcher = None

    if pd.Series not in getattr(str_length_dispatcher, "_function_registry", {}):
        assert version_tuple < (0, 27), (
            "Remove the temporary Pandera builtin-check registration once Pandera >= 0.27 is supported."
        )

        @pa.Check.register_builtin_check_fn
        def str_length(
            data: pd.Series,
            min_value: int | None = None,
            max_value: int | None = None,
        ) -> pd.Series:
            lengths = data.astype(str).str.len()
            valid = pd.Series(True, index=data.index)
            if min_value is not None:
                valid = valid & (data.isna() | (lengths >= min_value))
            if max_value is not None:
                valid = valid & (data.isna() | (lengths <= max_value))
            return valid


_register_missing_builtin_pandera_checks()


class DgsGeneralSchema(pa.DataFrameModel):
    """Schema for the DGS General sheet."""

    ID_a_40_: pat.Series[pa.String] = pa.Field(
        alias=ID_KEY,
        coerce=True,
    )
    Descr_a_40_: pat.Series[pa.String] = pa.Field(
        alias=DESC_KEY,
        coerce=True,
    )
    Val_a_40_: pat.Series[pa.String] = pa.Field(
        alias=VAL_KEY,
        coerce=True,
    )

    class Config:
        """Configuration for the DGS General schema."""

        strict = True
        coerce = True


class DgsElmCoupSchema(pa.DataFrameModel):
    """Schema for the switch updates in the DGS format."""

    FID_a_40: pat.Series[str] = pa.Field(
        alias="FID(a:40)",
        str_length={"max_value": 40},
        description="Foreign key of the switch to be updated.",
    )
    OP: pat.Series[str] = pa.Field(
        eq="U",
        description="Operation to be performed on the switch.",
    )
    on_off: pat.Series[int] = pa.Field(
        isin=[DGS_SWITCH_OPEN, DGS_SWITCH_CLOSED],
        description="Value to be set for the switch, 0 for open, 1 for closed.",
    )

    class Config:
        """Pandera Configuration for the DGS schema."""

        strict = True
        coerce = True
