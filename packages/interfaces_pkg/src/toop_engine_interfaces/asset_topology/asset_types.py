# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Shared type aliases for asset topology models."""

from beartype.typing import Literal, TypeAlias

BranchEnd: TypeAlias = Literal["from", "to", "hv", "mv", "lv"]
AssetBranchTypePandapower: TypeAlias = Literal[
    "line",
    "trafo",
    "trafo3w_lv",
    "trafo3w_mv",
    "trafo3w_hv",
    "impedance",
    "xward",
]
AssetBranchTypePowsybl: TypeAlias = Literal[
    "LINE",
    "TWO_WINDINGS_TRANSFORMER",
    "TIE_LINE",
]
AssetBranchType: TypeAlias = Literal[AssetBranchTypePandapower, AssetBranchTypePowsybl]

AssetInjectionTypePandapower: TypeAlias = Literal[
    "ext_grid",
    "gen",
    "load",
    "shunt",
    "sgen",
    "ward",
    "ward_load",
    "ward_shunt",
    "xward_load",
    "xward_shunt",
    "dcline_from",
    "dcline_to",
]
AssetInjectionTypePowsybl: TypeAlias = Literal[
    "LOAD",
    "GENERATOR",
    "BOUNDARY_LINE",
    "HVDC_CONVERTER_STATION",
    "STATIC_VAR_COMPENSATOR",
    "SHUNT_COMPENSATOR",
    "BATTERY",
]
AssetInjectionType: TypeAlias = Literal[AssetInjectionTypePandapower, AssetInjectionTypePowsybl]
AssetType: TypeAlias = Literal[AssetBranchType, AssetInjectionType]
