# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Detect cascade triggers caused by distance-protection relays."""

import math

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from toop_engine_contingency_analysis.pandapower.cascade.configuration import CascadeConfig
from toop_engine_contingency_analysis.pandapower.cascade.detection.switch_preparation import (
    get_complex_impedance,
)


def _build_poly(row: pd.Series) -> Polygon:
    """Build the protection zone shape for one relay.

    Parameters
    ----------
    row : pd.Series
        Row with relay zone dimensions and angle.

    Returns
    -------
    Polygon
        Polygon that represents the relay protection area.
    """
    third_point_y = row.r_i * math.tan(row.angle)
    fourth_point_y = row.r_v * math.tan(row.angle)

    return Polygon(
        [
            (0.0, 0.0),
            (row["r_i"], 0.0),
            (row["r_i"], third_point_y),
            (row["r_v"], fourth_point_y),
            (row["r_v"], row["x_v"]),
            (0.0, row["x_v"]),
        ]
    )


def get_warning_area(
    df: pd.DataFrame,
    basecase_distance_protection_factor: float,
    contingency_distance_protection_factor: float,
) -> pd.Series:
    """Check whether each relay measurement is inside the warning area.

    The warning area is a larger version of the danger area. It marks cases
    that are not yet in the strict trip zone but are close enough to continue
    the cascade check.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with impedance values and protection polygons.
    basecase_distance_protection_factor : float
        Warning-area scale for basecase rows.
    contingency_distance_protection_factor : float
        Warning-area scale for contingency rows.

    Returns
    -------
    pd.Series
        Boolean series where True means the row is inside the warning area.
    """
    fallback_factors = np.where(
        df["contingency"] == "BASECASE",
        basecase_distance_protection_factor,
        contingency_distance_protection_factor,
    )
    factors = df["custom_warning_distance_protection"].to_numpy()
    effective_factors = np.where(pd.isna(factors), fallback_factors, factors)

    x = (np.abs(df["r_ohm"].to_numpy()) / effective_factors).astype(float)
    y = (np.abs(df["x_ohm"].to_numpy()) / effective_factors).astype(float)
    flags = shapely.covers(df["poly"].to_numpy(), shapely.points(x, y))
    return pd.Series(flags, index=df.index)


def get_danger_area(df: pd.DataFrame) -> pd.Series:
    """Check whether each relay measurement is inside the danger area.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with impedance values and protection polygons.

    Returns
    -------
    pd.Series
        Boolean series where True means the row is inside the danger area.
    """
    x = np.abs(df["r_ohm"].to_numpy())
    y = np.abs(df["x_ohm"].to_numpy())
    flags = shapely.covers(df["poly"].to_numpy(), shapely.points(x, y))
    return pd.Series(flags, index=df.index)


def evaluate_distance_protection_triggers(
    switch_results: pd.DataFrame,
    cascade_configuration: CascadeConfig,
) -> pd.DataFrame:
    """Find switches that should trip because of distance protection.

    Parameters
    ----------
    switch_results : pd.DataFrame
        Switch result table already joined with relay characteristics.
    cascade_configuration : CascadeConfig
        Cascade settings with warning-area factors.

    Returns
    -------
    pd.DataFrame
        Subset of switch_results that is inside the warning or danger area.
    """
    switch_results["r_ohm"], switch_results["x_ohm"] = get_complex_impedance(switch_results)
    switch_results["danger_inside"] = get_danger_area(switch_results)
    switch_results["warning_inside"] = get_warning_area(
        switch_results,
        basecase_distance_protection_factor=cascade_configuration.basecase_distance_protection_factor,
        contingency_distance_protection_factor=cascade_configuration.contingency_distance_protection_factor,
    )
    return switch_results[switch_results["warning_inside"] | switch_results["danger_inside"]]
