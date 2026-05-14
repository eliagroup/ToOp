# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Prepare switch load-flow results for cascade relay checks."""

import numpy as np
import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeContext
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR


def prepare_switch_results_for_protection(
    net: pp.pandapowerNet,
    switch_results: pd.DataFrame,
    cascade_context: CascadeContext,
) -> pd.DataFrame:
    """Prepare switch results for distance protection checks.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network that contains switch metadata.
    switch_results : pd.DataFrame
        Switch load-flow result table.
    cascade_context : CascadeContext
        Precomputed cascade data with relay characteristics.

    Returns
    -------
    pd.DataFrame
        Switch result rows with switch ids, names, origin ids, and relay data.
    """
    switch_results = switch_results.reset_index(drop=False)
    switch_results["switch_id"] = switch_results["element"].str.split(SEPARATOR).str[0].astype(int)
    switch_results = switch_results[switch_results.i != 0]
    switch_results["origin_id"] = switch_results["switch_id"].map(net.switch.origin_id)
    switch_results["switch_name"] = switch_results["switch_id"].map(net.switch.name)
    return switch_results.merge(cascade_context.switch_characteristics, on="origin_id")


def get_complex_impedance(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute relay impedance from switch measurements.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with voltage, current, active power, and reactive power.

    Returns
    -------
    tuple
        Pair of series-like values: resistance and reactance.
    """
    v_phase_kv = df["vm"] / np.sqrt(3)
    z_ohm = v_phase_kv / df["i"]
    phi_rad = np.arctan2(df["q"], df["p"])
    return z_ohm * np.cos(phi_rad), z_ohm * np.sin(phi_rad)
