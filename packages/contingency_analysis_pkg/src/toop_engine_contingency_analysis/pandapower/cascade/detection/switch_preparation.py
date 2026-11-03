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
import pandera.typing as pat
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import PreparedRelays
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_interfaces.loadflow_results import SwitchResultsSchema


def prepare_switch_results_for_protection(
    net: pp.pandapowerNet,
    switch_results: pat.DataFrame[SwitchResultsSchema],
    relays: PreparedRelays,
) -> pd.DataFrame:
    """Prepare switch results for distance protection checks.

    The result has one row per relay: a breaker carrying several relays contributes its
    measurement to each of them, oriented to the side that relay measures from.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network that contains switch metadata.
    switch_results : pat.DataFrame[SwitchResultsSchema]
        Switch load-flow result table.
    relays : PreparedRelays
        The relays prepared once per run.

    Returns
    -------
    pd.DataFrame
        Relay rows with switch ids, names, origin ids, the breaker measurement and relay data.
    """
    switch_results = switch_results.reset_index(drop=False)
    switch_results["switch_id"] = switch_results["element"].str.split(SEPARATOR).str[0].astype(int)
    switch_results = switch_results[switch_results.i != 0]
    switch_results["origin_id"] = switch_results["switch_id"].map(net.switch.origin_id)
    switch_results["switch_name"] = switch_results["switch_id"].map(net.switch.name)
    # The switch state is not looked at: a breaker that is open now can be closed again during the cascade
    # (e.g. by a special protection scheme), and an open breaker has no result row to join anyway.
    relay_columns = [
        "relay_id",
        "relay_name",
        "breaker_uuid",
        "poly",
        "relay_side",
        # The factor each relay resolves to, per zone and case. prepare_relays folded
        # protection_element and the custom_* overrides into these once per job, so
        # neither is needed here. The names must match what resolve_effective_factors writes.
        "effective_alarm_basecase",
        "effective_alarm_contingency",
        "effective_warning_basecase",
        "effective_warning_contingency",
    ]
    relay_rows = switch_results.merge(relays.relays[relay_columns], left_on="origin_id", right_on="breaker_uuid")

    # Switch results are measured from the breaker's bus side. Flip them for relays on the element side,
    # so every relay reads the breaker from its own side.
    element_side = (relay_rows["relay_side"] == "element").to_numpy()
    relay_rows.loc[element_side, ["p", "q"]] = -relay_rows.loc[element_side, ["p", "q"]]
    return relay_rows


def get_complex_impedance(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute relay impedance from switch measurements.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with the following columns:

        - ``vm``: voltage magnitude in kV.
        - ``i``: current magnitude in A. Rows with ``i == 0`` produce NaN.
        - ``p``: active power in MW (used to determine the impedance angle).
        - ``q``: reactive power in Mvar (used to determine the impedance angle).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Pair ``(r, x)`` where resistance ``r`` and reactance ``x`` are both in Ohm.
        NaN is returned for rows where ``i == 0``.
    """
    v_phase_v = df["vm"] * 1000 / np.sqrt(3)
    i = df["i"].replace(0, np.nan)
    z_ohm = v_phase_v / i
    phi_rad = np.arctan2(df["q"], df["p"])
    return z_ohm * np.cos(phi_rad), z_ohm * np.sin(phi_rad)
