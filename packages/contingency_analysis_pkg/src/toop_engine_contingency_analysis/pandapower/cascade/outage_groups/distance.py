# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Build cascade outage groups caused by distance-protection trips."""

import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower.cascade.outage_groups.topology import get_outage_group_for_elements


def compute_relays_outage_group(
    net: pp.pandapowerNet,
    tripped_relays: pd.DataFrame,
    relay_targets: pd.DataFrame,
) -> dict[str, list[tuple[int, str]]]:
    """Build the outage group of every tripped relay from the elements it protects.

    A trip takes the protected elements out the way a contingency on them would: each of them is
    expanded to the whole contingency outage group it belongs to. The breaker and the side of it
    the relay is on do not matter here.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network to inspect.
    tripped_relays : pd.DataFrame
        Relay rows selected by the distance protection check.
    relay_targets : pd.DataFrame
        Protected elements of every relay (``relay_id``, ``table``, ``table_id``).

    Returns
    -------
    dict[str, list[tuple[int, str]]]
        Mapping from relay id to the elements that relay takes out.
    """
    if tripped_relays is None or tripped_relays.empty:
        return {}

    targets = relay_targets[relay_targets["relay_id"].isin(tripped_relays["relay_id"])]
    elements = pd.Series(list(zip(targets["table_id"], targets["table"], strict=True)), index=targets["relay_id"])
    protected_elements = elements.groupby(level=0).agg(list).to_dict()
    return get_outage_group_for_elements(net, protected_elements)
