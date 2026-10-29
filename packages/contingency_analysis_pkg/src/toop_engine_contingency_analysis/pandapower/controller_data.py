# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Access to the rows tap and Q(U) controllers record on a pandapower net.

DiscreteTapControl in the pandapower fork appends one dict per control step (and one per
finished run_control call) to ``net._controller_data_rows`` and rebuilds ``net.controller_data``
from that list only in ``finalize_control``. A run_control that raises ControllerNotConverged
never finalizes, so the list - not the DataFrame - is the complete record.

DERController (the Q(U) plant controllers) created with ``record_data=True`` appends one dict per
element and control step, plus a final one per controlled load flow, to
``net._der_controller_data_rows``; it keeps no DataFrame on the net.

Keys are read by name only, so stock pandapower, which never sets them, simply yields no rows.

Temporary, like all controller data: see TEMPORARY_CONTROLLER_DATA in toop_engine_interfaces.loadflow_results.
"""

import pandapower as pp
import pandas as pd
from beartype.typing import Final, Optional
from toop_engine_interfaces.loadflow_results import TEMPORARY_CONTROLLER_DATA
from typing_extensions import deprecated

CONTROLLER_DATA_KEY: Final[str] = "controller_data"
CONTROLLER_DATA_ROWS_KEY: Final[str] = "_controller_data_rows"
DER_CONTROLLER_DATA_ROWS_KEY: Final[str] = "_der_controller_data_rows"


@deprecated(TEMPORARY_CONTROLLER_DATA, category=None)
def reset_controller_data(net: pp.pandapowerNet) -> None:
    """Drop every tap and Q(U) controller row stored on ``net``.

    Used before an outage, because ``copy_net_for_outage`` deep-copies unknown keys and every
    outage would otherwise carry the rows its parent already held (for example from the
    base-case load flow), and before a power-flow retry, so the rows of the abandoned attempt
    do not mix with the retry's.
    """
    net.pop(CONTROLLER_DATA_KEY, None)
    net.pop(CONTROLLER_DATA_ROWS_KEY, None)
    net.pop(DER_CONTROLLER_DATA_ROWS_KEY, None)


def _read_rows(net: pp.pandapowerNet, key: str) -> Optional[pd.DataFrame]:
    rows = net.get(key)
    if not isinstance(rows, list) or not rows:
        return None
    data = pd.DataFrame(rows)
    return data


@deprecated(TEMPORARY_CONTROLLER_DATA, category=None)
def read_controller_data(net: pp.pandapowerNet) -> Optional[pd.DataFrame]:
    """Return the tap controller rows recorded on ``net`` as a DataFrame, None when there are none."""
    data = _read_rows(net, CONTROLLER_DATA_ROWS_KEY)
    return data


@deprecated(TEMPORARY_CONTROLLER_DATA, category=None)
def read_der_controller_data(net: pp.pandapowerNet) -> Optional[pd.DataFrame]:
    """Return the Q(U) controller rows recorded on ``net`` as a DataFrame, None when there are none."""
    data = _read_rows(net, DER_CONTROLLER_DATA_ROWS_KEY)
    return data
