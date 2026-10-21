# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Ids of the two-winding legs of a converted three-winding transformer.

``pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers`` replaces a three-winding
transformer ``<id>`` by the two-winding legs ``<id>-Leg1``, ``<id>-Leg2`` and ``<id>-Leg3`` around a star bus.
This module is the single place in ToOp that knows this naming.
"""

import re

# The id suffixes of the three legs, in leg order
TRAFO3W_LEG_SUFFIXES = ("-Leg1", "-Leg2", "-Leg3")

# Regex matching the id suffix of a leg, for use with ``re`` and the pandas ``str`` accessor
TRAFO3W_LEG_PATTERN = "-Leg[123]$"


def get_trafo3w_leg_ids(trafo3w_id: str) -> list[str]:
    """Get the ids of the three legs of a converted three-winding transformer.

    Parameters
    ----------
    trafo3w_id : str
        The id of the three-winding transformer before conversion.

    Returns
    -------
    list[str]
        The leg ids, in leg order.
    """
    return [f"{trafo3w_id}{suffix}" for suffix in TRAFO3W_LEG_SUFFIXES]


def get_trafo3w_id(leg_id: str) -> str | None:
    """Get the id of the three-winding transformer that a leg was converted from.

    Parameters
    ----------
    leg_id : str
        The id of a two-winding transformer.

    Returns
    -------
    str | None
        The three-winding transformer id, or None if ``leg_id`` is not a leg id.
    """
    trafo3w_id, n_replaced = re.subn(TRAFO3W_LEG_PATTERN, "", leg_id)
    return trafo3w_id if n_replaced else None
