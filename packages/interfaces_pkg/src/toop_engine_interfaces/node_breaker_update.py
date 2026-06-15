# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Low level node-breaker grid updates

An action can either be represented in a high-level view with rich semantics as in `stored_action_set.py` or in
a low level format which just includes the touched switches/psts without any higher semantics if a switch update is
part of a reassignment, breaker opening or branch disconnection.
"""

import pandera as pa
import pandera.typing as pat
from pydantic import BaseModel


class SwitchUpdateSchema(pa.DataFrameModel):
    """Schema for switch update dataframes."""

    grid_model_id: pat.Series[str] = pa.Field(coerce=True, nullable=True)
    """The grid_model_id to be updated."""

    open: pat.Series[bool] = pa.Field(coerce=True, nullable=True)
    """The value to be set for the switch, True for open, False for closed."""


class PSTUpdateSchema(pa.DataFrameModel):
    """Schema for PST tap changes"""

    grid_model_id: pat.Series[str] = pa.Field(coerce=True, nullable=True)
    """The grid model id of the PST"""

    tap: pat.Series[int] = pa.Field(coerce=True, nullable=True)
    """The tap of the trafo in the original gridmodel in absolute taps.

    In contrast to the jax-internal representation where taps always start at 0, this tap starts and ends at the integers
    defined in the gridmodel."""


class NodeBreakerUpdate(BaseModel):
    """A raw node-breaker update consisting of switch updates and PST tap updates."""

    switch_updates: pat.DataFrame[SwitchUpdateSchema]
    """Switch updates from reassignments, coupler openings and disconnections"""

    pst_updates: pat.DataFrame[PSTUpdateSchema]
    """PST tap updates"""
