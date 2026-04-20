# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Shared schema for switch updates induced by a toplogy.

A switch update is a collection of switch ids in the grid model and their new open/closed state. The internal representation
of actions does not follow this structure but instead reference the action set. To enable export of the actions to other
tools, this needs to be translated into multiple formats, e.g. .dgs for powerfactory or .json for OpenRAO.
"""

import pandera as pa
import pandera.typing as pat


class SwitchUpdateSchema(pa.DataFrameModel):
    """Schema for switch update dataframes."""

    grid_model_id: pat.Series[str] = pa.Field(coerce=True, nullable=True)
    """The grid_model_id to be updated."""

    open: pat.Series[bool] = pa.Field(coerce=True, nullable=True)
    """The value to be set for the switch, True for open, False for closed."""
