# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""String enums for SpPS (Special Protection Scheme) parameters and table columns.

Values match the wire / DataFrame / JSON string literals used in pandapower
and Pandera schemas. Subclass :class:`str` so members compare equal to their
string values and serialize as plain strings in Pydantic.
"""

from enum import StrEnum


def _values(e: type[StrEnum]) -> tuple[str, ...]:
    return tuple(m.value for m in e)


class SppsConditionType(StrEnum):
    """``condition_type`` in :class:`toop_engine_interfaces.nminus1_definition.Condition`."""

    CURRENT = "current"
    ACTIVE_POWER = "active_power"
    REACTIVE_POWER = "reactive_power"
    VOLTAGE = "voltage"
    STATE = "state"


class SppsConditionCheckType(StrEnum):
    """``condition_check_type`` in :class:`toop_engine_interfaces.nminus1_definition.Condition`."""

    GT = ">"
    LT = "<"
    EQ = "="
    FAILED = "failed"
    DE_ENERGIZED = "de_energized"


class SppsConditionSide(StrEnum):
    """``condition_side`` in :class:`toop_engine_interfaces.nminus1_definition.Condition`."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    MAXIMUM_VALUE = "maximum_value"


class SppsMeasureType(StrEnum):
    """``measure_type`` in :class:`toop_engine_interfaces.nminus1_definition.Action`."""

    ACTIVE_POWER = "active_power"
    REACTIVE_POWER = "reactive_power"
    VOLTAGE = "voltage"
    SWITCHING_STATE = "switching_state"


class SppsPowerFlowFailurePolicy(StrEnum):
    """In-loop power-flow error handling for :func:`run_spps` and contingency config."""

    RAISE = "raise"
    KEEP_PREVIOUS = "keep_previous"


class SppsSwitchActionTarget(StrEnum):
    """Lowercase wire values for switch ``switching_state`` (``measure_value`` strings)."""

    OPEN = "open"
    CLOSED = "closed"


# For Pandera ``isin=`` and similar (stable order)
SPPS_CONDITION_TYPE_VALUES: tuple[str, ...] = _values(SppsConditionType)
SPPS_CONDITION_CHECK_TYPE_VALUES: tuple[str, ...] = _values(SppsConditionCheckType)
SPPS_CONDITION_SIDE_VALUES: tuple[str, ...] = _values(SppsConditionSide)
SPPS_MEASURE_TYPE_VALUES: tuple[str, ...] = _values(SppsMeasureType)
