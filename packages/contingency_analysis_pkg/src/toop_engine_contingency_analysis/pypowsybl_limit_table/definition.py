# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Translate the N-1 definition for the limit table contingency analysis.

Contingencies are handed to powsybl one to one, and every violation of a contingency comes from the limit violation
table. The monitored elements are translated exactly like for the monitored element runner, so the base case results
of both runners match. They also decide which violations are kept: those of monitored branches and of the voltage
levels of monitored buses.
"""

import pandas as pd
import structlog
from beartype.typing import Optional, get_args
from pydantic import BaseModel
from pypowsybl.network import Network
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers import (
    PowsyblBranchLimitCacheProtocol,
    PowsyblContingency,
    PowsyblMonitoredElements,
    get_busbar_mapping,
    get_monitoring_frames_powsybl,
    resolve_branch_limits_for_powsybl,
    translate_contingency_to_powsybl,
    translate_monitored_elements_to_powsybl,
)
from toop_engine_interfaces.nminus1_definition import (
    POWSYBL_SUPPORTED_ID_TYPES,
    Contingency,
    GridElement,
    Nminus1Definition,
)

logger = structlog.get_logger(__name__)


class PowsyblLimitTableDefinition(BaseModel):
    """The N-1 definition translated for the limit table contingency analysis.

    Holds the contingencies as powsybl receives them plus the network data the result extraction needs. The monitored
    elements are monitored for the base case only, contingency results come from the limit violation table.
    """

    model_config = {"arbitrary_types_allowed": True}

    contingencies: list[PowsyblContingency]
    """The contingencies to run, translated one to one. A contingency without elements is the base case."""

    missing_contingencies: list[Contingency] = []
    """Contingencies whose elements are (partially) not present in the network. They are not run."""

    monitored_elements: PowsyblMonitoredElements
    """The monitored elements as translated for the monitored element runner, see
    translate_monitored_elements_to_powsybl. They are monitored for the base case and filter the violations."""

    missing_elements: list[GridElement] = []
    """Monitored elements that are not present in the network."""

    branch_limits: pd.DataFrame
    """The prepared current limits of the monitored two sided branches, indexed by element_id and side. The column
    ``value`` holds the permanent limit, plus one column per known limit name."""

    bus_map: pd.DataFrame
    """A mapping from busbar sections and bus breaker buses to bus breaker buses, electrical buses and voltage levels."""

    voltage_levels: pd.DataFrame
    """The voltage levels of the network with nominal_v, high_voltage_limit and low_voltage_limit, indexed by id."""

    element_name_mapping: dict[str, str]
    """A mapping from monitored element ids to their names, used to fill the element_name columns of the results."""

    contingency_name_mapping: dict[str, str]
    """A mapping from contingency ids to their names, used to fill the contingency_name columns of the results."""

    @property
    def basecase_id(self) -> Optional[str]:
        """The id of the base case contingency, or None if the definition has none."""
        return next((contingency.id for contingency in self.contingencies if contingency.is_basecase()), None)

    @property
    def outage_ids(self) -> list[str]:
        """The ids of all contingencies that outage something, in definition order."""
        return [contingency.id for contingency in self.contingencies if not contingency.is_basecase()]

    @property
    def monitored_branch_ids(self) -> list[str]:
        """The monitored two sided branches and three winding transformers, whose violations are kept."""
        return [*self.monitored_elements["branches"], *self.monitored_elements["trafo3w"]]

    @property
    def monitored_bus_voltage_level_ids(self) -> list[str]:
        """The voltage levels holding a monitored bus, whose voltage violations are kept."""
        return self.bus_map.loc[self.monitored_elements["buses"], "voltage_level_id"].astype(str).unique().tolist()


def translate_nminus1_for_limit_table(
    nminus1_definition: Nminus1Definition,
    net: Network,
    branch_limit_cache: Optional[PowsyblBranchLimitCacheProtocol] = None,
) -> PowsyblLimitTableDefinition:
    """Translate the N-1 definition for the limit table contingency analysis.

    The monitored elements are translated like for the monitored element runner. Bus contingencies are not expanded
    over busbar sections here: they are run as defined, see propagate_busbar_contingencies.

    Parameters
    ----------
    nminus1_definition : Nminus1Definition
        The generic N-1 definition to translate.
    net : Network
        The network the analysis runs on. Only read.
    branch_limit_cache : Optional[PowsyblBranchLimitCacheProtocol], optional
        Optional cache for the prepared branch limits. If stale, it is ignored.

    Returns
    -------
    PowsyblLimitTableDefinition
        The translated definition.

    Raises
    ------
    ValueError
        If the definition uses an id type powsybl does not support.
    """
    if nminus1_definition.id_type not in (supported_ids := get_args(POWSYBL_SUPPORTED_ID_TYPES)):
        raise ValueError(
            f"Unsupported id_type {nminus1_definition.id_type}. Only {supported_ids} are supported for Powsybl."
        )
    if nminus1_definition.spps_rules:
        logger.warning("powsybl_spps_rules_ignored", n_spps_rules=len(nminus1_definition.spps_rules))

    identifiables = net.get_identifiables(attributes=[]).index
    contingencies, missing_contingencies = translate_contingency_to_powsybl(nminus1_definition.contingencies, identifiables)

    bus_map = get_busbar_mapping(net)
    _, all_branches, switches = get_monitoring_frames_powsybl(net)
    monitored_elements, element_name_mapping, missing_elements = translate_monitored_elements_to_powsybl(
        nminus1_definition, all_branches, bus_map, switches
    )
    branch_limits = resolve_branch_limits_for_powsybl(
        net, monitored_branches=monitored_elements["branches"], branch_limit_cache=branch_limit_cache
    )

    voltage_levels = net.get_voltage_levels(attributes=["nominal_v", "high_voltage_limit", "low_voltage_limit"])
    voltage_levels["high_voltage_limit"] = voltage_levels["high_voltage_limit"].fillna(voltage_levels["nominal_v"] * 1.2)
    voltage_levels["low_voltage_limit"] = voltage_levels["low_voltage_limit"].fillna(voltage_levels["nominal_v"] * 0.8)
    contingency_name_mapping = {contingency.id: contingency.name or "" for contingency in nminus1_definition.contingencies}

    return PowsyblLimitTableDefinition(
        contingencies=contingencies,
        missing_contingencies=missing_contingencies,
        monitored_elements=monitored_elements,
        missing_elements=missing_elements,
        branch_limits=branch_limits,
        bus_map=bus_map,
        voltage_levels=voltage_levels,
        element_name_mapping=element_name_mapping,
        contingency_name_mapping=contingency_name_mapping,
    )
