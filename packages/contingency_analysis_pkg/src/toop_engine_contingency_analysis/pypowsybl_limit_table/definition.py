# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Translate the N-1 definition for the limit table contingency analysis.

The limit table analysis does not monitor elements. Contingencies are handed to powsybl one to one, and every branch
result of a contingency comes from the limit violation table. The only network data collected here is what the
extraction needs: the current limits of all branches, the bus map and the voltage levels for the fully monitored base
case, and the element ids to register for that base case.
"""

from enum import Enum

import pandas as pd
import structlog
from beartype.typing import Optional, get_args
from pydantic import BaseModel
from pypowsybl.network import Network
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers import (
    PowsyblBranchLimitCacheProtocol,
    PowsyblContingency,
    get_busbar_mapping,
    resolve_branch_limits_for_powsybl,
    translate_contingency_to_powsybl,
)
from toop_engine_interfaces.nminus1_definition import (
    POWSYBL_SUPPORTED_ID_TYPES,
    Contingency,
    Nminus1Definition,
)

logger = structlog.get_logger(__name__)


class LimitViolationFilter(str, Enum):
    """How many rows of a contingency result table to keep per element or contingency.

    The filter only touches post-contingency rows. Base case rows always stay, so N-0 is complete whatever the mode.
    """

    ALL = "all"
    """Keep every violation row."""

    WORST_OVERLOAD_PER_ELEMENT = "worst_overload_per_element"
    """Keep, per element, the single row with the highest severity across all contingencies and sides."""

    WORST_ELEMENT_PER_CONTINGENCY = "worst_element_per_contingency"
    """Keep, per contingency, the single row of the element with the highest severity."""


class PowsyblLimitTableDefinition(BaseModel):
    """The N-1 definition translated for the limit table contingency analysis.

    Holds the contingencies as powsybl receives them plus the network data the result extraction needs. There are no
    monitored elements: contingency results come from the limit violation table, the base case from full monitoring.
    """

    model_config = {"arbitrary_types_allowed": True}

    contingencies: list[PowsyblContingency]
    """The contingencies to run, translated one to one. A contingency without elements is the base case."""

    missing_contingencies: list[Contingency] = []
    """Contingencies whose elements are (partially) not present in the network. They are not run."""

    branch_limits: pd.DataFrame
    """The prepared current limits of every branch and three winding transformer of the network, indexed by element_id
    and side. The column ``value`` holds the permanent limit, plus one column per known limit name."""

    bus_map: pd.DataFrame
    """A mapping from busbar sections and bus breaker buses to bus breaker buses, electrical buses and voltage levels."""

    voltage_levels: pd.DataFrame
    """The voltage levels of the network with nominal_v, high_voltage_limit and low_voltage_limit, indexed by id."""

    branch_ids: list[str]
    """All two sided branches of the network, registered as monitored for the fully monitored base case."""

    trafo3w_ids: list[str]
    """All three winding transformers of the network, registered as monitored for the fully monitored base case."""

    voltage_level_ids: list[str]
    """All voltage levels of the network, registered as monitored for the fully monitored base case."""

    element_name_mapping: dict[str, str]
    """A mapping from element ids to their names, used to fill the element_name columns of the results."""

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


def translate_nminus1_for_limit_table(
    nminus1_definition: Nminus1Definition,
    net: Network,
    branch_limit_cache: Optional[PowsyblBranchLimitCacheProtocol] = None,
) -> PowsyblLimitTableDefinition:
    """Translate the N-1 definition for the limit table contingency analysis.

    The monitored elements of the definition are ignored, except that their names are used in the results. Bus
    contingencies are not expanded over busbar sections here, use powsybl's native contingency propagation instead.

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
    id_type = nminus1_definition.id_type or "powsybl"
    if id_type not in (supported_ids := get_args(POWSYBL_SUPPORTED_ID_TYPES)):
        raise ValueError(f"Unsupported id_type {id_type}. Only {supported_ids} are supported for Powsybl.")
    if nminus1_definition.spps_rules:
        logger.warning("powsybl_spps_rules_ignored", n_spps_rules=len(nminus1_definition.spps_rules))

    identifiables = net.get_identifiables(attributes=[]).index
    contingencies, missing_contingencies = translate_contingency_to_powsybl(nminus1_definition.contingencies, identifiables)

    branches = net.get_branches(attributes=["type"])
    branch_names = pd.concat(
        [
            net.get_lines(attributes=["name"]),
            net.get_2_windings_transformers(attributes=["name"]),
            net.get_tie_lines(attributes=["name"]),
        ],
        axis=0,
    )
    trafo3ws = net.get_3_windings_transformers(attributes=["name"])
    voltage_levels = net.get_voltage_levels(attributes=["name", "nominal_v", "high_voltage_limit", "low_voltage_limit"])
    voltage_levels["high_voltage_limit"] = voltage_levels["high_voltage_limit"].fillna(voltage_levels["nominal_v"] * 1.2)
    voltage_levels["low_voltage_limit"] = voltage_levels["low_voltage_limit"].fillna(voltage_levels["nominal_v"] * 0.8)

    branch_ids = branches.index.astype(str).tolist()
    trafo3w_ids = trafo3ws.index.astype(str).tolist()
    branch_limits = resolve_branch_limits_for_powsybl(
        net, monitored_branches=[*branch_ids, *trafo3w_ids], branch_limit_cache=branch_limit_cache
    )

    element_name_mapping = {
        **{str(index): str(name) for index, name in branch_names["name"].items() if name},
        **{str(index): str(name) for index, name in trafo3ws["name"].items() if name},
        **{str(index): str(name) for index, name in voltage_levels["name"].items() if name},
        **{element.id: element.name for element in nminus1_definition.monitored_elements if element.name},
    }
    contingency_name_mapping = {contingency.id: contingency.name or "" for contingency in nminus1_definition.contingencies}

    return PowsyblLimitTableDefinition(
        contingencies=contingencies,
        missing_contingencies=missing_contingencies,
        branch_limits=branch_limits,
        bus_map=get_busbar_mapping(net),
        voltage_levels=voltage_levels.drop(columns=["name"]),
        branch_ids=branch_ids,
        trafo3w_ids=trafo3w_ids,
        voltage_level_ids=voltage_levels.index.astype(str).tolist(),
        element_name_mapping=element_name_mapping,
        contingency_name_mapping=contingency_name_mapping,
    )
