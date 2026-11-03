# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Prepare the distance protection relays of a run from a pandapower network."""

import logging

import pandapower as pp
import pandas as pd
import pandera.typing as pat
from toop_engine_contingency_analysis.pandapower.cascade.configuration import CascadeConfig
from toop_engine_contingency_analysis.pandapower.cascade.detection.distance_protection import describe_relay_zones
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import PreparedRelays, RelaySchema

_logger = logging.getLogger(__name__)

#: pandapower tables a relay's protected element is looked up in: the branches a cascade step can outage.
PROTECTED_ELEMENT_TABLES = ("line", "trafo", "trafo3w")


def resolve_relay_targets(net: pp.pandapowerNet, relays: pd.DataFrame) -> pd.DataFrame:
    """Look up the pandapower rows of every element a relay protects.

    Each relay names its protected elements by external id in ``protected_elements``. This
    resolves the ids against the tables a relay can trip — lines, two- and three-winding
    transformers — so that a trip needs no lookup of its
    own. One id can resolve to several rows, e.g. a line split into segments, and all of them
    are kept. An id found in none of the tables is logged and dropped: the element is not part
    of this network, or not of a type a trip can take out.

    Parameters
    ----------
    net : pp.pandapowerNet
        Network the protected elements live in.
    relays : pd.DataFrame
        Relay table validated against :class:`RelaySchema`.

    Returns
    -------
    pd.DataFrame
        One row per relay and resolved element, with ``relay_id``, ``element_mrid``,
        ``table`` and ``table_id``.
    """
    protected = (
        relays[["relay_id", "protected_elements"]]
        .explode("protected_elements")
        .rename(columns={"protected_elements": "element_mrid"})
    )

    lookups = [
        pd.DataFrame(
            {
                "element_mrid": net[table]["origin_id"].to_numpy(),
                "table": table,
                "table_id": net[table].index.to_numpy(),
            }
        )
        for table in PROTECTED_ELEMENT_TABLES
        if table in net and "origin_id" in net[table].columns
    ]
    lookup = pd.concat(lookups, ignore_index=True).dropna(subset=["element_mrid"])

    resolved = protected.merge(lookup, on="element_mrid", how="left")
    unresolved = resolved.loc[resolved["table"].isna(), "element_mrid"]
    if not unresolved.empty:
        _logger.warning(
            "cascading: %d protected element(s) not found among lines, transformers and switches; ignored: %s",
            len(unresolved),
            sorted(set(unresolved)),
        )
    # a relay on several breakers has a row per breaker, and the rows can name the same element
    resolved = resolved.dropna(subset=["table"]).astype({"table_id": int})
    return resolved.drop_duplicates(subset=["relay_id", "table", "table_id"]).reset_index(drop=True)


def prepare_relays(
    net: pp.pandapowerNet,
    relays: pat.DataFrame[RelaySchema],
    cascade_configuration: CascadeConfig,
) -> PreparedRelays:
    """Validate the relays and prepare them once per run, so every outage only reads them.

    A relay's polygon, its factors and its protected elements depend only on the relay, the
    configuration and the network's elements, never on which contingency is running, so they
    are worked out here rather than per outage:

    1. :func:`~toop_engine_contingency_analysis.pandapower.cascade.detection.distance_protection.describe_relay_zones`
       converts ``angle`` to radians, derives the ``poly`` polygon and resolves the factor every
       relay gets for both zones and both cases.
    2. :func:`resolve_relay_targets` resolves every protected element to its pandapower row.

    Parameters
    ----------
    net : pp.pandapowerNet
        Base-case pandapower network the relays measure in.
    relays : pat.DataFrame[RelaySchema]
        The relays, one row per relay.
    cascade_configuration : CascadeConfig
        Cascade settings the factors are resolved from.

    Returns
    -------
    PreparedRelays
        The prepared relay table and the resolved protected elements.

    Raises
    ------
    pandera.errors.SchemaError
        If *relays* does not match :class:`RelaySchema`.
    """
    relays = RelaySchema.validate(relays)

    return PreparedRelays(
        relays=describe_relay_zones(relays, cascade_configuration),
        targets=resolve_relay_targets(net, relays),
    )
