# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Shared network-mask dataclasses used across packages."""

from dataclasses import dataclass

import numpy as np
from pypowsybl.network.impl.network import Network


@dataclass(frozen=True)
class NetworkMasks:
    """Class to hold the network masks.

    See class PowsyblBackend(BackendInterface) in the DC solver for more information.
    """

    relevant_subs: np.ndarray
    """relevant_subs.npy (a boolean mask of relevant nodes)."""

    line_for_nminus1: np.ndarray
    """line_for_nminus1.npy (a boolean mask of lines that are relevant for n-1)."""

    line_for_reward: np.ndarray
    """line_for_reward.npy (a boolean mask of lines that are relevant for the reward)."""

    line_overload_weight: np.ndarray
    """line_overload_weight.npy (a float mask of weights for the overload)."""

    line_disconnectable: np.ndarray
    """line_disconnectable.npy (a boolean mask of lines that can be disconnected)."""

    line_blacklisted: np.ndarray
    """line_blacklisted.npy (a boolean mask of lines that are blacklisted).

    Currently only used during importing and not part of the PowsyblBackend.
    """

    line_tso_border: np.ndarray
    """line_tso_border.npy (a boolean mask of lines leading to TSOs outside the reward area).

    Currently only used during importing and not part of the PowsyblBackend.
    """

    trafo_for_nminus1: np.ndarray
    """trafo_for_nminus1.npy (a boolean mask of transformers that are relevant for n-1)."""

    trafo_for_reward: np.ndarray
    """trafo_for_reward.npy (a boolean mask of transformers that are relevant for the reward)."""

    trafo_overload_weight: np.ndarray
    """trafo_overload_weight.npy (a float mask of weights for the overload)."""

    trafo_disconnectable: np.ndarray
    """trafo_disconnectable.npy (a boolean mask of transformers that can be disconnected)."""

    trafo_blacklisted: np.ndarray
    """trafo_blacklisted.npy (a boolean mask of transformers that are blacklisted).

    Currently only used during importing and not part of the PowsyblBackend.
    """

    trafo_n0_n1_max_diff_factor: np.ndarray
    """trafo_n0_n1_max_diff_factor.npy stores optional N-0 to N-1 difference factors."""

    trafo_dso_border: np.ndarray
    """trafo_dso_border.npy marks transformers bordering the DSO control area.

    Currently only used during importing and not part of the PowsyblBackend.
    """

    trafo_controllable: np.ndarray
    """trafo_controllable.npy marks controllable transformers within the control area."""

    tie_line_for_reward: np.ndarray
    """tie_line_for_reward.npy (a boolean mask of tie lines that are relevant for the reward)."""

    tie_line_for_nminus1: np.ndarray
    """tie_line_for_nminus1.npy (a boolean mask of tie lines that are relevant for n-1)."""

    tie_line_overload_weight: np.ndarray
    """tie_line_overload_weight.npy (a float mask of weights for the overload)."""

    tie_line_disconnectable: np.ndarray
    """tie_line_disconnectable.npy (a boolean mask of tie lines that can be disconnected)."""

    tie_line_tso_border: np.ndarray
    """tie_line_tso_border.npy marks tie lines leading to TSOs outside the reward area.

    Currently only used during importing and not part of the PowsyblBackend.
    """

    boundary_line_for_nminus1: np.ndarray
    """boundary_line_for_nminus1.npy (a boolean mask of boundary lines that are relevant for n-1)."""

    generator_for_nminus1: np.ndarray
    """generator_for_nminus1.npy (a boolean mask of generators that are relevant for n-1)."""

    load_for_nminus1: np.ndarray
    """load_for_nminus1.npy (a boolean mask of loads that are relevant for n-1)."""

    switch_for_nminus1: np.ndarray
    """switches_nminus1.npy (a boolean mask of switches that are relevant for n-1)."""

    switch_for_reward: np.ndarray
    """switches_reward.npy (a boolean mask of switches that are relevant for the reward)."""

    busbar_for_nminus1: np.ndarray
    """busbar_for_nminus1.npy (a boolean mask of busbars that are relevant for n-1)."""


def create_default_network_masks(network: Network) -> NetworkMasks:
    """Create a default Powsybl ``NetworkMasks`` object with all masks set to False.

    Parameters
    ----------
    network : Network
        Powsybl network-like object exposing the standard ``get_*`` accessors used during
        preprocessing.

    Returns
    -------
    NetworkMasks
        Default masks aligned to the current network element tables.
    """
    bus_df = network.get_buses(attributes=[])
    lines_df = network.get_lines(attributes=[])
    trafo_df = network.get_2_windings_transformers(attributes=[]).sort_index()
    tie_df = network.get_tie_lines(attributes=[])
    dangling_df = network.get_boundary_lines(attributes=[])
    generator_df = network.get_generators(attributes=[])
    load_df = network.get_loads(attributes=[])
    switches_df = network.get_switches(attributes=[])
    busbar_df = network.get_busbar_sections(attributes=[])

    return NetworkMasks(
        relevant_subs=np.zeros(len(bus_df), dtype=bool),
        line_for_nminus1=np.zeros(len(lines_df), dtype=bool),
        line_for_reward=np.zeros(len(lines_df), dtype=bool),
        line_overload_weight=np.ones(len(lines_df), dtype=float),
        line_disconnectable=np.zeros(len(lines_df), dtype=bool),
        line_blacklisted=np.zeros(len(lines_df), dtype=bool),
        line_tso_border=np.zeros(len(lines_df), dtype=bool),
        trafo_for_nminus1=np.zeros(len(trafo_df), dtype=bool),
        trafo_for_reward=np.zeros(len(trafo_df), dtype=bool),
        trafo_overload_weight=np.ones(len(trafo_df), dtype=float),
        trafo_disconnectable=np.zeros(len(trafo_df), dtype=bool),
        trafo_controllable=np.zeros(len(trafo_df), dtype=bool),
        trafo_blacklisted=np.zeros(len(trafo_df), dtype=bool),
        trafo_n0_n1_max_diff_factor=np.ones(len(trafo_df), dtype=float) * -1,
        trafo_dso_border=np.zeros(len(trafo_df), dtype=bool),
        tie_line_for_reward=np.zeros(len(tie_df), dtype=bool),
        tie_line_for_nminus1=np.zeros(len(tie_df), dtype=bool),
        tie_line_overload_weight=np.ones(len(tie_df), dtype=float),
        tie_line_disconnectable=np.zeros(len(tie_df), dtype=bool),
        tie_line_tso_border=np.zeros(len(tie_df), dtype=bool),
        boundary_line_for_nminus1=np.zeros(len(dangling_df), dtype=bool),
        generator_for_nminus1=np.zeros(len(generator_df), dtype=bool),
        load_for_nminus1=np.zeros(len(load_df), dtype=bool),
        switch_for_nminus1=np.zeros(len(switches_df), dtype=bool),
        switch_for_reward=np.zeros(len(switches_df), dtype=bool),
        busbar_for_nminus1=np.zeros(len(busbar_df), dtype=bool),
    )
