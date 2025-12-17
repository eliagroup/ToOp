# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds functions helping with some pandapower related tasks"""

import logging

import numpy as np
import pandapower as pp
from jaxtyping import Float, Int
from pandapower.pypower.idx_brch import F_BUS, T_BUS

logger = logging.Logger(__name__)


def get_max_line_flow(net: pp.pandapowerNet) -> Float[np.ndarray, " n_pp_lines"]:
    """Get the rated power for each line in the network

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network containing the line

    Returns
    -------
    Float[np.ndarray, " n_pp_lines"]
        The rated power of each line branch in the network
    """
    voltage_rating = net.bus.loc[net.line["from_bus"].values, "vn_kv"].values * np.sqrt(3.0)
    max_i_ka = net.line.max_i_ka.values
    derating_factor = net.line.df.values
    parallel = net.line.parallel.values
    max_line_flow = max_i_ka * derating_factor * parallel * voltage_rating
    return max_line_flow


def get_max_trafo_flow(net: pp.pandapowerNet) -> Float[np.ndarray, " n_pp_trafos"]:
    """Get the rated power for each trafo in the network.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network containing the trafos

    Returns
    -------
    Float[np.ndarray, " n_pp_trafos"]
        The rated power of each trafo branch in the network
    """
    sn_mva = net.trafo.sn_mva.values
    derating_factor = net.trafo.df.values
    parallel = net.trafo.parallel.values
    max_trafo_flow = sn_mva * derating_factor * parallel
    return max_trafo_flow


def get_max_trafo3w_flow(
    net: pp.pandapowerNet,
) -> Float[np.ndarray, " n_pp_trafo3ws_times_3"]:
    """Get the rated power of the different voltage levels of all trafo3ws in the network

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network containing the trafo3ws

    Returns
    -------
    Float[np.ndarray, " n_pp_trafo3ws*3"]
        The rated power of each trafo3w branch in the network
    """
    max_trafo3w_hv_flow = net.trafo3w.sn_hv_mva.values
    max_trafo3w_mv_flow = net.trafo3w.sn_mv_mva.values
    max_trafo3w_lv_flow = net.trafo3w.sn_lv_mva.values
    return np.concatenate([max_trafo3w_hv_flow, max_trafo3w_mv_flow, max_trafo3w_lv_flow])


def get_trafo3w_ppc_branch_idx(
    net: pp.pandapowerNet, trafo3w_pp_idx: Int[np.ndarray, " rel_trafo3ws"]
) -> Int[np.ndarray, " 3 rel_trafo3ws"]:
    """Get the corresponding branch ids of the trafo3w pandapower ids.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network containing the trafo3ws
    trafo3w_pp_idx: Int[np.ndarray, " rel_trafo3ws"]
        An array of trafo3w indices from pandapower

    Returns
    -------
    Int[np.ndarray, " 3 rel_trafo3ws"]
        Return a 3xnum_trafo3w_pp_idx array containing the hv, mv and lv ppci_branch_idx of the trafo3w
    """
    # 3 winding trafos are listed after lines and 2w-trafos
    idx_before = net.line.shape[0] + net.trafo.shape[0]
    # Each voltage level of the trafo3w is its own branch.
    # The 3 branches have the ids
    # id += voltage_multiplier * n_trafo3w with voltage_multiplier = (hv->0, mv->1, lv->2)
    num_trafo3w = net.trafo3w.shape[0]
    return np.array(
        [
            idx_before + trafo3w_pp_idx,
            idx_before + trafo3w_pp_idx + num_trafo3w,
            idx_before + trafo3w_pp_idx + 2 * num_trafo3w,
        ]
    )


def get_trafo3w_ppc_node_idx(
    ppci: dict, trafo3w_branch_idx: Int[np.ndarray, " 3 rel_trafo3ws"]
) -> Int[np.ndarray, " rel_trafo3ws"]:
    """Get the corresponding node indices of the trafo3w pandapower ids

    Parameters
    ----------
    ppci : dict
        The ppci dict from pandapower
    trafo3w_branch_idx: Int[np.ndarray, " 3 rel_trafo3ws"]
        An array of trafo3w indices as obtained from get_trafo3w_ppc_branch_idx

    Returns
    -------
    Int[np.ndarray, " rel_trafo3ws"]
        Return a num_trafo3w_pp_idx array containing the ppci node indices of the trafo3w
    """
    center_bus = ppci["branch"][trafo3w_branch_idx[0, :], T_BUS].astype(int)
    center_bus_2 = ppci["branch"][trafo3w_branch_idx[1, :], F_BUS].astype(int)
    center_bus_3 = ppci["branch"][trafo3w_branch_idx[2, :], F_BUS].astype(int)
    assert np.array_equal(center_bus, center_bus_2)
    assert np.array_equal(center_bus, center_bus_3)
    return center_bus
