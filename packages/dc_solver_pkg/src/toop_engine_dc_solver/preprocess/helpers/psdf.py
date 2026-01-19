# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds a function to compute the psdf of a grid."""

import numpy as np
from jaxtyping import Bool, Float, Int


def compute_psdf(
    ptdf: Float[np.ndarray, " n_branch n_node"],
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    susceptance: Float[np.ndarray, " n_branch"],
    phaseshift_mask: Bool[np.ndarray, " n_branch"],
    base_mva: float,
) -> Float[np.ndarray, " n_branch n_phaseshifters"]:
    """Calculate the psdf.

    Calculate the psdf to compute the influence of phase shifters in the grid on the loadflow
    Loadflow = PTDF * injection_vector + psdf * phaseshift_vector

    Parameters
    ----------
    ptdf : Float[np.ndarray, " n_branch n_node"]
        PTDF matrix which was extended with extra columns for the second bus
        for the relevant substations.
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-bus
        of a branch can be set to the second bus of a substation.
    susceptance : Float[np.ndarray, " n_branch"]
        Vector with the susceptances for the branches.
    phaseshift_mask: Bool[np.ndarray, " n_branch"]
        Boolean mask identifying which branches are phaseshifters and thus, are relevant columns of the psdf
    base_mva: float
        Factor to multiply the resulting psdf with to get real world mw-flows

    Returns
    -------
    Float[np.ndarray, " n_branch n_phaseshifters"]
        BranchxBranch Matrix showing the influence of shifting the phase from one branch to another.
        Masked down to only phaseshifters Given in MW/°
    """
    # Compute psdf in MW/radiant
    phase_shift_idx = np.flatnonzero(phaseshift_mask)
    psdf = -susceptance[phase_shift_idx] * (ptdf[:, from_node[phase_shift_idx]] - ptdf[:, to_node[phase_shift_idx]])
    for i, branch_idx in enumerate(phase_shift_idx):
        psdf[branch_idx][i] += susceptance[branch_idx]
    psdf = base_mva * (np.pi / 180) * psdf
    return psdf
