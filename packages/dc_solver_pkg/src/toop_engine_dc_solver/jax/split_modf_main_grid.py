# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Compute islanding MODF results for the component containing the slack bus.

This fallback is for contingencies already known to island the grid, either
because preprocessing identified a bridge outage or because the ordinary
LODF/MODF denominator failed its rank or determinant check. Disconnected
islands are not solved.

Note, this is a demonstration of the split-MODF algorithm. It is not optimized for performance.
- known bridges could be handled with a single LODF computation instead of a full MODF.
- known bridge and 3-winding-transformer masks should be prepared during preprocessing.
- contingency cases are batched by fixed outage width; topology batching remains the outer vmap.
- _pinv_small -> svd is likely a bad idea for performance


"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

DEFAULT_RANK_TOL = 1e-10


@partial(jax.jit, static_argnames=("n_bus",))
def main_grid_reachable_mask(
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    outages: Int[Array, " n_outages"],
    slack_bus: Int[Array, ""],
    n_bus: int,
) -> Bool[Array, " n_bus"]:
    """Find buses reachable from the slack bus after applying outages.

    Parameters
    ----------
    from_node : Int[Array, " n_branches"]
        From-node index for each branch in the current topology.
    to_node : Int[Array, " n_branches"]
        To-node index for each branch in the current topology.
    outages : Int[Array, " n_outages"]
        Indices of branches removed by the contingency.
    slack_bus : Int[Array, ""]
        Index of the original slack bus.
    n_bus : int
        Number of buses in the grid.

    Returns
    -------
    Bool[Array, " n_bus"]
        Mask identifying buses in the component containing ``slack_bus``.

    Notes
    -----
    Branches already disconnected by a topology action have invalid endpoints
    and are ignored. The fixed-shape edge-parallel kernel avoids constructing
    component labels for the split-MODF fallback.

    Reachability must be evaluated on a topology-complete physical grid. Grid
    reduction may remove electrically relevant connecting branches merely
    because they are not monitored, not eligible for outage, or outside the
    optimization area. Those branches must still be retained in ``from_node``
    and ``to_node``: removing them changes connected components and can make an
    ordinary meshed branch appear to be a bridge. Non-physical auxiliary PTDF
    coordinates do not need topology edges and must not be treated as buses in
    the connectivity graph.
    """
    n_branch = from_node.shape[0]
    edge_id = jnp.arange(n_branch, dtype=outages.dtype)

    valid_outage = (outages >= 0) & (outages < n_branch)
    removed = jnp.any(
        edge_id[:, None] == jnp.where(valid_outage, outages, -1)[None, :],
        axis=1,
    )

    valid_edge = (from_node >= 0) & (from_node < n_bus) & (to_node >= 0) & (to_node < n_bus) & ~removed

    # Invalid endpoints must be replaced before gather/scatter.
    f = jnp.where(valid_edge, from_node, 0)
    t = jnp.where(valid_edge, to_node, 0)

    reached = jnp.zeros((n_bus,), dtype=bool).at[slack_bus].set(True)

    def cond(
        state: tuple[Bool[Array, " n_bus"], Bool[Array, ""]],
    ) -> Bool[Array, ""]:
        _, changed = state
        return changed

    def body(
        state: tuple[Bool[Array, " n_bus"], Bool[Array, ""]],
    ) -> tuple[Bool[Array, " n_bus"], Bool[Array, ""]]:
        reached, _ = state

        edge_reached = (reached[f] | reached[t]) & valid_edge

        # scatter-max is OR for booleans.
        new = reached
        new = new.at[f].max(edge_reached)
        new = new.at[t].max(edge_reached)

        changed = jnp.any(new != reached)
        return new, changed

    reached, _ = jax.lax.while_loop(
        cond,
        body,
        (reached, jnp.array(True)),
    )
    return reached


# TODO: Only a robust first implementation -> REMOVE ME
@partial(jax.jit, static_argnames=("n_bus",))
def main_grid_component_labels(
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    outages: Int[Array, " n_outages"],
    n_bus: int,
) -> tuple[Int[Array, " n_bus"], Int[Array, ""]]:
    """Label post-outage components with parallel propagation and pointer jumping.

    Each connected component converges to the smallest node index in that
    component. Edge propagation connects adjacent labels, while pointer jumping
    shortens the label chains between propagation rounds. This avoids advancing
    only one edge away from the slack per round on deep grids.

    Parameters
    ----------
    from_node : Int[Array, " n_branches"]
        From-node index for each branch in the current topology.
    to_node : Int[Array, " n_branches"]
        To-node index for each branch in the current topology.
    outages : Int[Array, " n_outages"]
        Indices of branches removed by the contingency. Invalid indices are
        treated as padding.
    n_bus : int
        Number of physical and auxiliary node coordinates.

    Returns
    -------
    tuple[Int[Array, " n_bus"], Int[Array, ""]]
        Canonical component label for every node and the number of propagation
        rounds required to converge.
    """
    n_branch = from_node.shape[0]
    valid_outage = (outages >= 0) & (outages < n_branch)
    safe_outages = jnp.where(valid_outage, outages, 0)

    # Scatter avoids materializing an n_branch x n_outages comparison matrix.
    removed = jnp.zeros((n_branch,), dtype=bool).at[safe_outages].max(valid_outage)
    valid_edge = (from_node >= 0) & (from_node < n_bus) & (to_node >= 0) & (to_node < n_bus) & ~removed

    # Invalid endpoints must be replaced before gather/scatter. Their resulting
    # self-update at node zero cannot change any component label.
    f = jnp.where(valid_edge, from_node, 0)
    t = jnp.where(valid_edge, to_node, 0)
    labels = jnp.arange(n_bus, dtype=from_node.dtype)

    def cond(
        state: tuple[Int[Array, " n_bus"], Bool[Array, ""], Int[Array, ""]],
    ) -> Bool[Array, ""]:
        _labels, changed, _rounds = state
        return changed

    def body(
        state: tuple[Int[Array, " n_bus"], Bool[Array, ""], Int[Array, ""]],
    ) -> tuple[Int[Array, " n_bus"], Bool[Array, ""], Int[Array, ""]]:
        labels, _changed, rounds = state

        edge_label = jnp.minimum(labels[f], labels[t])
        new = labels.at[f].min(edge_label)
        new = new.at[t].min(edge_label)

        # Path compression propagates labels over exponentially increasing
        # distances on deep components instead of one edge per round.
        new = new[new]
        changed = jnp.any(new != labels)
        return new, changed, rounds + 1

    labels, _changed, rounds = jax.lax.while_loop(
        cond,
        body,
        (labels, jnp.array(True), jnp.array(0, dtype=jnp.int32)),
    )
    return labels, rounds


@partial(jax.jit, static_argnames=("n_bus",))
def main_grid_reachable_mask_parallel(
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    outages: Int[Array, " n_outages"],
    slack_bus: Int[Array, ""],
    n_bus: int,
) -> Bool[Array, " n_bus"]:
    """Find the slack component using parallel post-outage component labels."""
    labels, _rounds = main_grid_component_labels(
        from_node=from_node,
        to_node=to_node,
        outages=outages,
        n_bus=n_bus,
    )
    return labels == labels[slack_bus]


@partial(jax.jit, static_argnames=("n_bus",))
def main_grid_reachable_masks_parallel(
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    outages: Int[Array, " n_cases max_n_outages"],
    slack_bus: Int[Array, ""],
    n_bus: int,
) -> Bool[Array, " n_cases n_bus"]:
    """Find the slack component for a fixed-width batch of outage cases.

    The contingency axis matches the failure axis used by production
    contingency analysis. A surrounding topology batch can therefore use an
    outer ``vmap`` without changing this function's data layout.
    """
    return jax.vmap(
        partial(main_grid_reachable_mask_parallel, n_bus=n_bus),
        in_axes=(None, None, 0, None),
    )(from_node, to_node, outages, slack_bus)


def _build_split_denominator_and_projected_numerator(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    outages: Int[Array, " n_outages"],
    branches_monitored: Int[Array, " n_monitored"],
) -> tuple[
    Float[Array, " n_outages n_outages"],
    Float[Array, " n_monitored n_outages"],
    Int[Array, " n_outages"],
    Bool[Array, " n_outages"],
]:
    """Build the MODF denominator and projected numerator for monitored branches.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        Power transfer distribution factor matrix.
    from_node : Int[Array, " n_branches"]
        From-node index for each branch.
    to_node : Int[Array, " n_branches"]
        To-node index for each branch.
    outages : Int[Array, " n_outages"]
        Indices of branches removed by the contingency.
    branches_monitored : Int[Array, " n_monitored"]
        Indices of branches for which flows are computed.

    Returns
    -------
    tuple[
        Float[Array, " n_outages n_outages"],
        Float[Array, " n_monitored n_outages"],
        Int[Array, " n_outages"],
        Bool[Array, " n_outages"],
    ]
        MODF denominator, monitored numerator, safe outage indices, and valid
        outage mask.
    """
    n_branch = ptdf.shape[0]
    n_bus = ptdf.shape[1]
    m = outages.shape[0]

    outage_index_valid = (outages >= 0) & (outages < n_branch)
    safe_outages = jnp.where(outage_index_valid, outages, 0)

    outage_f_raw = from_node[safe_outages]
    outage_t_raw = to_node[safe_outages]
    outage_endpoint_valid = (
        outage_index_valid & (outage_f_raw >= 0) & (outage_f_raw < n_bus) & (outage_t_raw >= 0) & (outage_t_raw < n_bus)
    )

    outage_f = jnp.where(outage_endpoint_valid, outage_f_raw, 0)
    outage_t = jnp.where(outage_endpoint_valid, outage_t_raw, 0)

    # H_OO[i,j] = PTDF[outage_i, from(outage_j)] -
    #             PTDF[outage_i, to(outage_j)]
    h_oo = ptdf[safe_outages[:, None], outage_f[None, :]] - ptdf[safe_outages[:, None], outage_t[None, :]]
    denom = jnp.eye(m, dtype=ptdf.dtype) - h_oo

    ptdf_mon = ptdf[branches_monitored]
    nom = ptdf_mon[:, outage_f] - ptdf_mon[:, outage_t]

    # Match existing MODF convention: a monitored outaged branch gets
    # the corresponding -denominator row, so application gives exactly zero.
    monitored_is_outage = (branches_monitored[:, None] == safe_outages[None, :]) & outage_endpoint_valid[None, :]
    replacement = -(monitored_is_outage.astype(ptdf.dtype) @ denom)
    nom = jnp.where(
        jnp.any(monitored_is_outage, axis=1)[:, None],
        replacement,
        nom,
    )

    # Padded / already-disconnected entries become identity directions with
    # zero projected response, consistent with the current ToOp MODF code.
    eye = jnp.eye(m, dtype=ptdf.dtype)
    nom = jnp.where(outage_endpoint_valid[None, :], nom, 0.0)
    denom = jnp.where(outage_endpoint_valid[:, None], denom, eye)
    denom = jnp.where(outage_endpoint_valid[None, :], denom, eye)

    return denom, nom, safe_outages, outage_endpoint_valid


def _pinv_small(
    a: Float[Array, " m m"],
    rank_tol: float,
) -> tuple[Float[Array, " m m"], Int[Array, ""]]:
    """Compute the Moore-Penrose inverse and numerical rank of a small matrix.

    Parameters
    ----------
    a : Float[Array, " m m"]
        Square matrix to invert.
    rank_tol : float
        Singular-value threshold used to determine the numerical rank.

    Returns
    -------
    tuple[Float[Array, " m m"], Int[Array, ""]]
        Moore-Penrose inverse and numerical rank.
    """
    u, s, vh = jnp.linalg.svd(a, full_matrices=False)
    keep = s > rank_tol
    inv_s = jnp.where(keep, 1.0 / jnp.where(keep, s, 1.0), 0.0)
    a_pinv = (vh.T * inv_s) @ u.T
    return a_pinv, jnp.sum(keep)


@partial(jax.jit, static_argnames=("rank_tol",))
def _compute_split_modf_main_grid_case_from_mask(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    nodal_injections: Float[Array, " n_timesteps n_bus"],
    n_0_flow: Float[Array, " n_timesteps n_branches"],
    outages: Int[Array, " n_outages"],
    branches_monitored: Int[Array, " n_monitored"],
    main_bus: Bool[Array, " n_bus"],
    slack_bus: Int[Array, ""],
    rank_tol: float = DEFAULT_RANK_TOL,
) -> tuple[
    Float[Array, " n_timesteps n_monitored"],
    Bool[Array, " n_monitored"],
    Float[Array, " n_timesteps"],
    Int[Array, ""],
]:
    """Compute one islanding N-k case for a supplied retained-grid mask.

    This is the electrical split-MODF kernel. Connectivity discovery is kept
    outside it so known islanding contingencies can use masks prepared during
    preprocessing instead of repeating a graph search at runtime.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        Power transfer distribution factor matrix.
    from_node : Int[Array, " n_branches"]
        From-node index for each branch in the current topology.
    to_node : Int[Array, " n_branches"]
        To-node index for each branch in the current topology.
    nodal_injections : Float[Array, " n_timesteps n_bus"]
        Base-case nodal injections for each timestep.
    n_0_flow : Float[Array, " n_timesteps n_branches"]
        Base-case branch flows for each timestep.
    outages : Int[Array, " n_outages"]
        Indices of branches removed by the contingency.
    branches_monitored : Int[Array, " n_monitored"]
        Indices of branches for which flows are computed.
    main_bus : Bool[Array, " n_bus"]
        Buses retained for the split calculation. This must describe the
        physical post-contingency component containing ``slack_bus``.
    slack_bus : Int[Array, ""]
        Index of the original slack bus.
    rank_tol : float, default=DEFAULT_RANK_TOL
        Singular-value threshold used to determine MODF rank.

    Preconditions
    -------------
    The caller has already classified this contingency as islanding and
    supplied its retained main-grid mask. The mask may be preprocessed for a
    known bridge or 3-winding-transformer outage, or discovered dynamically
    for a MODF failure that was not known in advance.

    Algorithm
    ---------
    1. Set injections outside the supplied component to zero.
    2. Rebalance the retained component at the original slack.
    3. Correct the current N-0 monitored/outage flows for that injection change.
       This preserves any non-injection contribution already present in
       n_0_flow (e.g. phase-shifter contribution).
    4. Apply a Moore-Penrose MODF in outage space.  Null cut directions are
       discarded; any regular directions in a mixed N-k set are retained.
    5. Return NaN for monitored branches outside the retained main component.

    Returns
    -------
    Float[Array, " n_timesteps n_monitored"]
        Main-grid monitored flows. Branches outside the main component are NaN.
        Outaged monitored branches whose endpoints remain in the main component
        are exactly zero.
    Bool[Array, " n_monitored"]
        Monitored branches whose endpoints both belong to the main component.
    Float[Array, " n_timesteps"]
        Per-timestep imbalance after discarding all non-main-grid injections.
        The opposite of this value is applied at the original slack.
    Int[Array, ""]
        Numerical nullity of the padded MODF denominator.  For one bridge this
        is 1.  This is diagnostic only; reachability defines the retained grid.
    """
    n_bus = ptdf.shape[1]

    # Keep no injection from discarded islands.  Since we only want the main
    # grid, there is no need to identify or balance those other components.
    p_main = jnp.where(main_bus[None, :], nodal_injections, 0.0)
    main_imbalance = jnp.sum(p_main, axis=1)
    p_main = p_main.at[:, slack_bus].add(-main_imbalance)

    # Delta relative to the already-current ToOp base case.  Updating n_0_flow
    # instead of rebuilding it from H@p preserves PST/other additive base flow.
    delta_p = p_main - nodal_injections

    denom, nom_mon, safe_outages, valid_outages = _build_split_denominator_and_projected_numerator(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        outages=outages,
        branches_monitored=branches_monitored,
    )

    denom_pinv_t, rank = _pinv_small(denom.T, rank_tol)
    modf_mon = (denom_pinv_t @ nom_mon.T).T

    # Only rows required by the split result are corrected.  No full
    # n_branch x n_bus recomputation is needed.
    base_mon = n_0_flow[:, branches_monitored] + delta_p @ ptdf[branches_monitored].T
    base_out = n_0_flow[:, safe_outages] + delta_p @ ptdf[safe_outages].T
    base_out = jnp.where(valid_outages[None, :], base_out, 0.0)

    flows = base_mon + jnp.einsum("mo,to->tm", modf_mon, base_out)

    # Determine which monitored branches belong to the retained bus set.
    mon_f_raw = from_node[branches_monitored]
    mon_t_raw = to_node[branches_monitored]
    mon_endpoint_valid = (mon_f_raw >= 0) & (mon_f_raw < n_bus) & (mon_t_raw >= 0) & (mon_t_raw < n_bus)
    mon_f = jnp.where(mon_endpoint_valid, mon_f_raw, 0)
    mon_t = jnp.where(mon_endpoint_valid, mon_t_raw, 0)
    main_monitored = mon_endpoint_valid & main_bus[mon_f] & main_bus[mon_t]

    # An internal branch outage can still have both endpoints reachable through
    # another path; such an outaged monitored branch should read exactly zero.
    monitored_is_outaged = jnp.any(
        (branches_monitored[:, None] == safe_outages[None, :]) & valid_outages[None, :],
        axis=1,
    )
    flows = jnp.where(
        main_monitored[None, :] & monitored_is_outaged[None, :],
        0.0,
        flows,
    )

    # Main-grid-only contract: don't invent results for discarded components.
    flows = jnp.where(main_monitored[None, :], flows, jnp.nan)

    rank_loss = denom.shape[0] - rank
    return flows, main_monitored, main_imbalance, rank_loss


@partial(jax.jit, static_argnames=("rank_tol",))
def compute_split_modf_main_grid_from_mask(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    nodal_injections: Float[Array, " n_timesteps n_bus"],
    n_0_flow: Float[Array, " n_timesteps n_branches"],
    outages: Int[Array, " n_cases max_n_outages"],
    branches_monitored: Int[Array, " n_monitored"],
    main_bus: Bool[Array, " n_cases n_bus"],
    slack_bus: Int[Array, ""],
    rank_tol: float = DEFAULT_RANK_TOL,
) -> tuple[
    Float[Array, " n_timesteps n_cases n_monitored"],
    Bool[Array, " n_cases n_monitored"],
    Float[Array, " n_timesteps n_cases"],
    Int[Array, " n_cases"],
]:
    """Compute a fixed-width batch of islanding cases from supplied masks.

    The function consumes one topology and batches over its contingency cases.
    Its output uses the production contingency-analysis layout of timestep,
    failure, then monitored branch. Cases with different outage counts must be
    grouped into separate fixed-width batches, as for existing MODF matrices.
    A surrounding topology batch can apply an outer ``vmap``.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        Power transfer distribution factor matrix.
    from_node : Int[Array, " n_branches"]
        From-node index for each branch in the current topology.
    to_node : Int[Array, " n_branches"]
        To-node index for each branch in the current topology.
    nodal_injections : Float[Array, " n_timesteps n_bus"]
        Base-case nodal injections for each timestep.
    n_0_flow : Float[Array, " n_timesteps n_branches"]
        Base-case branch flows for each timestep.
    outages : Int[Array, " n_cases max_n_outages"]
        Fixed-width branch-outage sets. Invalid indices are padding.
    branches_monitored : Int[Array, " n_monitored"]
        Indices of branches for which flows are computed.
    main_bus : Bool[Array, " n_cases n_bus"]
        Precomputed retained-main-grid mask for every outage case.
    slack_bus : Int[Array, ""]
        Index of the original slack bus.
    rank_tol : float, default=DEFAULT_RANK_TOL
        Singular-value threshold used to determine MODF rank.

    Returns
    -------
    Float[Array, " n_timesteps n_cases n_monitored"]
        Main-grid monitored flows. Branches outside each component are NaN.
    Bool[Array, " n_cases n_monitored"]
        Monitored branches retained for each contingency case.
    Float[Array, " n_timesteps n_cases"]
        Per-timestep retained-grid imbalance for each contingency case.
    Int[Array, " n_cases"]
        Numerical nullity of each padded MODF denominator.
    """
    flows, main_monitored, main_imbalance, rank_loss = jax.vmap(
        partial(_compute_split_modf_main_grid_case_from_mask, rank_tol=rank_tol),
        in_axes=(None, None, None, None, None, 0, None, 0, None),
    )(
        ptdf,
        from_node,
        to_node,
        nodal_injections,
        n_0_flow,
        outages,
        branches_monitored,
        main_bus,
        slack_bus,
    )
    return (
        jnp.transpose(flows, (1, 0, 2)),
        main_monitored,
        jnp.transpose(main_imbalance, (1, 0)),
        rank_loss,
    )


@partial(jax.jit, static_argnames=("rank_tol",))
def compute_split_modf_main_grid(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    nodal_injections: Float[Array, " n_timesteps n_bus"],
    n_0_flow: Float[Array, " n_timesteps n_branches"],
    outages: Int[Array, " n_cases max_n_outages"],
    branches_monitored: Int[Array, " n_monitored"],
    slack_bus: Int[Array, ""],
    rank_tol: float = DEFAULT_RANK_TOL,
) -> tuple[
    Float[Array, " n_timesteps n_cases n_monitored"],
    Bool[Array, " n_cases n_bus"],
    Bool[Array, " n_cases n_monitored"],
    Float[Array, " n_timesteps n_cases"],
    Int[Array, " n_cases"],
]:
    """Search and compute a fixed-width batch of unknown islanding cases.

    This convenience path matches the production failure-axis layout but pays
    for connectivity discovery. Known islanding contingencies should use
    :func:`compute_split_modf_main_grid_from_mask` with masks prepared during
    preprocessing.
    """
    main_bus = main_grid_reachable_masks_parallel(
        from_node=from_node,
        to_node=to_node,
        outages=outages,
        slack_bus=slack_bus,
        n_bus=ptdf.shape[1],
    )
    flows, main_monitored, main_imbalance, rank_loss = compute_split_modf_main_grid_from_mask(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        nodal_injections=nodal_injections,
        n_0_flow=n_0_flow,
        outages=outages,
        branches_monitored=branches_monitored,
        main_bus=main_bus,
        slack_bus=slack_bus,
        rank_tol=rank_tol,
    )
    return flows, main_bus, main_monitored, main_imbalance, rank_loss
