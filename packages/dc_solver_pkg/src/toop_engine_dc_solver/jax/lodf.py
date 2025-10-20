"""Contains the functions to calculate the Line Outage Distribution Factors (LODFs).

The formula for the change in load when taking out a branch b and computing the effect on branch a is:
LODF_{a,b} = (PTDF_{a,f_b} - PTDF_{a,t_b}) / (1 - (PTDF_{b,f_b} - PTDF_{b,t_b}))

This module provides this calculation for a lot of single-outages at once. For multi-outages,
use the multi_outages module
"""

from functools import partial

import jax
from beartype.typing import Optional
from jax import numpy as jnp  # pylint: disable=no-name-in-module
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.types import int_max


def calc_lodf(
    branch_to_outage: Int[Array, " "],
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    branches_monitored: Optional[Int[Array, " n_branches_monitored"]],
) -> tuple[Float[Array, " n_branches_monitored"], Bool[Array, " "]]:
    """
    Calculate the LODF vector for a single outage or disconnection

    The formula for this, when taking out a branch b and computing the effect on branch a is:
    LODF_{a,b} = (PTDF_{a,f_b} - PTDF_{a,t_b}) / (1 - (PTDF_{b,f_b} - PTDF_{b,t_b}))

    Parameters
    ----------
    branch_to_outage : Int[Array, " "]
        Which branch is outaged.
    ptdf : Float[Array, " n_branches n_bus"]
        PTDF matrix.
    from_node : Int[Array, " n_branches"]
        From node of each branch.
    to_node : Int[Array, " n_branches"]
        To node of each branch.
    branches_monitored : Optional[Int[Array, " n_branches_monitored"]]
        Selection of monitored branches. If None is passed, all branches are returned. (static)

    Returns
    -------
    Float[Array, "n_branches_monitored"]
        LODF for each monitored branch.
    Bool[Array, " "]
        Whether the LODF was defined. False if the network split
    """
    # From/to nodes of the branch that is outaged
    # This could be int_max in case the branch was disconnected or an invalid outage index was passed.
    from_node_outage = from_node.at[branch_to_outage].get(mode="fill", fill_value=int_max())
    to_node_outage = to_node.at[branch_to_outage].get(mode="fill", fill_value=int_max())

    # Denominator, use .at to ensure that the branch that is outaged has a value of 1
    denom = (
        1
        - ptdf.at[branch_to_outage, from_node_outage].get(mode="fill", fill_value=0.0)
        + ptdf.at[branch_to_outage, to_node_outage].get(mode="fill", fill_value=0.0)
    )

    # Nominator
    nom = ptdf.at[:, from_node_outage].get(mode="fill", fill_value=0.0) - ptdf.at[:, to_node_outage].get(
        mode="fill", fill_value=0.0
    )

    # The lodf of the outaged branch must be -1,
    # so we ensure this
    nom = nom.at[branch_to_outage].set(-denom, mode="drop")

    # Check if the network does not split (the network splits if denom is 0)
    success = jnp.abs(denom) > 1e-11

    if branches_monitored is not None:
        nom = nom.at[branches_monitored].get(mode="fill", fill_value=0)

    return nom / denom, success


def calc_lodf_matrix(
    branches_to_outage: Int[Array, " n_failures"],
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    branches_monitored: Optional[Int[Array, " n_branches_monitored"]],
) -> tuple[Float[Array, " n_failures n_branches_monitored"], Bool[Array, " n_failures"]]:
    """Calculate the LODF matrix.

    Parameters
    ----------
    branches_to_outage : Int[Array, "n_failures"]
        Which branch is outaged.
    ptdf : Float[Array, "n_branches n_bus"]
        PTDF matrix.
    from_node : Int[Array, "n_branches"]
        From node of each branch.
    to_node : Int[Array, "n_branches"]
        To node of each branch.
    branches_monitored : Optional[Int[Array, "n_branches_monitored"]]
        Selection of monitored branches. (static)

    Returns
    -------
    Float[Array, "n_failures n_branches_monitored"]
        LODF for each monitored branch.
    Bool[Array, " n_failures"]
        Whether the LODF was defined. False if the network split
    """
    calc_lodf_partial = partial(
        calc_lodf,
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        branches_monitored=branches_monitored,
    )

    lodf, success = jax.vmap(calc_lodf_partial)(branches_to_outage)
    return lodf, success


def get_failure_cases_to_zero(
    branches_to_zero: Int[Array, " n_branches_to_zero"],
    branches_to_outage: Int[Array, " n_failures"],
) -> Bool[Array, " n_failures"]:
    """Get the failure cases that should be zeroed

    Parameters
    ----------
    branches_to_zero : Int[Array, "n_branches_to_zero"]
        The branches to zero out, indexing into all branches
    branches_to_outage : Int[Array, "n_failures"]
        The list of N-1 failure cases, indexing into all branches

    Returns
    -------
    Bool[Array, "n_failures"]
        A boolean mask over failure cases, where True means that the case should be zeroed
    """
    assert branches_to_zero.size > 0, "branches_to_zero must not be empty"
    assert branches_to_outage.size > 0, "branches_to_outage must not be empty"

    comparison_matrix = branches_to_outage[:, None] == branches_to_zero[None, :]
    return jnp.any(comparison_matrix, axis=1)


def zero_lodf_matrix(
    lodf_matrix: Float[Array, " n_failures n_branches_monitored"],
    success: Bool[Array, " n_failures"],
    branches_to_zero: Int[Array, " n_branches_to_zero"],
    branches_to_outage: Int[Array, " n_failures"],
) -> tuple[Float[Array, " n_failures n_branches_monitored"], Bool[Array, " n_failures"]]:
    """Zeroes the LODF matrix for specific branches

    This is useful if these branches have been outaged due to a solver input and should not be
    considered in the analysis

    Parameters
    ----------
    lodf_matrix : Float[Array, "n_failures n_branches_monitored"]
        The LODF matrix
    success : Bool[Array, "n_failures"]
        Whether the LODF was defined. False if the network split
    branches_to_zero : Int[Array, "n_branches_to_zero"]
        The branches to zero out, indexing into all branches
    branches_to_outage : Int[Array, "n_failures"]
        The list of N-1 failure cases. Only branches in this list are zeroed

    Returns
    -------
    Float[Array, "n_failures n_branches_monitored"]
        The zeroed LODF matrix where the rows that correspond to branches_to_zero are zeroed
    Bool[Array, "n_failures"]
        The success vector where the rows that correspond to branches_to_zero are set to True
    """
    comparison_matrix = branches_to_outage[:, None] == branches_to_zero[None, :]
    failure_cases_to_zero: Bool[Array, " n_failures"] = jnp.any(comparison_matrix, axis=1)

    lodf_matrix = jnp.where(failure_cases_to_zero[:, None], 0, lodf_matrix)
    success = jnp.where(failure_cases_to_zero, True, success)
    return lodf_matrix, success
