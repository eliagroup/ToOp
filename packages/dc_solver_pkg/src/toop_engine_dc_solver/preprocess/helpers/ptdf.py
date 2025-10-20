"""Holds functions to compute the PTDF."""

import numpy as np
from beartype.typing import Optional
from jaxtyping import Bool, Float, Int
from scipy.sparse import csc_matrix, spmatrix
from scipy.sparse.linalg import spsolve


def compute_ptdf(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    susceptances: Float[np.ndarray, " n_branch"],
    slack_bus: int,
) -> Float[np.ndarray, " n_branch n_node"]:
    """Calculate the PTDF (Power Transfer Distiribution Factors).

    Calculate the PTDF to compute the influence of phase shifters in the grid on the loadflow
    Loadflow = PTDF * injection_vector + PSDF * phaseshift_vector

    Taken from:
    https://github.com/e2nIEE/pandapower/blob/7086cf72ac2ff579150d9cca60fffa2f352e4cb9/pandapower/pypower/makePTDF.py#L24

    Parameters
    ----------
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-bus
        of a branch can be set to the second bus of a substation.
    susceptances : Float[np.ndarray, " n_branch"]
        Vector with the susceptances for the branches.
    slack_bus: int
        The slack bus of the grid. Cannot be distributed for the bsdf formulation to work
    number_of_busses: int
        How many busbars are in the system-> How many nodes in the ptdf

    Returns
    -------
    Float[np.ndarray, " n_branch n_node"]
        BranchxNode Matrix showing the influence of nodal injections. A.k.a. the PTDF
    """
    number_of_busses = int(np.max(np.concatenate([from_node, to_node]))) + 1
    number_of_branches = int(from_node.shape[0])

    # use bus 1 for voltage angle reference
    noref = np.arange(1, number_of_busses)
    noslack = np.ones(number_of_busses, dtype=bool)
    noslack[slack_bus] = False

    ptdf = np.zeros((number_of_branches, number_of_busses))
    node_node_susceptance, branch_node_susceptance, *_ = get_susceptance_matrices(
        from_node, to_node, susceptances, number_of_branches, number_of_busses
    )
    ptdf[:, noslack] = spsolve(
        node_node_susceptance[np.ix_(noslack, noref)].T,
        branch_node_susceptance[:, noref].toarray().T,
    ).T
    assert np.isnan(ptdf).sum() == 0, (
        "PTDF contains NaNs - this could be because susceptances are NaN, very small or very large"
        + " or the grid is not connected and hence the node-node susceptance matrix is singular."
    )
    return ptdf


def get_susceptance_matrices(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    susceptances: Float[np.ndarray, " n_branch"],
    number_of_branches: int,
    number_of_busses: int,
) -> tuple[spmatrix, spmatrix]:
    """Build the B matrices necessary for DC power flow and PTDF calculation.

    Taken from makeBdc of pandapower https://github.com/e2nIEE/pandapower/blob/v2.0.1/pandapower/pypower/makeBdc.py
    The bus real power injections are related to bus voltage angles by::
        P = node_node_susceptance * Va + PBusinj
    The real power flows at the from end the lines are related to the bus
    voltage angles by::
        Pf = branch_node_susceptance * Va + Pfinj

    Parameters
    ----------
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-bus
        of a branch can be set to the second bus of a substation.
    susceptances : Float[np.ndarray, " n_branch"]
        Vector with the susceptances for the branches.
    number_of_branches: int
        How many branches are in the system
    number_of_busses: int
        How many busbars are in the system

    Returns
    -------
    node_node_susceptance: Float[np.ndarray, " n_node n_node"]
        NodexNode Matrix allowing for the calculation of the nodal injection
    branch_node_susceptance: Float[np.ndarray, " n_branch n_node"]
        BranchxNode Matrix allowing for the calculation of the power flow
    """
    # build connection matrix Cft = Cf - Ct for line and from - to busses
    connectivity_matrix = get_connectivity_matrix(from_node, to_node, number_of_branches, number_of_busses)

    # Build branch-to-node susceptance matrix
    branch_node_susceptance = csc_matrix(connectivity_matrix.multiply(susceptances[:, np.newaxis]))

    # build node_node_susceptance
    node_node_susceptance = connectivity_matrix.T * branch_node_susceptance
    return node_node_susceptance, branch_node_susceptance


def get_connectivity_matrix(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_branches: int,
    number_of_busses: int,
    directed: bool = True,
) -> spmatrix:
    """Get connectivity matrix (n_branchxn_bus) of the grid.

    With values 1 if branch is connected to node, 0 otherwise

    Parameters
    ----------
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-bus
        of a branch can be set to the second bus of a substation.
    number_of_branches: int
        The number of branches in the grid
    number_of_busses: int
        The number of busses in the grid
    directed: bool
        Whether the grid is directed or not. If true, the matrix will be
        asymmetric. If false, the matrix will be symmetric.

    Returns
    -------
    spmatrix [Shape[" n_branch, n_node"], int]
        The connectivity matrix of the grid in sparse format
    """
    # Build connectivity matrix without outaged branches
    if directed:
        data = np.r_[np.ones(number_of_branches), -np.ones(number_of_branches)]
    else:
        data = np.r_[np.ones(number_of_branches), np.ones(number_of_branches)]
    row_indices = np.r_[range(number_of_branches), range(number_of_branches)]
    column_indices = np.r_[from_node, to_node]
    connectivity_matrix = csc_matrix(
        (data, (row_indices, column_indices)),
        shape=(number_of_branches, number_of_busses),
        dtype=int,
    )
    return connectivity_matrix


def get_extended_ptdf(
    ptdf: Float[np.ndarray, " n_branch n_node"],
    relevant_node_mask: Bool[np.ndarray, " n_node"],
) -> Float[np.ndarray, " n_branch n_node_ext"]:
    """Get the extended PTDF including the 2nd busbar for each relevant busbar in the grid model.

    The extended busses are added as new columns in the end
    The new columns are copys of the old.

    Parameters
    ----------
    ptdf: Float[np.ndarray, " n_branch n_node"]
        The unextended PTDF of the grid
    relevant_node_mask: Bool[np.ndarray, " n_node"]
        A mask of the relevant nodes in the network

    Returns
    -------
    Float[np.ndarray, " n_branch n_node_ext"]
        The extended ptdf
    """
    # Build connectivity matrix without outaged branches
    number_of_relevant_nodes = relevant_node_mask.sum()
    number_of_branches, number_of_nodes = ptdf.shape
    extended_number_of_nodes = number_of_nodes + number_of_relevant_nodes

    ptdf_ext = np.zeros(shape=(number_of_branches, extended_number_of_nodes), dtype=float)
    ptdf_ext[:, :number_of_nodes] = ptdf
    ptdf_ext[:, number_of_nodes:] = ptdf[:, relevant_node_mask]
    return ptdf_ext


def get_extended_nodal_injections(
    nodal_injection: Optional[Float[np.ndarray, " n_timestep n_node"]],
    relevant_node_mask: Bool[np.ndarray, " n_node"],
) -> Optional[Float[np.ndarray, " n_timestep n_node_plus_b"]]:
    """Get the extended nodal_injections including the 2nd busbar for each relevant busbar in the grid model.

    Set injection=0 for the added nodes
    The extended nodal injections are added as new columns in the end

    Parameters
    ----------
    nodal_injection: Optional[Float[np.ndarray, " n_timestep n_node"]]
        The unextended nodal injections of the grid. Can be none if not computed yed
    relevant_node_mask: Bool[np.ndarray, " n_node"]
        A mask of the relevant nodes in the network

    Returns
    -------
    Optional[Float[np.ndarray, " n_timestep n_node_plus_b"]]
        The extended nodal injections or None if not computed yet
    """
    if nodal_injection is None:
        return nodal_injection

    n_timestep = nodal_injection.shape[0]
    n_relevant_nodes = relevant_node_mask.sum()
    nodal_injection = np.concatenate(
        [
            nodal_injection,
            np.zeros((n_timestep, n_relevant_nodes), dtype=float),
        ],
        axis=1,
    )
    return nodal_injection
