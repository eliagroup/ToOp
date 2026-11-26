# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Compare split MODF with pypowsybl DC for every island-creating outage set on small grids.

The PTDF is not taken from the preprocessing pipeline. It is rebuilt from the DC loadflow of the
grid itself, so that any grid pypowsybl can create can be used as a test case.
"""

import itertools
from collections.abc import Callable

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
import pypowsybl.loadflow as plf
import pypowsybl.network as pn
import pytest
from toop_engine_dc_solver.jax.split_modf_main_grid import compute_split_modf_main_grid

TOL = 1e-6
BRANCH_COLUMNS = ["bus1_id", "bus2_id", "connected1", "connected2"]
DISTRIBUTED_SLACK = plf.Parameters(distributed_slack=True, balance_type=plf.BalanceType.PROPORTIONAL_TO_GENERATION_P)
SINGLE_SLACK = plf.Parameters(distributed_slack=False)


def _branch_flows(net: pn.Network) -> pd.Series:
    return pd.concat([net.get_lines(attributes=["p1"]), net.get_2_windings_transformers(attributes=["p1"])]).p1


def _run_dc(net: pn.Network, params: plf.Parameters) -> None:
    all_components = plf.Parameters(
        distributed_slack=params.distributed_slack,
        balance_type=params.balance_type,
        component_mode=plf.ComponentMode.ALL_CONNECTED,
    )
    plf.run_dc(net, all_components)


def _dc_state(
    net: pn.Network, params: plf.Parameters, branches: pd.DataFrame, incidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the DC branch flows and angle differences of the intact grid."""
    _run_dc(net, params)
    angles = np.deg2rad(net.get_buses().v_angle.loc[net.get_buses().index].to_numpy())
    return _branch_flows(net).loc[branches.index].to_numpy(), incidence @ angles


def _build_model(factory: Callable[[], pn.Network], params: plf.Parameters) -> dict:
    """Build PTDF, base flows and nodal injections of the intact grid from pypowsybl DC loadflows."""
    net = factory()
    branches = pd.concat(
        [net.get_lines(attributes=BRANCH_COLUMNS), net.get_2_windings_transformers(attributes=BRANCH_COLUMNS)]
    )
    branches = branches[branches.connected1 & branches.connected2]
    buses = net.get_buses().index
    from_node = buses.get_indexer(branches.bus1_id)
    to_node = buses.get_indexer(branches.bus2_id)
    incidence = np.zeros((len(branches), len(buses)))
    incidence[np.arange(len(branches)), from_node] = 1.0
    incidence[np.arange(len(branches)), to_node] = -1.0

    results = plf.run_dc(net, SINGLE_SLACK)
    slack = buses.get_loc(results[0].reference_bus_id)
    fit_flow, fit_angle = _dc_state(net, SINGLE_SLACK, branches, incidence)
    _run_dc(net, params)
    n_0_flow = _branch_flows(net).loc[branches.index].to_numpy()

    # Fit the susceptances from perturbed injections; phase shifter offsets cancel in the differences
    rng = np.random.default_rng(0)
    d_flow, d_angle = [], []
    for _ in range(6):
        perturbed = factory()
        gens, loads = perturbed.get_generators(), perturbed.get_loads()
        perturbed.update_generators(
            id=gens.index.tolist(), target_p=(gens.target_p * rng.uniform(0.5, 1.5, len(gens))).tolist()
        )
        perturbed.update_loads(id=loads.index.tolist(), p0=(loads.p0 * rng.uniform(0.5, 1.5, len(loads))).tolist())
        flow, angle = _dc_state(perturbed, SINGLE_SLACK, branches, incidence)
        d_flow.append(flow - fit_flow)
        d_angle.append(angle - fit_angle)
    d_flow, d_angle = np.array(d_flow), np.array(d_angle)
    susceptance = (d_flow * d_angle).sum(0) / (d_angle**2).sum(0)

    keep = np.arange(len(buses)) != slack
    bus_matrix = incidence.T @ (susceptance[:, None] * incidence)
    ptdf = np.zeros_like(incidence)
    ptdf[:, keep] = (susceptance[:, None] * incidence[:, keep]) @ np.linalg.inv(bus_matrix[np.ix_(keep, keep)])
    return {
        "branches": branches,
        "n_buses": len(buses),
        "from_node": from_node,
        "to_node": to_node,
        "slack": slack,
        "n_0_flow": n_0_flow,
        "nodal_injection": incidence.T @ n_0_flow,
        "ptdf": ptdf,
    }


def _island_creating_outages(model: dict, max_outages: int) -> list[np.ndarray]:
    """Enumerate all branch outage sets that increase the number of electrical islands."""
    edges = list(zip(model["from_node"].tolist(), model["to_node"].tolist(), strict=True))

    def n_components(removed: set[int]) -> int:
        graph = nx.Graph()
        graph.add_nodes_from(range(model["n_buses"]))
        graph.add_edges_from(edge for index, edge in enumerate(edges) if index not in removed)
        return nx.number_connected_components(graph)

    intact = n_components(set())
    return [
        np.array(outage_set)
        for n_outages in range(1, max_outages + 1)
        for outage_set in itertools.combinations(range(len(edges)), n_outages)
        if n_components(set(outage_set)) > intact
    ]


def _open_branches(net: pn.Network, branch_ids: list[str]) -> None:
    lines = net.get_lines().index
    for branch_id in branch_ids:
        update = net.update_lines if branch_id in lines else net.update_2_windings_transformers
        update(id=branch_id, connected1=False, connected2=False)


def _compare_with_powsybl(factory: Callable[[], pn.Network], params: plf.Parameters, max_outages: int) -> list[str]:
    """Return a description of every island-creating outage set where split MODF differs from pypowsybl DC."""
    model = _build_model(factory, params)
    branches = model["branches"]
    outage_sets = _island_creating_outages(model, max_outages)
    assert len(outage_sets) > 0

    split_flow, _main_bus, main_monitored, _imbalance, _rank_loss = compute_split_modf_main_grid(
        ptdf=jnp.asarray(model["ptdf"]),
        from_node=jnp.asarray(model["from_node"], dtype=jnp.int32),
        to_node=jnp.asarray(model["to_node"], dtype=jnp.int32),
        nodal_injections=jnp.asarray(model["nodal_injection"][None, :]),
        n_0_flow=jnp.asarray(model["n_0_flow"][None, :]),
        outages=jnp.asarray(
            np.stack([np.pad(s, (0, max_outages - len(s)), constant_values=-1) for s in outage_sets]), dtype=jnp.int32
        ),
        branches_monitored=jnp.arange(len(branches), dtype=jnp.int32),
        slack_bus=jnp.asarray(model["slack"], dtype=jnp.int32),
    )
    split_flow = np.asarray(split_flow[0])
    main_monitored = np.asarray(main_monitored)

    deviations = []
    for case, outage_set in enumerate(outage_sets):
        outaged_ids = branches.index[outage_set].tolist()
        cut_net = factory()
        _open_branches(cut_net, outaged_ids)
        _run_dc(cut_net, params)
        reference = _branch_flows(cut_net).reindex(branches.index).to_numpy()
        reference[outage_set] = 0.0
        difference = np.abs(split_flow[case] - reference)[main_monitored[case]]
        if np.any(np.isnan(difference)) or np.any(difference > TOL):
            deviations.append(f"{outaged_ids}: max |diff| {np.nanmax(difference):.3f} MW")
    return deviations


@pytest.mark.parametrize(
    ("factory", "max_outages"),
    [
        pytest.param(pn.create_ieee14, 3, id="ieee14-n3"),
        pytest.param(pn.create_ieee30, 2, id="ieee30-n2"),
    ],
)
def test_split_modf_matches_powsybl_dc_for_island_creating_outages_with_single_slack(
    factory: Callable[[], pn.Network], max_outages: int
) -> None:
    """Match pypowsybl DC for all island-creating N-k outages when the slack bus absorbs the imbalance."""
    assert _compare_with_powsybl(factory, SINGLE_SLACK, max_outages) == []


@pytest.mark.xfail(
    strict=True,
    reason="Split MODF assigns the imbalance of discarded islands to the original slack bus, "
    "pypowsybl distributes it over the generators of the retained island",
)
def test_split_modf_matches_powsybl_dc_for_island_creating_outages_with_distributed_slack() -> None:
    """Match pypowsybl DC for all island-creating N-k outages with the production distributed slack."""
    assert _compare_with_powsybl(pn.create_ieee14, DISTRIBUTED_SLACK, 3) == []
