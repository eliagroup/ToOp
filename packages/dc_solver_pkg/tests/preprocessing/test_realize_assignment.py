# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax.numpy as jnp
import numpy as np
import pytest
from toop_engine_dc_solver.postprocess.realize_assignment import (
    compute_switching_table,
    realise_ba_to_physical_topo_per_station_jax,
)
from toop_engine_dc_solver.preprocess.action_set import make_action_repo
from toop_engine_dc_solver.preprocess.preprocess_switching import (
    make_optimal_separation_set,
)
from toop_engine_interfaces.asset_topology import Busbar, BusbarCoupler, Station, SwitchableAsset


def test_realize_ba_to_physical_topo_per_station_simple():
    station = Station(
        grid_model_id="teststation",
        busbars=[Busbar(grid_model_id="BB1", int_id=1), Busbar(grid_model_id="BB2", int_id=2)],
        couplers=[BusbarCoupler(grid_model_id="BC1", busbar_from_id=1, busbar_to_id=2, open=False)],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
            SwitchableAsset(grid_model_id="line5"),
        ],
        asset_connectivity=np.ones((2, 5), dtype=bool),
        asset_switching_table=np.array([[True, True, False, False, False], [False, False, True, True, True]]),
    )

    action_set = np.array(
        [
            [False, False, False, False, False],
            [True, True, False, False, False],
            [False, False, True, True, True],
            [False, False, True, False, True],
        ]
    )
    separation_set_station = make_optimal_separation_set(station)

    realized_stations, updated_action_set, busbar_a_mappings, reassignment_distance = (
        realise_ba_to_physical_topo_per_station_jax(
            local_branch_action_set=action_set,
            station=station,
            separation_set_info=separation_set_station,
            choice_heuristic="first",
        )
    )

    assert len(realized_stations) == len(action_set)
    assert np.array_equal(updated_action_set, action_set)

    # The first action is the unsplit action
    assert realized_stations[0] == station
    assert reassignment_distance[0] == 0
    assert busbar_a_mappings[0] == [0, 1]

    # The second action should be without any reassignments
    assert realized_stations[1].couplers[0].open is True
    assert np.array_equal(realized_stations[1].asset_switching_table, station.asset_switching_table)
    assert reassignment_distance[1] == 0
    assert busbar_a_mappings[1] == [1]

    # The third action is again without reassignments but also without inversion
    assert realized_stations[2].couplers[0].open is True
    assert np.array_equal(realized_stations[2].asset_switching_table, station.asset_switching_table)
    assert reassignment_distance[2] == 0
    assert busbar_a_mappings[2] == [0]

    # The fourth action has one reassignment
    assert realized_stations[3].couplers[0].open is True
    expected_switching_table = np.array([[True, True, False, True, False], [False, False, True, False, True]])
    assert np.array_equal(realized_stations[3].asset_switching_table, expected_switching_table)
    assert reassignment_distance[3] == 2
    assert busbar_a_mappings[3] == [0]

    # Also recompute the second action directly
    (
        current_switching_table,
        chosen_coupler_state,
        chosen_busbar_mapping,
        el_reassignment_distance,
        phy_reassignment_distance,
        failed_assignments,
    ) = compute_switching_table(
        local_action=jnp.array(action_set[1]),
        current_coupler_state=jnp.array([True]),
        separation_set=jnp.array(station.asset_switching_table[None]),
        coupler_states=jnp.array([[False]]),
        busbar_mapping=jnp.array([[False, True]]),
        current_switching_table=jnp.array(station.asset_switching_table),
        asset_connectivity=jnp.array(station.asset_connectivity),
        choice_heuristic="first",
    )
    assert np.array_equal(current_switching_table, station.asset_switching_table)
    assert np.array_equal(chosen_coupler_state, np.array([False]))
    assert np.array_equal(chosen_busbar_mapping, np.array([True, False]))  # Inverted
    assert el_reassignment_distance == 0
    assert phy_reassignment_distance == 0
    assert failed_assignments.item() is False

    # With 2 busbars, the "first" heuristic should be equivalent to "least_connected"
    realized_stations_2, updated_action_set_2, busbar_a_mappings_2, reassignment_distance_2 = (
        realise_ba_to_physical_topo_per_station_jax(
            local_branch_action_set=action_set,
            station=station,
            separation_set_info=separation_set_station,
            choice_heuristic="least_connected_busbar",
        )
    )
    assert realized_stations_2 == realized_stations
    assert np.array_equal(updated_action_set_2, updated_action_set)
    assert busbar_a_mappings_2 == busbar_a_mappings
    assert reassignment_distance_2 == reassignment_distance


def test_realize_ba_to_physical_topo_per_station_3_busbars():
    station = Station(
        grid_model_id="teststation",
        busbars=[
            Busbar(grid_model_id="BB1", int_id=1),
            Busbar(grid_model_id="BB2", int_id=2),
            Busbar(grid_model_id="BB3", int_id=3),
        ],
        couplers=[
            BusbarCoupler(grid_model_id="BC1", busbar_from_id=1, busbar_to_id=2, open=False),
            BusbarCoupler(grid_model_id="BC2", busbar_from_id=2, busbar_to_id=3, open=False),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
            SwitchableAsset(grid_model_id="line5"),
        ],
        asset_connectivity=np.ones((3, 5), dtype=bool),
        asset_switching_table=np.array(
            [[True, True, False, False, False], [False, False, True, True, False], [False, False, False, False, True]]
        ),
    )

    action_set = np.array(
        [
            [False, False, False, False, False],
            [True, True, False, False, False],
            [False, False, True, True, True],
            [False, False, True, True, False],
            [False, True, True, False, True],
            [False, True, True, True, True],
        ]
    )
    separation_set_station = make_optimal_separation_set(station)

    realized_stations, updated_action_set, busbar_a_mappings, reassignment_distance = (
        realise_ba_to_physical_topo_per_station_jax(
            local_branch_action_set=action_set,
            station=station,
            separation_set_info=separation_set_station,
            choice_heuristic="first",
        )
    )

    assert len(realized_stations) == len(action_set)
    assert np.array_equal(updated_action_set, action_set)

    # The first action is the unsplit action
    assert realized_stations[0] == station
    assert reassignment_distance[0] == 0
    assert busbar_a_mappings[0] == [0, 1, 2]

    # The second action should be without any reassignments
    assert realized_stations[1].couplers[0].open is True
    assert realized_stations[1].couplers[1].open is False
    assert np.array_equal(realized_stations[1].asset_switching_table, station.asset_switching_table)
    assert reassignment_distance[1] == 0
    assert busbar_a_mappings[1] == [1, 2]

    # The third action is again without reassignments but also without inversion
    assert realized_stations[2].couplers[0].open is True
    assert realized_stations[2].couplers[1].open is False
    assert np.array_equal(realized_stations[2].asset_switching_table, station.asset_switching_table)
    assert reassignment_distance[2] == 0
    assert busbar_a_mappings[2] == [0]

    # The fourth action has one reassignment
    assert realized_stations[3].couplers[0].open is True
    assert realized_stations[3].couplers[1].open is False
    expected_switching_table = np.array(
        [[True, True, False, False, True], [False, False, True, True, False], [False, False, False, False, False]]
    )
    assert np.array_equal(realized_stations[3].asset_switching_table, expected_switching_table)
    assert reassignment_distance[3] == 2
    assert busbar_a_mappings[3] == [0]

    # The fifth action has two reassignments
    assert realized_stations[4].couplers[0].open is True
    assert realized_stations[4].couplers[1].open is False
    expected_switching_table = np.array(
        [[True, False, False, True, False], [False, True, True, False, False], [False, False, False, False, True]]
    )
    assert np.array_equal(realized_stations[4].asset_switching_table, expected_switching_table)
    assert reassignment_distance[4] == 4
    assert busbar_a_mappings[4] == [0]

    # The sixth action is interesting because it will lead to different results depending on the heuristic
    # The second asset will land on busbar BB2 if first is used but on busbar BB3 if least_connected is used
    assert realized_stations[5].couplers[0].open is True
    assert realized_stations[5].couplers[1].open is False

    expected_switching_table_first = np.array(
        [[True, False, False, False, False], [False, True, True, True, False], [False, False, False, False, True]]
    )
    expected_switching_table_least_connected = np.array(
        [[True, False, False, False, False], [False, False, True, True, False], [False, True, False, False, True]]
    )
    assert np.array_equal(realized_stations[5].asset_switching_table, expected_switching_table_first)
    assert reassignment_distance[5] == 2
    assert busbar_a_mappings[5] == [0]

    # Recompute with "least_connected_busbar" heuristic
    realized_stations_2, updated_action_set_2, busbar_a_mappings_2, reassignment_distance_2 = (
        realise_ba_to_physical_topo_per_station_jax(
            local_branch_action_set=action_set,
            station=station,
            separation_set_info=separation_set_station,
            choice_heuristic="least_connected_busbar",
        )
    )

    assert np.array_equal(updated_action_set_2, updated_action_set)
    assert realized_stations_2[5].couplers[0].open is True
    assert realized_stations_2[5].couplers[1].open is False
    assert np.array_equal(realized_stations_2[5].asset_switching_table, expected_switching_table_least_connected)
    assert reassignment_distance_2[5] == 2
    assert busbar_a_mappings_2[5] == [0]

    # Recompute with "most_connected_busbar" heuristic
    # Most connected will yield the same result as first
    realized_stations_3, updated_action_set_3, busbar_a_mappings_3, reassignment_distance_3 = (
        realise_ba_to_physical_topo_per_station_jax(
            local_branch_action_set=action_set,
            station=station,
            separation_set_info=separation_set_station,
            choice_heuristic="most_connected_busbar",
        )
    )

    assert np.array_equal(updated_action_set_3, updated_action_set)
    assert realized_stations_3[5].couplers[0].open is True
    assert realized_stations_3[5].couplers[1].open is False
    assert np.array_equal(realized_stations_3[5].asset_switching_table, expected_switching_table_first)
    assert reassignment_distance_3[5] == 2
    assert busbar_a_mappings_3[5] == [0]


# TODO: set timeout to 40 and run without xdist
@pytest.mark.xdist_group("performance")
@pytest.mark.timeout(80)
def test_realize_ba_to_physical_topo_per_station_large():
    n_assets = 20  # Should give around half a million actions
    np.random.seed(42)  # For reproducibility
    busbar_choices = np.random.choice(3, size=(n_assets,), replace=True)
    switching_table = np.zeros((3, n_assets), dtype=bool)
    switching_table[busbar_choices, range(n_assets)] = True

    station = Station(
        grid_model_id="teststation",
        busbars=[
            Busbar(grid_model_id="BB1", int_id=1),
            Busbar(grid_model_id="BB2", int_id=2),
            Busbar(grid_model_id="BB3", int_id=3),
        ],
        couplers=[
            BusbarCoupler(grid_model_id="BC1", busbar_from_id=1, busbar_to_id=2, open=False),
            BusbarCoupler(grid_model_id="BC2", busbar_from_id=2, busbar_to_id=3, open=False),
        ],
        assets=[SwitchableAsset(grid_model_id=f"line{i + 1}") for i in range(n_assets)],
        asset_connectivity=np.ones((3, n_assets), dtype=bool),
        asset_switching_table=switching_table,
    )
    separation_set_station = make_optimal_separation_set(station)

    action_set = make_action_repo(20, separation_set_station.separation_set)

    realized_stations, updated_action_set, busbar_a_mappings, reassignment_distance = (
        realise_ba_to_physical_topo_per_station_jax(
            local_branch_action_set=action_set,
            station=station,
            separation_set_info=separation_set_station,
            choice_heuristic="least_connected_busbar",
        )
    )

    assert len(realized_stations) == len(action_set)
    assert np.array_equal(updated_action_set, action_set)
    assert len(busbar_a_mappings) == len(action_set)
    assert len(reassignment_distance) == len(action_set)


def test_realize_ba_to_physical_topo_per_station_limited_connectivity():
    station = Station(
        grid_model_id="teststation",
        busbars=[
            Busbar(grid_model_id="BB1", int_id=1),
            Busbar(grid_model_id="BB2", int_id=2),
            Busbar(grid_model_id="BB3", int_id=3),
        ],
        couplers=[
            BusbarCoupler(grid_model_id="BC1", busbar_from_id=1, busbar_to_id=2, open=False),
            BusbarCoupler(grid_model_id="BC2", busbar_from_id=2, busbar_to_id=3, open=False),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
            SwitchableAsset(grid_model_id="line5"),
        ],
        asset_connectivity=np.array(
            [[True, True, True, True, False], [True, True, True, True, False], [True, True, False, False, True]]
        ),
        asset_switching_table=np.array(
            [[True, True, False, False, False], [False, False, True, True, False], [False, False, False, False, True]]
        ),
    )

    action_set = np.array(
        [
            [False, False, False, False, False],
            [True, True, False, False, False],
            [False, False, True, True, True],
            [True, True, False, False, True],
            [False, False, True, True, False],
            [False, False, True, False, False],
        ]
    )
    separation_set_station = make_optimal_separation_set(station)

    realized_stations, updated_action_set, busbar_a_mappings, reassignment_distance = (
        realise_ba_to_physical_topo_per_station_jax(
            local_branch_action_set=action_set,
            station=station,
            separation_set_info=separation_set_station,
            choice_heuristic="first",
            validate=True,
        )
    )

    assert len(realized_stations) == len(action_set)
    assert np.array_equal(updated_action_set, action_set)

    assert realized_stations[1].couplers[0].open is True
    assert realized_stations[1].couplers[1].open is False
    assert np.array_equal(realized_stations[1].asset_switching_table, station.asset_switching_table)
    assert reassignment_distance[1] == 0
    assert busbar_a_mappings[1] == [1, 2]

    assert realized_stations[2].couplers[0].open is True
    assert realized_stations[2].couplers[1].open is False
    assert np.array_equal(realized_stations[2].asset_switching_table, station.asset_switching_table)
    assert reassignment_distance[2] == 0
    assert busbar_a_mappings[2] == [0]

    # The fourth action has a lot of reassignments to be possible
    expected_switching = np.array(
        [[False, False, True, True, False], [True, True, False, False, False], [False, False, False, False, True]]
    )
    assert np.array_equal(realized_stations[3].asset_switching_table, expected_switching)
    assert realized_stations[3].couplers[0].open is True
    assert realized_stations[3].couplers[1].open is False
    assert reassignment_distance[3] == np.logical_xor(expected_switching, station.asset_switching_table).sum()
    assert busbar_a_mappings[3] == [0]

    # Fourth action is the same as the fifth, but inverted
    assert np.array_equal(realized_stations[4].asset_switching_table, expected_switching)
    assert realized_stations[4].couplers[0].open is True
    assert realized_stations[4].couplers[1].open is False
    assert reassignment_distance[4] == np.logical_xor(expected_switching, station.asset_switching_table).sum()
    assert busbar_a_mappings[4] == [1, 2]

    # Sixth action is possible by putting asset[3] on busbar[0]
    expected_switching = np.array(
        [[False, False, True, False, False], [True, True, False, True, False], [False, False, False, False, True]]
    )
    assert np.array_equal(realized_stations[5].asset_switching_table, expected_switching)
    assert realized_stations[5].couplers[0].open is True
    assert realized_stations[5].couplers[1].open is False


def test_realize_ba_to_physical_topo_per_station_invalid_actions():
    station = Station(
        grid_model_id="teststation",
        busbars=[
            Busbar(grid_model_id="BB1", int_id=1),
            Busbar(grid_model_id="BB2", int_id=2),
            Busbar(grid_model_id="BB3", int_id=3),
        ],
        couplers=[
            BusbarCoupler(grid_model_id="BC1", busbar_from_id=1, busbar_to_id=2, open=False),
            BusbarCoupler(grid_model_id="BC2", busbar_from_id=2, busbar_to_id=3, open=False),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
            SwitchableAsset(grid_model_id="line5"),
        ],
        asset_connectivity=np.array(
            [[True, True, False, False, True], [True, True, True, True, True], [True, True, False, False, True]]
        ),
        asset_switching_table=np.array(
            [[True, True, False, False, False], [False, False, True, True, False], [False, False, False, False, True]]
        ),
    )
    separation_set_station = make_optimal_separation_set(station)

    action_set = np.array(
        [
            [False, False, False, False, False],
            [True, True, False, False, False],
            [False, False, True, True, True],
            [True, True, False, False, True],
            [False, False, True, True, False],
            [False, False, True, False, False],  # Impossible
        ]
    )

    realized_stations, updated_action_set, busbar_a_mappings, reassignment_distance = (
        realise_ba_to_physical_topo_per_station_jax(
            local_branch_action_set=action_set,
            station=station,
            separation_set_info=separation_set_station,
            choice_heuristic="first",
            validate=True,
        )
    )

    assert len(realized_stations) == len(action_set) - 1
    assert np.array_equal(updated_action_set, action_set[:-1])
    assert len(busbar_a_mappings) == len(action_set) - 1
    assert len(reassignment_distance) == len(action_set) - 1


@pytest.mark.xfail(reason="These edge cases are not yet handled")
def test_realize_ba_to_physical_topo_per_station_invalid_actions_hard():
    station = Station(
        grid_model_id="teststation",
        busbars=[
            Busbar(grid_model_id="BB1", int_id=1),
            Busbar(grid_model_id="BB2", int_id=2),
            Busbar(grid_model_id="BB3", int_id=3),
        ],
        couplers=[
            BusbarCoupler(grid_model_id="BC1", busbar_from_id=1, busbar_to_id=2, open=False),
            BusbarCoupler(grid_model_id="BC2", busbar_from_id=2, busbar_to_id=3, open=False),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
            SwitchableAsset(grid_model_id="line5"),
        ],
        asset_connectivity=np.array(
            [[True, True, False, False, False], [True, True, True, True, False], [True, True, False, False, True]]
        ),
        asset_switching_table=np.array(
            [[True, True, False, False, False], [False, False, True, True, False], [False, False, False, False, True]]
        ),
    )

    action_set = np.array(
        [
            [False, False, False, False, False],
            [True, True, False, False, False],
            [False, False, True, True, True],
            [True, True, False, False, True],  # Possible if busbar b = [2] and asset 0 and 1 assigned to 2
            [False, False, True, True, False],  # Same as above but inverted
            [False, False, True, False, False],  # Impossible
        ]
    )
    separation_set_station = make_optimal_separation_set(station)

    realized_stations, updated_action_set, busbar_a_mappings, reassignment_distance = (
        realise_ba_to_physical_topo_per_station_jax(
            local_branch_action_set=action_set,
            station=station,
            separation_set_info=separation_set_station,
            choice_heuristic="first",
        )
    )

    assert len(realized_stations) == len(action_set) - 1
    assert np.array_equal(updated_action_set, action_set[:-1])
    assert len(busbar_a_mappings) == len(action_set) - 1
    assert len(reassignment_distance) == len(action_set) - 1

    assert realized_stations[1].couplers[0].open is True
    assert realized_stations[1].couplers[1].open is False
    assert np.array_equal(realized_stations[1].asset_switching_table, station.asset_switching_table)
    assert reassignment_distance[1] == 0
    assert busbar_a_mappings[1] == [1, 2]
