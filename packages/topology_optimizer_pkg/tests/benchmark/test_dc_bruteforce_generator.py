# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from time import perf_counter

from toop_engine_topology_optimizer.dc_bruteforce.generator import iter_workset, take_workset_chunk


def test_dc_bruteforce_generator_benchmark(record_property) -> None:
    split_action_groups = tuple(tuple(range(group_start, group_start + 8)) for group_start in range(0, 40 * 8, 8))
    action_start_indices = tuple(group_start for group_start in range(0, 40 * 8, 8))
    n_actions_per_sub = (8,) * 40
    chunk_size = 50_000

    start = perf_counter()
    chunk = take_workset_chunk(
        iter_workset(
            action_start_indices=action_start_indices,
            n_actions_per_sub=n_actions_per_sub,
            max_num_splits=3,
            n_disconnectable_branches=20,
            max_num_disconnections=2,
        ),
        chunk_size,
    )
    elapsed_seconds = perf_counter() - start
    throughput = chunk_size / elapsed_seconds

    record_property("dc_bruteforce_generator_chunk_size", chunk_size)
    record_property("dc_bruteforce_generator_elapsed_seconds", elapsed_seconds)
    record_property("dc_bruteforce_generator_topologies_per_second", throughput)

    assert len(chunk) == chunk_size
    assert elapsed_seconds < 2.0
