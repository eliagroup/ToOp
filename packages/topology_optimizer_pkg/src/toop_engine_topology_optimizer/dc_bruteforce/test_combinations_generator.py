import jax.numpy as jnp
from toop_engine_topology_optimizer.dc.dc_bruteforce.brute_force import next_subsample
import itertools


def test_combinations_generator():
    n = 4
    p = 2

    def generate_true_combinations(n, p):

        res = []
        for k in range(p+1):
            list_combi = list(itertools.combinations(list(range(n)), k))
            sub_res_arr = jnp.full((len(list_combi), p), -1, dtype=jnp.int64)
            res_arr = sub_res_arr.at[:, :k].set(jnp.array(list_combi, dtype=jnp.int64))
            res.append(res_arr)
        res = jnp.concatenate(res, axis=0)
        return res

    true_combis = generate_true_combinations(n, p)

    combi = jnp.array([[0] * p], dtype=int)
    generated_combis = []

    while True:
        generated_combis.append(combi)
        combi = next_subsample(combi, n + 1, p)
        if jnp.all(combi == 0):
            break

    generated_combis = jnp.concat(generated_combis, axis=0) - 1

    true_combis = jnp.sort(true_combis, axis=1)
    generated_combis = jnp.sort(generated_combis, axis=1)
    assert jnp.array_equal(true_combis, generated_combis)