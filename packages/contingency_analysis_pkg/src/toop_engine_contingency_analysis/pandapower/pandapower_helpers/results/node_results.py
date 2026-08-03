# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


"""Extract pandapower bus (node) simulation results per contingency as a flat polars frame."""

import polars as pl
from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.result_constants import (
    MAX_ALLOWED_VM_DEVIATION,
    ResultConstants,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
)

#: Value for numeric result columns on the failed (non-converged) path. NaN, not null: the
#: pandas pipeline this replaced used ``np.nan`` and downstream polars metrics rely on NaN
#: semantics (an all-missing ``max()`` must be NaN, not null which collects to None).
_MISSING = float("nan")


def _failed_key_frame(
    timestep: int,
    contingencies: list[str],
    elements: list[str],
    sides: list[int] | None,
) -> pl.DataFrame:
    """Cross product of ``contingencies`` x ``elements`` (x ``sides``) as key columns.

    Returns a frame with ``timestep``/``contingency``/``element`` (and ``side`` when *sides*
    is given). An empty input on any axis yields a 0-row frame with the columns and dtypes
    preserved, so the caller can build result rows on it unconditionally.
    """
    frame = pl.DataFrame({"contingency": pl.Series(contingencies, dtype=pl.String)}).join(
        pl.DataFrame({"element": pl.Series(elements, dtype=pl.String)}),
        how="cross",
    )
    if sides is not None:
        frame = frame.join(pl.DataFrame({"side": pl.Series(sides, dtype=pl.Int64)}), how="cross")
    return frame.with_columns(pl.lit(timestep, dtype=pl.Int64).alias("timestep"))


def get_node_results_polars(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
    constants: ResultConstants,
) -> pl.DataFrame:
    """Node (bus) results for one contingency as a polars frame.

    Reads the ``res_bus_polars`` snapshot and the per-job constants (base-case voltages,
    bus element ids, voltage levels) precomputed in :class:`ResultConstants`.
    """
    res_bus = net["res_bus_polars"]
    basecase_vm = pl.Series("bc_vm", constants.basecase_vm)

    node_results = res_bus.rename({"vm_pu": "vm", "va_degree": "va", "p_mw": "p", "q_mvar": "q"})
    return node_results.with_columns(
        pl.lit(timestep, dtype=pl.Int64).alias("timestep"),
        pl.lit(contingency.unique_id).alias("contingency"),
        pl.Series("element", constants.element_uids["bus"], dtype=pl.String),
        # Deviation from the base-case voltage, in percent. Still per-unit at this point.
        (((pl.col("vm") - basecase_vm).abs() / basecase_vm) * 100).alias("vm_basecase_deviation"),
        ((pl.col("vm") - 1) / MAX_ALLOWED_VM_DEVIATION).alias("vm_loading"),
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    ).with_columns(
        # Scale to kV only after the per-unit quantities above have been derived.
        (pl.col("vm") * pl.Series("vn_kv", constants.voltage_levels)).alias("vm"),
    )


def get_failed_node_results_polars(
    timestep: int,
    failed_outages: list[str],
    monitored_nodes: list[str],
) -> pl.DataFrame:
    """All-NaN node results for outages whose load flow did not converge.

    Counterpart of :func:`get_node_results_polars`: one row per monitored bus with the
    electrical columns NaN (see :data:`_MISSING`). Same flat layout, so the two concatenate
    directly.
    """
    return _failed_key_frame(timestep, failed_outages, monitored_nodes, sides=None).select(
        "timestep",
        "contingency",
        "element",
        pl.lit(_MISSING, dtype=pl.Float64).alias("vm"),
        pl.lit(_MISSING, dtype=pl.Float64).alias("va"),
        pl.lit(_MISSING, dtype=pl.Float64).alias("p"),
        pl.lit(_MISSING, dtype=pl.Float64).alias("q"),
        pl.lit(_MISSING, dtype=pl.Float64).alias("vm_basecase_deviation"),
        pl.lit(_MISSING, dtype=pl.Float64).alias("vm_loading"),
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    )
