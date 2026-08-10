# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Extract pandapower branch result metrics per contingency as a flat polars frame."""

import pandera as pa
import pandera.typing.polars as patpl
import polars as pl
from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.result_constants import (
    BRANCH_TYPES,
    MAX_AMOUNT_OF_SIDES,
    ResultConstants,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.branch_res_power_columns import (
    branch_res_power_columns,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
)
from toop_engine_interfaces.loadflow_results import BranchSide
from toop_engine_interfaces.loadflow_results_polars import BranchResultSchemaPolars

#: Value for numeric result columns on the failed (non-converged) path.
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


@pa.check_types
def get_branch_results_polars(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
    constants: ResultConstants,
) -> patpl.DataFrame[BranchResultSchemaPolars]:
    """Branch results for one contingency as a polars frame.

    One row per branch terminal, with ``timestep``/``contingency``/``element``/``side`` as
    columns. ``p`` and ``q`` are blanked wherever ``i`` is null, matching the convention
    that an unsupplied terminal reports no flow. Reads the ``res_*_polars`` snapshots and the
    per-job constants (element ids, rated currents) precomputed in :class:`ResultConstants`.
    """
    frames = []
    for branch_type in BRANCH_TYPES:
        res_table = net[f"res_{branch_type}_polars"]
        if res_table.is_empty():
            continue

        uids = constants.element_uids[branch_type]
        for side in range(MAX_AMOUNT_OF_SIDES):
            try:
                columns = branch_res_power_columns(branch_type, side=side)
            except IndexError:
                break  # this branch type has no further sides

            present = [column for column in columns if column in res_table.columns]
            if not present:
                continue

            frame = res_table.select(present).rename(
                dict(zip(present, ["p", "q", "i", "loading"], strict=False)),
            )
            frame = frame.with_columns(
                pl.lit(timestep, dtype=pl.Int64).alias("timestep"),
                pl.lit(contingency.unique_id).alias("contingency"),
                pl.Series("element", uids, dtype=pl.String),
                pl.lit(side + 1, dtype=pl.Int64).alias("side"),
            )

            if "i" in frame.columns:
                # pandapower reports kA; the schema wants A.
                frame = frame.with_columns((pl.col("i") * 1000).alias("i"))
                i_max = constants.i_max.get((branch_type, side))
                if i_max is not None:
                    frame = frame.with_columns((pl.col("i") / pl.Series("i_max", i_max * 1000)).alias("loading"))
                # A terminal with no current carries no meaningful power either.
                frame = frame.with_columns(
                    pl.when(pl.col("i").is_null()).then(None).otherwise(pl.col("p")).alias("p"),
                    pl.when(pl.col("i").is_null()).then(None).otherwise(pl.col("q")).alias("q"),
                )

            frames.append(frame)

    branch_results = pl.concat(frames, how="diagonal")
    return branch_results.with_columns(
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    )


@pa.check_types
def get_failed_branch_results_polars(
    timestep: int,
    failed_outages: list[str],
    monitored_2_end_branches: list[str],
    monitored_3_end_branches: list[str],
) -> patpl.DataFrame[BranchResultSchemaPolars]:
    """All-NaN branch results for outages whose load flow did not converge.

    Counterpart of :func:`get_branch_results_polars`: one row per monitored branch terminal
    (two sides for lines/2W trafos, three for 3W trafos), with the electrical columns NaN
    (see :data:`_MISSING`). Same flat layout, so the two concatenate directly.
    """
    two_end = _failed_key_frame(
        timestep, failed_outages, monitored_2_end_branches, [BranchSide.ONE.value, BranchSide.TWO.value]
    )
    three_end = _failed_key_frame(
        timestep,
        failed_outages,
        monitored_3_end_branches,
        [BranchSide.ONE.value, BranchSide.TWO.value, BranchSide.THREE.value],
    )
    return pl.concat([two_end, three_end], how="vertical").select(
        "timestep",
        "contingency",
        "element",
        "side",
        pl.lit(_MISSING, dtype=pl.Float64).alias("p"),
        pl.lit(_MISSING, dtype=pl.Float64).alias("q"),
        pl.lit(_MISSING, dtype=pl.Float64).alias("i"),
        pl.lit(_MISSING, dtype=pl.Float64).alias("loading"),
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    )
