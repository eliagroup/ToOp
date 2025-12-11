"""Helper functions to translate the N-1 definition into a usable format for Powsybl.

This includes translating contingencies, monitored elements and collecting
the necessary data from the network, so this only has to happen once.
"""

import pandera as pa
import pandera.typing.polars as patpl
import polars as pl
import pypowsybl
from beartype.typing import Literal, Optional
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_result_helpers import get_failed_branch_results, get_failed_node_results
from toop_engine_interfaces.loadflow_results import (
    BranchSide,
    ConvergenceStatus,
    VADiffResultSchema,
)
from toop_engine_interfaces.loadflow_results_polars import (
    BranchResultSchemaPolars,
    LoadflowResultTablePolars,
    NodeResultSchemaPolars,
    VADiffResultSchemaPolars,
)

POWSYBL_CONVERGENCE_MAP = {
    pypowsybl.loadflow.ComponentStatus.CONVERGED.value: ConvergenceStatus.CONVERGED.value,
    pypowsybl.loadflow.ComponentStatus.FAILED.value: ConvergenceStatus.FAILED.value,
    pypowsybl.loadflow.ComponentStatus.NO_CALCULATION.value: ConvergenceStatus.NO_CALCULATION.value,
    pypowsybl.loadflow.ComponentStatus.MAX_ITERATION_REACHED.value: ConvergenceStatus.MAX_ITERATION_REACHED.value,
}


@pa.check_types
def get_node_results_polars(
    bus_results: pl.LazyFrame,
    monitored_buses: list[str],
    bus_map: pl.LazyFrame,
    voltage_levels: pl.LazyFrame,
    failed_outages: list[str],
    timestep: int,
    method: Literal["ac", "dc"],
) -> patpl.LazyFrame[NodeResultSchemaPolars]:
    """Get the node results for the given outages and timestep.

    TODO: This is currently faking the sum of p and q at the node

    Parameters
    ----------
    bus_results : pl.LazyFrame
        The bus results from the powsybl security analysis
    monitored_buses : list[str]
        The list of monitored buses to get the node results for
    bus_map: pl.LazyFrame
        A mapping from busbar sections or bus_breaker_buses to the electrical buses.
        This is used to map the buses from bus_results to electrical buses and back to the monitored buses.
    voltage_levels: pl.LazyFrame
        The voltage levels of the buses. This is used to determine
        voltage limits and nominal v in DC.
    failed_outages : list[str]
        The list of failed outages to get nan-node results for
    timestep : int
        The timestep to get the node results for
    method : Literal["ac", "dc"]
        The method to use for the node results. Either "ac" or "dc"

    Returns
    -------
    patpl.DataFrame[NodeResultSchemaPolars]
        The node results for the given outages and timestep
    """
    if bus_results.limit(1).collect().is_empty():
        return get_failed_node_results_polars(timestep, failed_outages, monitored_buses)
    # Translate bus_ids that could be busbar sections or bus_breaker_buses to the monitored buses
    # Should work for both busbar and bus_breaker models
    node_results = bus_results.drop("operator_strategy_id")
    node_results = node_results.rename({"contingency_id": "contingency"})
    node_results = node_results.join(
        bus_map.select("id", "bus_breaker_bus_id"), left_on=["bus_id"], right_on=["id"], how="left"
    )  # m:1 join
    node_results = node_results.drop_nulls("bus_breaker_bus_id")

    monitored_bus_map = bus_map.filter(pl.col("id").is_in(monitored_buses))
    bus_to_element_map = monitored_bus_map.select(
        pl.col("bus_breaker_bus_id"),
        pl.col("id").alias("element"),
    )
    node_results = node_results.join(bus_to_element_map, on=["bus_breaker_bus_id"], how="left")
    # remove not monitored buses
    node_results = node_results.drop_nulls("element")

    # Merge the actual voltage level in kV
    node_results = node_results.join(
        voltage_levels,
        left_on=["voltage_level_id"],
        right_on=["id"],
        how="left",
    )

    # set timestamp column
    node_results = node_results.with_columns(timestep=pl.lit(timestep))

    node_results = node_results.rename({"v_mag": "vm", "v_angle": "va"})

    # Calculate the values
    if method == "dc":
        node_results = node_results.with_columns(
            pl.when(pl.col("va").is_not_null())
            .then(pl.col("nominal_v"))  # fill vm with nominal v if va is present
            .otherwise(pl.col("vm"))  # keep original vm
            .alias("vm")
        )
    node_results = node_results.with_columns((pl.col("vm") - pl.col("nominal_v")).alias("vm_deviation"))
    node_results = node_results.with_columns(
        (pl.col("vm_deviation") / (pl.col("high_voltage_limit") - pl.col("nominal_v"))).alias("deviation_to_max")
    )
    node_results = node_results.with_columns(
        (pl.col("vm_deviation") / (pl.col("nominal_v") - pl.col("low_voltage_limit"))).alias("deviation_to_min")
    )
    node_results = node_results.with_columns(
        pl.when(pl.col("vm_deviation") >= 0)
        .then(pl.col("deviation_to_max"))
        .otherwise(pl.col("deviation_to_min"))  # keep original vm
        .alias("vm_loading")
    )

    failed_node_results = get_failed_node_results_polars(timestep, failed_outages, monitored_buses)

    # TODO: va_loading is not defined yet
    node_results = node_results.cast({"timestep": pl.Int64})

    # TODO: add p and q calculation at the node
    node_results = node_results.with_columns(
        p=pl.lit(float("nan")),  # TODO
        q=pl.lit(float("nan")),  # TODO
        element_name=pl.lit(""),  # will be filled later
        contingency_name=pl.lit(""),  # will be filled later
    )

    node_results = node_results.select(
        [
            "timestep",
            "contingency",
            "element",
            "vm",
            "va",
            "vm_loading",
            "p",
            "q",
            "element_name",
            "contingency_name",
        ]
    )
    all_node_results = pl.concat([node_results, failed_node_results])

    return all_node_results


@pa.check_types
def get_branch_results_polars(
    branch_results: pl.LazyFrame,
    three_winding_results: pl.LazyFrame,
    monitored_branches: list[str],
    monitored_trafo3w: list[str],
    failed_outages: list[str],
    timestep: int,
    branch_limits: pl.LazyFrame,
) -> patpl.LazyFrame[BranchResultSchemaPolars]:
    """Get the branch results for the given outages and timestep.

    Parameters
    ----------
    branch_results : pl.LazyFrame
        The branch results from the powsybl security analysis
    three_winding_results : pl.LazyFrame
        The three winding transformer results from the powsybl security analysis
    monitored_branches : list[str]
        The list of monitored branches with 2 sides to get the branch results for
    monitored_trafo3w : list[str]
        The list of monitored three winding transformers to get the branch results for
    failed_outages : list[str]
        The list of failed outages to get nan-branch results for
    timestep : int
        The timestep to get the branch results for
    branch_limits : pl.LazyFrame
        The branch limits from the powsybl network

    Returns
    -------
    patpl.DataFrame[BranchResultSchemaPolars]
        The polars branch results for the given outages and timestep
    """
    # Align all indizes
    branch_results = branch_results.drop("operator_strategy_id")
    branch_results = branch_results.rename({"contingency_id": "contingency", "branch_id": "element"})
    three_winding_results = three_winding_results.rename({"contingency_id": "contingency", "transformer_id": "element"})

    side_one_results = (
        pl.concat(
            [
                branch_results.select(["contingency", "element", "p1", "q1", "i1"]),
                three_winding_results.select(["contingency", "element", "p1", "q1", "i1"]),
            ]
        )
        .with_columns(side=pl.lit(BranchSide.ONE.value))
        .rename({"p1": "p", "q1": "q", "i1": "i"})
    )
    side_two_results = (
        pl.concat(
            [
                branch_results.select(["contingency", "element", "p2", "q2", "i2"]),
                three_winding_results.select(["contingency", "element", "p2", "q2", "i2"]),
            ]
        )
        .with_columns(side=pl.lit(BranchSide.TWO.value))
        .rename({"p2": "p", "q2": "q", "i2": "i"})
    )
    side_three_results = (
        three_winding_results.select(["contingency", "element", "p3", "q3", "i3"])
        .with_columns(side=pl.lit(BranchSide.THREE.value))
        .rename({"p3": "p", "q3": "q", "i3": "i"})
    )
    # Combine and Add timestep column
    converted_branch_results = pl.concat([side_one_results, side_two_results, side_three_results]).with_columns(
        timestep=pl.lit(timestep)
    )
    converted_branch_results = converted_branch_results.cast({"timestep": pl.Int64, "side": pl.Int64})
    branch_limits = branch_limits.cast({"side": pl.Int64, "value": pl.Float64})

    if not converted_branch_results.limit(1).collect().is_empty():
        converted_branch_results = (
            converted_branch_results.join(
                branch_limits, left_on=["element", "side"], right_on=["element_id", "side"], how="left"
            )  # m:1 join
            .with_columns(loading=pl.col("i") / pl.col("value"))
            .drop("value")
        )
    else:
        # add i column
        converted_branch_results = converted_branch_results.with_columns(i=pl.lit(float("nan")))
        # add loading column
        converted_branch_results = converted_branch_results.with_columns(loading=pl.lit(float("nan")))
        # cast null to str
        converted_branch_results = converted_branch_results.cast({"contingency": pl.String, "element": pl.String})
    # fill loading nulls with nans for loading
    converted_branch_results = converted_branch_results.with_columns(pl.col("loading").fill_null(float("nan")))
    # add empty element_name and contingency_name columns to match the schema
    converted_branch_results = converted_branch_results.with_columns(
        element_name=pl.lit(""),
        contingency_name=pl.lit(""),
    )

    # Add results for non convergent contingencies
    failed_branch_results = get_failed_branch_results_polars(timestep, failed_outages, monitored_branches, monitored_trafo3w)

    converted_branch_results = converted_branch_results.select(
        [
            "timestep",
            "contingency",
            "element",
            "side",
            "p",
            "q",
            "i",
            "loading",
            "element_name",
            "contingency_name",
        ]
    )
    converted_branch_results = pl.concat([converted_branch_results, failed_branch_results])

    return converted_branch_results


@pa.check_types
def get_va_diff_results_polars(
    bus_results: pl.LazyFrame, outages: list[str], va_diff_with_buses: pl.LazyFrame, bus_map: pl.LazyFrame, timestep: int
) -> patpl.LazyFrame[VADiffResultSchemaPolars]:
    """Get the voltage angle difference results for the given outages and bus results.

    Parameters
    ----------
    bus_results : pl.LazyFrame
        The dataframe containing the bus results of powsybl contingency analysis.
    outages : list[str]
        The list of outages to be considered. These are the contingency ids that are outaged.
    va_diff_with_buses : pl.LazyFrame
        The dataframe containing the voltage angle difference results with the bus pairs that need checking.
    bus_map: pl.LazyFrame
        A mapping from busbar sections to bus breaker buses. This is used to convert the busbar sections to bus breaker buses
        in the Node Breaker model.
    timestep : int
        The timestep of the results.

    Returns
    -------
    VADiffResultSchemaPolars
        The dataframe containing the voltage angle difference results for the given outages.
    """
    if len(outages) == 0 or bus_results.limit(1).collect().is_empty():
        return (
            pl.from_pandas(get_empty_dataframe_from_model(VADiffResultSchema), include_index=True, nan_to_null=False)
            .lazy()
            .cast({"timestep": pl.Int64, "va_diff": pl.Float64})
        )
    basecase_in_result = ""
    iteration_va_diff = va_diff_with_buses.filter(pl.col("contingency").is_in([basecase_in_result, *outages]))

    iteration_va_diff = iteration_va_diff.with_columns(timestep=pl.lit(timestep).cast(pl.Int64))
    # Map busbar sections where there are any. For the rest use the bus_breaker_bus_id from the results (here the bus id)
    bus_results = bus_results.join(
        bus_map.select("id", "bus_breaker_bus_id"), left_on=["bus_id"], right_on=["id"], how="left"
    )  # m:1 join

    # get the voltage angles for both buses in the va_diff definition
    iteration_va_diff = iteration_va_diff.join(
        bus_results.select("contingency_id", "bus_breaker_bus_id", "v_angle"),
        left_on=["contingency", "bus_breaker_bus1_id"],
        right_on=["contingency_id", "bus_breaker_bus_id"],
        how="left",
    )  # m:1 join
    iteration_va_diff = iteration_va_diff.rename({"v_angle": "v_angle_1"})
    iteration_va_diff = iteration_va_diff.join(
        bus_results.select("contingency_id", "bus_breaker_bus_id", "v_angle"),
        left_on=["contingency", "bus_breaker_bus2_id"],
        right_on=["contingency_id", "bus_breaker_bus_id"],
        how="left",
    )  # m:1 join
    iteration_va_diff = iteration_va_diff.rename({"v_angle": "v_angle_2"})

    # Calculate the voltage angle difference
    iteration_va_diff = iteration_va_diff.with_columns((pl.col("v_angle_1") - pl.col("v_angle_2")).alias("va_diff"))

    # drop duplicates
    iteration_va_diff = iteration_va_diff.unique()

    # add empty element_name and contingency_name columns to match the schema
    iteration_va_diff = iteration_va_diff.with_columns(
        element_name=pl.lit(""),  # will be filled later
        contingency_name=pl.lit(""),  # will be filled later
    )

    iteration_va_diff = iteration_va_diff.select(
        [
            "timestep",
            "contingency",
            "element",
            "va_diff",
            "element_name",
            "contingency_name",
        ]
    )

    return iteration_va_diff


@pa.check_types
def update_basename_polars(
    result_df: LoadflowResultTablePolars,
    basecase_name: Optional[str] = None,
) -> LoadflowResultTablePolars:
    """Update the basecase name in the results dataframes.

    This function updates the contingency index level of the results dataframes to
    reflect the basecase name. If the basecase is not included in the run, it will
    remove it from the results. Powsybl includes the basecase as an empty string by default.

    The Dataframes are expected to have a multi-index with a "contingency" level.
    The Dataframes are updated inplace.

    Parameters
    ----------
    result_df: LoadflowResultTablePolars
        The dataframe containing the branch / node / VADiff results
    basecase_name: Optional[str], optional
        The name of the basecase contingency, if it is included in the run. Otherwise None, by default None

    Returns
    -------
    LoadflowResultTablePolars
        The updated dataframes with the basecase name set or removed.
    """
    if basecase_name is not None:
        # Replace the empty string with the basecase name
        result_df = result_df.with_columns(
            pl.when(pl.col("contingency") == "")
            .then(pl.lit(basecase_name))
            .otherwise(pl.col("contingency"))
            .alias("contingency")
        )

    else:
        # Remove the basecase from the results if it is not included in the run
        result_df = result_df.filter(pl.col("contingency") != "")
    return result_df


@pa.check_types
def add_name_column_polars(
    result_df: LoadflowResultTablePolars,
    name_map: dict[str, str],
    index_level: str = "element",
) -> LoadflowResultTablePolars:
    """Translate the element ids in the results dataframes to the original names.

    This function translates the element names in the results dataframes to the original names
    from the Powsybl network. This is useful for debugging and for displaying the results.

    Parameters
    ----------
    result_df: LoadflowResultTablePolars
        The dataframe containing the node / branch / VADiff results
    name_map: dict[str | str]
        A mapping from the element ids to the original names. This is used to translate the element names in the results.
    index_level: str, optional
        The index level storing the ids that should be mapped to the names. by default "element" for the monitored elements.

    Returns
    -------
    LoadflowResultTablePolars
        The updated dataframe with the ids translated to the original names.
    """
    result_df = result_df.with_columns(
        pl.when(pl.col(index_level).is_in(list(name_map.keys())))
        .then(pl.col(index_level).replace(name_map))
        .otherwise(pl.col(f"{index_level}_name"))
        .alias(f"{index_level}_name")
    )
    # fill nulls with empty string
    result_df = result_df.with_columns(pl.col(f"{index_level}_name").fill_null(""))
    return result_df


@pa.check_types
def get_failed_node_results_polars(
    timestep: int, failed_outages: list[str], monitored_nodes: list[str]
) -> patpl.LazyFrame[NodeResultSchemaPolars]:
    """Get the failed node results for the given outages and timestep.

    A wrapper around get_failed_node_results to convert the pandas dataframe to a polars dataframe.

    Parameters
    ----------
    timestep : int
        The timestep to get the node results for
    failed_outages : list[str]
        The list of failed outages to get nan-node results for
    monitored_nodes : list[str]
        The list of monitored nodes to get the node results for

    Returns
    -------
    patpl.DataFrame[NodeResultSchemaPolars]
        The polars dataframe containing the failed node results for the given outages and timestep
    """
    failed_node_results = get_failed_node_results(timestep, failed_outages, monitored_nodes)
    failed_node_results = pl.from_pandas(failed_node_results, include_index=True, nan_to_null=False).lazy()
    failed_node_results = failed_node_results.cast({"timestep": pl.Int64})
    return failed_node_results


@pa.check_types
def get_failed_branch_results_polars(
    timestep: int, failed_outages: list[str], monitored_branches: list[str], monitored_trafo3w: list[str]
) -> patpl.LazyFrame[BranchResultSchemaPolars]:
    """Get the failed branch results for the given outages and timestep.

    A wrapper around get_failed_branch_results to convert the pandas dataframe to a polars dataframe.

    Parameters
    ----------
    timestep : int
        The timestep to get the branch results for
    failed_outages : list[str]
        The list of failed outages to get nan-branch results for
    monitored_branches : list[str]
        The list of monitored branches with 2 sides to get the branch results for
    monitored_trafo3w : list[str]
        The list of monitored three winding transformers to get the branch results for

    Returns
    -------
    patpl.DataFrame[BranchResultSchemaPolars]
        The polars dataframe containing the failed branch results for the given outages and timestep
    """
    failed_branch_results = get_failed_branch_results(timestep, failed_outages, monitored_branches, monitored_trafo3w)
    failed_branch_results = pl.from_pandas(failed_branch_results, include_index=True, nan_to_null=False).lazy()
    failed_branch_results = failed_branch_results.cast({"timestep": pl.Int64, "side": pl.Int64})
    return failed_branch_results
