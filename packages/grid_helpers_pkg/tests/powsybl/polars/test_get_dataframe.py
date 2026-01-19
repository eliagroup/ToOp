# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import polars as pl
import pypowsybl
from polars.testing import assert_frame_equal
from toop_engine_grid_helpers.powsybl.polars.get_dataframe import (
    get_ca_branch_results,
    get_ca_bus_results,
    get_ca_three_windings_transformer_results,
)


def test_get_ca_results_polars():
    # TODO: choose a grid with 3w trafos
    network = pypowsybl.network.create_ieee14()
    security_analysis = pypowsybl.security.create_analysis()

    security_analysis.add_single_element_contingencies(list(network.get_branches().index))
    security_analysis.add_monitored_elements(
        branch_ids=list(network.get_branches().index),
        voltage_level_ids=list(network.get_voltage_levels().index),
    )

    result = security_analysis.run_ac(network)
    branches_df = pl.from_pandas(result.branch_results, include_index=True, nan_to_null=False)
    buses_df = pl.from_pandas(result.bus_results, include_index=True, nan_to_null=False)
    trafo3w = pl.from_pandas(result.three_windings_transformer_results, include_index=True, nan_to_null=False)

    branches_df_polars = get_ca_branch_results(result, lazy=False)
    buses_df_polars = get_ca_bus_results(result, lazy=False)
    trafo3w_df_polars = get_ca_three_windings_transformer_results(result, lazy=False)

    kw_args_testing = {
        "check_row_order": False,
        "check_column_order": False,
        "check_dtypes": False,
        "check_exact": False,
        "abs_tol": 1e-6,
    }

    assert_frame_equal(branches_df_polars, branches_df, **kw_args_testing)
    assert_frame_equal(buses_df_polars, buses_df, **kw_args_testing)
    assert_frame_equal(trafo3w_df_polars, trafo3w, **kw_args_testing)

    # test lazy
    branches_df_polars = get_ca_branch_results(result, lazy=True)
    buses_df_polars = get_ca_bus_results(result, lazy=True)
    trafo3w_df_polars = get_ca_three_windings_transformer_results(result, lazy=True)
    assert isinstance(branches_df_polars, pl.LazyFrame)
    assert isinstance(buses_df_polars, pl.LazyFrame)
    assert isinstance(trafo3w_df_polars, pl.LazyFrame)
    branches_df_polars = branches_df_polars.collect()
    buses_df_polars = buses_df_polars.collect()
    trafo3w_df_polars = trafo3w_df_polars.collect()

    kw_args_testing = {
        "check_row_order": False,
        "check_column_order": False,
        "check_dtypes": False,
        "check_exact": False,
        "abs_tol": 1e-6,
    }

    assert_frame_equal(branches_df_polars, branches_df, **kw_args_testing)
    assert_frame_equal(buses_df_polars, buses_df, **kw_args_testing)
    assert_frame_equal(trafo3w_df_polars, trafo3w, **kw_args_testing)
