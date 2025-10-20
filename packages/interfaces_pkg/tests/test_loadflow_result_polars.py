import polars as pl
import pypowsybl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from polars.testing import assert_frame_equal
from test_loadflow_results_new import get_loadflow_results_example
from toop_engine_contingency_analysis.pypowsybl import (
    run_contingency_analysis_powsybl,
)
from toop_engine_grid_helpers.powsybl.example_grids import powsybl_extended_case57
from toop_engine_interfaces.loadflow_result_helpers import convert_pandas_loadflow_results_to_polars
from toop_engine_interfaces.loadflow_result_helpers_polars import (
    concatenate_loadflow_results_polars,
    load_loadflow_results_polars,
    save_loadflow_results_polars,
)
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition


def test_save_and_load_loadflow_results_polars(tmp_path):
    loadflow_results_polars = convert_pandas_loadflow_results_to_polars(
        get_loadflow_results_example(job_id="test", timestep=0, size=5)
    )
    fs = DirFileSystem(tmp_path)
    ref = save_loadflow_results_polars(fs, "test_loadflow_results", loadflow_results_polars)
    loadflow_results_loaded = load_loadflow_results_polars(fs, ref)
    assert loadflow_results_loaded == loadflow_results_polars


def test_save_and_load_loadflow_results_no_validate_polars(tmp_path):
    loadflow_results_polars = convert_pandas_loadflow_results_to_polars(
        get_loadflow_results_example(job_id="test", timestep=0, size=5)
    )

    fs = DirFileSystem(tmp_path)
    ref = save_loadflow_results_polars(fs, "test_loadflow_results", loadflow_results_polars)
    loadflow_results_loaded = load_loadflow_results_polars(fs, ref, validate=False)
    assert loadflow_results_polars == loadflow_results_loaded, "Loadflow results should be equal even when validate is False"


def test_extract_branch_results():
    net = powsybl_extended_case57()

    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in list(net.get_branches().iterrows())[:10]
        ]
        + [
            GridElement(id=index, name=row.name, kind="bus", type="bus")
            for index, row in list(net.get_voltage_levels().iterrows())[:10]
        ],
        contingencies=[
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in list(net.get_branches().iterrows())  # [3:7]
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="ac", polars=True
    )


def test_concatenate_loadflow_results_polars():
    net = pypowsybl.network.create_ieee14()
    nminus1_def_1 = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )

    nminus1_def_2 = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res_1 = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_1, job_id="same_test_job", timestep=0, method="dc", polars=True
    )
    res_2 = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_2, job_id="same_test_job", timestep=1, method="dc", polars=True
    )

    res = concatenate_loadflow_results_polars([res_1, res_2])

    assert len(res.additional_information) == len(res_1.additional_information) + len(res_2.additional_information)
    assert len(res.warnings) == len(res_1.warnings) + len(res_2.warnings)

    res_branch_results = res.branch_results.collect()
    res_node_results = res.node_results.collect()
    res_regulating_element_results = res.regulating_element_results.collect()
    res_va_diff_results = res.va_diff_results.collect()
    res_converged = res.converged.collect()

    res_1_branch_results = res_1.branch_results.collect()
    res_1_node_results = res_1.node_results.collect()
    res_1_regulating_element_results = res_1.regulating_element_results.collect()
    res_1_va_diff_results = res_1.va_diff_results.collect()
    res_1_converged = res_1.converged.collect()

    res_2_branch_results = res_2.branch_results.collect()
    res_2_node_results = res_2.node_results.collect()
    res_2_regulating_element_results = res_2.regulating_element_results.collect()
    res_2_va_diff_results = res_2.va_diff_results.collect()
    res_2_converged = res_2.converged.collect()

    assert len(res_branch_results) == len(res_1_branch_results) + len(res_2_branch_results)
    assert len(res_node_results) == len(res_1_node_results) + len(res_2_node_results)
    assert len(res_regulating_element_results) == len(res_1_regulating_element_results) + len(
        res_2_regulating_element_results
    )
    assert len(res_va_diff_results) == len(res_1_va_diff_results) + len(res_2_va_diff_results)
    assert len(res_converged) == len(res_1_converged) + len(res_2_converged)

    assert_frame_equal_kwargs = {
        "check_row_order": False,
        "check_column_order": False,
        "check_dtypes": True,
        "check_exact": False,
        "abs_tol": 10e-6,
    }
    assert_frame_equal(
        res_node_results.filter(pl.col("timestep").is_in([0])), res_1_node_results, **assert_frame_equal_kwargs
    )
    assert_frame_equal(
        res_node_results.filter(pl.col("timestep").is_in([1])), res_2_node_results, **assert_frame_equal_kwargs
    )
    assert_frame_equal(
        res_branch_results.filter(pl.col("timestep").is_in([0])), res_1_branch_results, **assert_frame_equal_kwargs
    )
    assert_frame_equal(
        res_branch_results.filter(pl.col("timestep").is_in([1])), res_2_branch_results, **assert_frame_equal_kwargs
    )
    assert_frame_equal(
        res_regulating_element_results.filter(pl.col("timestep").is_in([0])),
        res_1_regulating_element_results,
        **assert_frame_equal_kwargs,
    )
    assert_frame_equal(
        res_regulating_element_results.filter(pl.col("timestep").is_in([1])),
        res_2_regulating_element_results,
        **assert_frame_equal_kwargs,
    )
    assert_frame_equal(
        res_va_diff_results.filter(pl.col("timestep").is_in([0])), res_1_va_diff_results, **assert_frame_equal_kwargs
    )
    assert_frame_equal(
        res_va_diff_results.filter(pl.col("timestep").is_in([1])), res_2_va_diff_results, **assert_frame_equal_kwargs
    )
    assert_frame_equal(res_converged.filter(pl.col("timestep").is_in([0])), res_1_converged, **assert_frame_equal_kwargs)
    assert_frame_equal(res_converged.filter(pl.col("timestep").is_in([1])), res_2_converged, **assert_frame_equal_kwargs)

    other_job_res = res_2.model_copy(update={"job_id": "other_test_job"})
    with pytest.raises(AssertionError):
        res = concatenate_loadflow_results_polars([res_1, other_job_res])

    with pytest.raises(AssertionError):
        concatenate_loadflow_results_polars([])
