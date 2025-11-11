from pathlib import Path

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from toop_engine_interfaces.messages.preprocess_results_factory import (
    create_error_result,
    create_power_factory_import_result,
    create_preprocessing_started_result,
    create_preprocessing_success_result,
    create_result,
    create_static_information_stats,
    create_ucte_import_result,
)
from toop_engine_interfaces.messages.protobuf_schema.lf_service.stored_loadflow_reference_pb2 import StoredLoadflowReference
from toop_engine_interfaces.messages.protobuf_schema.preprocess.preprocess_results_pb2 import (
    ErrorResult,
    PowerFactoryImportResult,
    PreprocessingStartedResult,
    PreprocessingSuccessResult,
    Result,
    StaticInformationStats,
    UcteImportResult,
)


def test_ucte_import_result_defaults(tmp_path):
    tmp_path = Path(tmp_path)
    result = create_ucte_import_result(data_folder=tmp_path)
    assert isinstance(result, UcteImportResult)
    assert result.data_folder == str(tmp_path)


def test_static_information_stats_defaults():
    result = create_static_information_stats()
    assert isinstance(result, StaticInformationStats)


def test_power_factory_import_result_defaults():
    result = create_power_factory_import_result()
    assert isinstance(result, PowerFactoryImportResult)
    assert result.grid_type == "power_factory"


def test_preprocessing_success_result_defaults(tmp_path):
    result = create_preprocessing_success_result(
        data_folder=tmp_path,
        static_information_stats=create_static_information_stats(),
        initial_loadflow=StoredLoadflowReference(relative_path="does/not/exist"),
        initial_metrics={"max_flow_n_1": 1.5, "overload_energy_n_1": 2.0},
        importer_results=create_ucte_import_result(data_folder=tmp_path),
    )
    assert isinstance(result, PreprocessingSuccessResult)
    assert result.data_folder == str(tmp_path)
    assert isinstance(result.static_information_stats, StaticInformationStats)
    assert isinstance(result.ucte_result, UcteImportResult)


def test_preprocessing_started_result_defaults():
    result = create_preprocessing_started_result()
    assert isinstance(result, PreprocessingStartedResult)
    assert result.result_type == "preprocessing_started"


def test_error_result_defaults():
    result = create_error_result(error="An error occurred")
    assert isinstance(result, ErrorResult)
    assert result.error == "An error occurred"
    assert result.result_type == "error"
    with pytest.raises(BeartypeCallHintParamViolation):
        create_error_result(error=123)


def test_result():
    error_result = create_error_result(error="Test error")
    result = Result(preprocess_id="id123", runtime=1.23, error_result=error_result)
    assert isinstance(result, Result)
    assert isinstance(result.error_result, ErrorResult)
    assert result.error_result.error == "Test error"


def test_result_with_preprocessing_started_result():
    started_result = create_preprocessing_started_result()
    result = Result(preprocess_id="id456", runtime=0.0, started_result=started_result)
    assert isinstance(result, Result)
    assert isinstance(result.started_result, PreprocessingStartedResult)
    assert result.started_result.result_type == "preprocessing_started"

    with pytest.raises(ValueError):
        res = create_result(preprocess_id="id", runtime=-1.0, result=started_result)
