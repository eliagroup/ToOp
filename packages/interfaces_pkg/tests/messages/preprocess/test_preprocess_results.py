# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import pytest
from pydantic import ValidationError
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
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
    result = UcteImportResult(data_folder=tmp_path)
    assert isinstance(result, UcteImportResult)
    assert result.data_folder == tmp_path


def test_static_information_stats_defaults():
    result = StaticInformationStats()
    assert isinstance(result, StaticInformationStats)


def test_power_factory_import_result_defaults():
    result = PowerFactoryImportResult()
    assert isinstance(result, PowerFactoryImportResult)
    assert result.grid_type == "power_factory"


def test_preprocessing_success_result_defaults(tmp_path):
    result = PreprocessingSuccessResult(
        data_folder=tmp_path,
        static_information_stats=StaticInformationStats(),
        initial_loadflow=StoredLoadflowReference(relative_path="does/not/exist"),
        initial_metrics={"max_flow_n_1": 1.5, "overload_energy_n_1": 2.0},
        importer_results=UcteImportResult(data_folder=tmp_path),
    )
    assert isinstance(result, PreprocessingSuccessResult)
    assert result.data_folder == tmp_path
    assert isinstance(result.static_information_stats, StaticInformationStats)
    assert isinstance(result.importer_results, UcteImportResult)


def test_preprocessing_started_result_defaults():
    result = PreprocessingStartedResult()
    assert isinstance(result, PreprocessingStartedResult)
    assert result.result_type == "preprocessing_started"


def test_error_result_defaults():
    result = ErrorResult(error="An error occurred")
    assert isinstance(result, ErrorResult)
    assert result.error == "An error occurred"
    assert result.result_type == "error"
    with pytest.raises(ValidationError):
        ErrorResult(error=123)


def test_result():
    error_result = ErrorResult(error="Test error")
    result = Result(preprocess_id="id123", runtime=1.23, result=error_result)
    assert isinstance(result, Result)
    assert isinstance(result.result, ErrorResult)
    assert result.result.error == "Test error"


def test_result_with_preprocessing_started_result():
    started_result = PreprocessingStartedResult()
    result = Result(preprocess_id="id456", runtime=0.0, result=started_result)
    assert isinstance(result, Result)
    assert isinstance(result.result, PreprocessingStartedResult)
    assert result.result.result_type == "preprocessing_started"

    with pytest.raises(ValidationError):
        Result(preprocess_id="id", runtime=-1, result=PreprocessingStartedResult())
