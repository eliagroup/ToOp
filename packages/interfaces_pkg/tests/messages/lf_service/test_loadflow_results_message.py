# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from toop_engine_interfaces.messages.lf_service import loadflow_results as lf_res


def test_stored_loadflow_reference():
    ref = lf_res.StoredLoadflowReference(relative_path="result1.h5")
    assert ref.relative_path == "result1.h5"


def test_loadflow_stream_result():
    ref = lf_res.StoredLoadflowReference(relative_path="result2.h5")
    stream = lf_res.LoadflowStreamResult(loadflow_reference=ref, solved_timesteps=[0, 1], remainging_timesteps=[2, 3])
    assert stream.loadflow_reference == ref
    assert stream.solved_timesteps == [0, 1]
    assert stream.remainging_timesteps == [2, 3]
    assert stream.result_type == "loadflow_stream"


def test_loadflow_success_result():
    ref = lf_res.StoredLoadflowReference(relative_path="result3.h5")
    success = lf_res.LoadflowSuccessResult(loadflow_reference=ref)
    assert success.loadflow_reference == ref
    assert success.result_type == "loadflow_success"


def test_loadflow_started_result():
    started = lf_res.LoadflowStartedResult()
    assert started.result_type == "loadflow_started"


def test_error_result():
    err = lf_res.ErrorResult(error="Something went wrong")
    assert err.error == "Something went wrong"
    assert err.result_type == "error"


def test_loadflow_base_result_success():
    ref = lf_res.StoredLoadflowReference(relative_path="result4.h5")
    success = lf_res.LoadflowSuccessResult(loadflow_reference=ref)
    base = lf_res.LoadflowBaseResult(loadflow_id="lf1", job_id="job1", runtime=1.23, result=success)
    assert base.loadflow_id == "lf1"
    assert base.job_id == "job1"
    assert base.runtime == 1.23
    assert base.result == success
    assert isinstance(base.uuid, str)
    assert isinstance(base.timestamp, str)


def test_loadflow_base_result_error():
    err = lf_res.ErrorResult(error="fail")
    base = lf_res.LoadflowBaseResult(loadflow_id="lf2", job_id="job2", runtime=0.0, result=err)
    assert base.result == err
    assert base.loadflow_id == "lf2"
    assert base.job_id == "job2"
