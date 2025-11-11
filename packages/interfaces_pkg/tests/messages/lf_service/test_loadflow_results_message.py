from toop_engine_interfaces.messages.loadflow_results_factory import (
    create_error_result,
    create_loadflow_base_result,
    create_loadflow_started_result,
    create_loadflow_stream_result,
    create_loadflow_success_result,
)
from toop_engine_interfaces.messages.protobuf_schema.lf_service.stored_loadflow_reference_pb2 import StoredLoadflowReference


def test_stored_loadflow_reference():
    ref = StoredLoadflowReference(relative_path="result1.h5")
    assert ref.relative_path == "result1.h5"


def test_loadflow_stream_result():
    ref = StoredLoadflowReference(relative_path="result2.h5")
    stream = create_loadflow_stream_result(loadflow_reference=ref, solved_timesteps=[0, 1], remaining_timesteps=[2, 3])
    assert stream.loadflow_reference == ref
    assert stream.solved_timesteps == [0, 1]
    assert stream.remaining_timesteps == [2, 3]
    assert stream.result_type == "loadflow_stream"


def test_loadflow_success_result():
    ref = StoredLoadflowReference(relative_path="result3.h5")
    success = create_loadflow_success_result(loadflow_reference=ref)
    assert success.loadflow_reference == ref
    assert success.result_type == "loadflow_success"


def test_loadflow_started_result():
    started = create_loadflow_started_result()
    assert started.result_type == "loadflow_started"


def test_error_result():
    err = create_error_result(error="Something went wrong")
    assert err.error == "Something went wrong"
    assert err.result_type == "error"


def test_loadflow_base_result_success():
    ref = StoredLoadflowReference(relative_path="result4.h5")
    success = create_loadflow_success_result(loadflow_reference=ref)
    base = create_loadflow_base_result(loadflow_id="lf1", job_id="job1", runtime=1.23, result=success)
    assert base.loadflow_id == "lf1"
    assert base.job_id == "job1"
    assert base.runtime == 1.23
    assert base.success_result == success
    assert isinstance(base.uuid, str)
    assert isinstance(base.timestamp, str)


def test_loadflow_base_result_error():
    err = create_error_result(error="fail")
    base = create_loadflow_base_result(loadflow_id="lf2", job_id="job2", runtime=0.0, result=err)
    assert base.error_result == err
    assert base.loadflow_id == "lf2"
    assert base.job_id == "job2"
