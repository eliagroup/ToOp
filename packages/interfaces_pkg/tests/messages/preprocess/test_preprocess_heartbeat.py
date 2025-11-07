import logbook
import pytest
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat_factory import (
    PreprocessStage,
    empty_status_update_fn,
    get_preprocess_heartbeat,
    get_preprocess_status_info,
)

# from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import (
#     PreprocessHeartbeat,
#     PreprocessStage,
#     PreprocessStatusInfo,
#     empty_status_update_fn,
# )


def test_preprocess_status_info_creation():
    status = get_preprocess_status_info(
        preprocess_id="job123", runtime=12.5, stage="preprocess_started", message="Started preprocessing"
    )
    assert status.preprocess_id == "job123"
    assert status.runtime == 12.5
    assert status.stage == "preprocess_started"
    assert status.message == "Started preprocessing"


def test_preprocess_heartbeat_idle():
    heartbeat = get_preprocess_heartbeat(idle=True)
    assert heartbeat.idle is True
    # assert heartbeat.status_info is None
    assert isinstance(heartbeat.instance_id, str)
    assert isinstance(heartbeat.timestamp, str)
    assert isinstance(heartbeat.uuid, str)


def test_preprocess_heartbeat_with_status_info():
    status = get_preprocess_status_info(preprocess_id="job456", runtime=5.0, stage="convert_to_jax_started", message="")
    heartbeat = get_preprocess_heartbeat(idle=False, status_info=status)
    assert heartbeat.idle is False
    assert heartbeat.status_info == status


def test_empty_status_update_fn_logs():
    with logbook.handlers.TestHandler() as log_handler:
        stage = "preprocess_started"
        message = ""
        empty_status_update_fn(stage, message)
        assert log_handler.has_infos

        message = "Custom message"
        empty_status_update_fn(stage, message)
        assert log_handler.has_infos


@pytest.mark.parametrize("stage", PreprocessStage.__args__[0:5])
def test_preprocess_stage_literals(stage):
    # Should not raise error for valid stages
    status = get_preprocess_status_info(preprocess_id="job789", runtime=1.0, stage=stage, message="")
    assert status.stage == stage
