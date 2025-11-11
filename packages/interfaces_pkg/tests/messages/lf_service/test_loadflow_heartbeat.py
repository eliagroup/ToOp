from toop_engine_interfaces.messages.loadflow_heartbeat_factory import (
    create_loadflow_heartbeat,
    create_loadflow_status_info,
)


def test_loadflow_status_info():
    info = create_loadflow_status_info(loadflow_id="lf1", runtime=12.5, message="Running")
    assert info.loadflow_id == "lf1"
    assert info.runtime == 12.5
    assert info.message == "Running"

    info2 = create_loadflow_status_info(loadflow_id="lf2", runtime=0.0)
    assert info2.message == ""


def test_loadflow_heartbeat():
    info = create_loadflow_status_info(loadflow_id="lf1", runtime=1.0)
    hb = create_loadflow_heartbeat(idle=False, status_info=info)
    assert hb.idle is False
    assert hb.status_info == info
    assert isinstance(hb.timestamp, str)
    assert isinstance(hb.uuid, str)

    hb_idle = create_loadflow_heartbeat(idle=True, status_info=None)
    assert hb_idle.idle is True
    assert hb_idle.status_info is None
