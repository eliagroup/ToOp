# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from toop_engine_interfaces.messages.lf_service import loadflow_heartbeat as lf_hb


def test_loadflow_status_info():
    info = lf_hb.LoadflowStatusInfo(loadflow_id="lf1", runtime=12.5, message="Running")
    assert info.loadflow_id == "lf1"
    assert info.runtime == 12.5
    assert info.message == "Running"

    info2 = lf_hb.LoadflowStatusInfo(loadflow_id="lf2", runtime=0.0)
    assert info2.message == ""


def test_loadflow_heartbeat():
    info = lf_hb.LoadflowStatusInfo(loadflow_id="lf1", runtime=1.0)
    hb = lf_hb.LoadflowHeartbeat(idle=False, status_info=info)
    assert hb.idle is False
    assert hb.status_info == info
    assert isinstance(hb.timestamp, str)
    assert isinstance(hb.uuid, str)

    hb_idle = lf_hb.LoadflowHeartbeat(idle=True, status_info=None)
    assert hb_idle.idle is True
    assert hb_idle.status_info is None
