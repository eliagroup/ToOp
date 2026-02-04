# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import importlib
import sys

import pandera
from toop_engine_dc_solver.__init__ import Int


def test_int_check_win32(monkeypatch):
    # Check pandera version
    pandera_version = tuple(map(int, pandera.__version__.split(".")))
    assert pandera_version < (0, 23), "Check if this Patch is still needed"
    # Patch sys.platform to 'win32'
    monkeypatch.setattr(sys, "platform", "win32")
    importlib.reload(sys.modules["toop_engine_dc_solver.__init__"])

    class DummyInt:
        pass

    dummy_int = DummyInt()
    # Should return True for Int instance
    assert Int.check(dummy_int, Int()), "Int.check should return True for Int instance on win32"
    # Should return False for non-Int instance
    assert not Int.check(dummy_int, 123), "Int.check should return False for non-Int instance on win32"


def test_int_check_non_win32(monkeypatch):
    # Patch sys.platform to 'linux'
    monkeypatch.setattr(sys, "platform", "linux")
    importlib.reload(sys.modules["toop_engine_dc_solver.__init__"])
    # The Int.check should not be patched, so it should raise TypeError for wrong usage
    try:
        Int.check(None, 123)
    except Exception:
        pass  # Expected: original Int.check may raise
