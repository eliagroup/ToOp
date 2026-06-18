# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandas as pd
from toop_engine_contingency_analysis.pandapower.cascade.detection.switch_preparation import get_complex_impedance


def make_df(vm, i, p, q):
    return pd.DataFrame({"vm": [vm], "i": [i], "p": [p], "q": [q]})


def test_get_complex_impedance_pure_resistive():
    # q=0 → phi=0 → r=z, x=0
    # Pick vm=1 kV, i=1000/sqrt(3) A so that z_ohm=1 exactly
    i_val = 1000.0 / np.sqrt(3)
    df = make_df(vm=1.0, i=i_val, p=1.0, q=0.0)

    r, x = get_complex_impedance(df)

    assert np.isclose(r.iloc[0], 1.0)
    assert np.isclose(x.iloc[0], 0.0, atol=1e-12)


def test_get_complex_impedance_pure_reactive():
    # p=0 → phi=pi/2 → r=0, x=z
    i_val = 1000.0 / np.sqrt(3)
    df = make_df(vm=1.0, i=i_val, p=0.0, q=1.0)

    r, x = get_complex_impedance(df)

    assert np.isclose(r.iloc[0], 0.0, atol=1e-12)
    assert np.isclose(x.iloc[0], 1.0)


def test_get_complex_impedance_mixed():
    # p=3, q=4 → s=5, phi=arctan(4/3)
    # cos(phi)=3/5=0.6, sin(phi)=4/5=0.8
    # Use vm=1 kV, i=1000/sqrt(3) A → z_ohm=1
    # Expected: r=0.6, x=0.8
    i_val = 1000.0 / np.sqrt(3)
    df = make_df(vm=1.0, i=i_val, p=3.0, q=4.0)

    r, x = get_complex_impedance(df)

    assert np.isclose(r.iloc[0], 0.6)
    assert np.isclose(x.iloc[0], 0.8)


def test_get_complex_impedance_zero_current_returns_nan():
    df = make_df(vm=110.0, i=0.0, p=5.0, q=3.0)

    r, x = get_complex_impedance(df)

    assert pd.isna(r.iloc[0])
    assert pd.isna(x.iloc[0])
