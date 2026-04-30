# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandas as pd
from toop_engine_contingency_analysis.pandapower.spps.engine import _satisfied_scheme_names


def test_satisfied_scheme_names_legacy_defaults_to_all() -> None:
    df = pd.DataFrame(
        {
            "scheme_name": ["s1", "s1"],
            "is_condition": [True, True],
        }
    )
    assert _satisfied_scheme_names(df) == {"s1"}

    df2 = pd.DataFrame(
        {
            "scheme_name": ["s1", "s1"],
            "is_condition": [True, False],
        }
    )
    assert _satisfied_scheme_names(df2) == set()


def test_satisfied_scheme_names_all_requires_every_row() -> None:
    df = pd.DataFrame(
        {
            "scheme_name": ["s1", "s1"],
            "condition_logic": ["all", "all"],
            "is_condition": [True, False],
        }
    )
    assert _satisfied_scheme_names(df) == set()

    df_ok = pd.DataFrame(
        {
            "scheme_name": ["s1", "s1"],
            "condition_logic": ["all", "all"],
            "is_condition": [True, True],
        }
    )
    assert _satisfied_scheme_names(df_ok) == {"s1"}


def test_satisfied_scheme_names_any_one_true_suffices() -> None:
    df = pd.DataFrame(
        {
            "scheme_name": ["s1", "s1"],
            "condition_logic": ["any", "any"],
            "is_condition": [False, True],
        }
    )
    assert _satisfied_scheme_names(df) == {"s1"}

    df_none = pd.DataFrame(
        {
            "scheme_name": ["s1", "s1"],
            "condition_logic": ["any", "any"],
            "is_condition": [False, False],
        }
    )
    assert _satisfied_scheme_names(df_none) == set()
