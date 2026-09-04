# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""OverloadConfig.threshold: which loading threshold applies to one branch.

Unlike the distance-protection factors, these keep a fallback chain: the more specific value
wins, then the base-case one for the same element type, then the scalar that applies to
everything.
"""

import pytest
from toop_engine_contingency_analysis.pandapower.cascade.configuration import OverloadConfig

SCALAR = 1.0


def _overload(**overrides: float) -> OverloadConfig:
    return OverloadConfig(current_loading_threshold=SCALAR, **overrides)


class TestNoOverrides:
    """With nothing set, every branch is compared against the scalar threshold."""

    @pytest.mark.parametrize("element_table", ["line", "trafo", "trafo3w", "impedance"])
    @pytest.mark.parametrize("basecase", [True, False])
    def test_everything_uses_the_scalar(self, element_table: str, basecase: bool):
        assert _overload().threshold(element_table, basecase=basecase) == SCALAR


class TestLineThresholds:
    def test_basecase_uses_the_basecase_line_value(self):
        assert _overload(basecase_line=1.5).threshold("line", basecase=True) == 1.5

    def test_contingency_prefers_its_own_value(self):
        cfg = _overload(basecase_line=1.5, contingency_line=1.8)

        assert cfg.threshold("line", basecase=False) == 1.8

    def test_contingency_falls_back_to_the_basecase_value(self):
        assert _overload(basecase_line=1.5).threshold("line", basecase=False) == 1.5

    def test_a_contingency_only_value_leaves_the_basecase_on_the_scalar(self):
        cfg = _overload(contingency_line=1.8)

        assert cfg.threshold("line", basecase=True) == SCALAR
        assert cfg.threshold("line", basecase=False) == 1.8


class TestTransformerThresholds:
    """trafo and trafo3w share one pair of overrides."""

    @pytest.mark.parametrize("element_table", ["trafo", "trafo3w"])
    def test_basecase_uses_the_basecase_transformer_value(self, element_table: str):
        assert _overload(basecase_transformer=1.8).threshold(element_table, basecase=True) == 1.8

    @pytest.mark.parametrize("element_table", ["trafo", "trafo3w"])
    def test_contingency_prefers_its_own_value(self, element_table: str):
        cfg = _overload(basecase_transformer=1.8, contingency_transformer=2.0)

        assert cfg.threshold(element_table, basecase=False) == 2.0

    @pytest.mark.parametrize("element_table", ["trafo", "trafo3w"])
    def test_contingency_falls_back_to_the_basecase_value(self, element_table: str):
        assert _overload(basecase_transformer=1.8).threshold(element_table, basecase=False) == 1.8


class TestAxesDoNotLeak:
    """A threshold set for one element type must not reach the other."""

    def test_a_line_override_does_not_move_transformers(self):
        cfg = _overload(basecase_line=1.5, contingency_line=1.5)

        assert cfg.threshold("trafo", basecase=True) == SCALAR
        assert cfg.threshold("trafo", basecase=False) == SCALAR

    def test_a_transformer_override_does_not_move_lines(self):
        cfg = _overload(basecase_transformer=1.8, contingency_transformer=1.8)

        assert cfg.threshold("line", basecase=True) == SCALAR
        assert cfg.threshold("line", basecase=False) == SCALAR

    @pytest.mark.parametrize("basecase", [True, False])
    def test_other_tables_always_use_the_scalar(self, basecase: bool):
        """impedance has no dedicated threshold, so no override may reach it."""
        cfg = _overload(
            basecase_line=1.5,
            contingency_line=1.5,
            basecase_transformer=1.8,
            contingency_transformer=1.8,
        )

        assert cfg.threshold("impedance", basecase=basecase) == SCALAR


def test_the_scalar_threshold_is_required():
    """It is the root of every fallback chain, so it has no default."""
    with pytest.raises(ValueError):
        OverloadConfig()
