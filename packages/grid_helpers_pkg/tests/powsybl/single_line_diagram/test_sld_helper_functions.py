# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from toop_engine_grid_helpers.powsybl.single_line_diagram.sld_helper_functions import (
    extract_sld_bus_numbers,
    get_most_common_bus,
)


def test_extract_sld_vl_numbers():
    tags = ["foo sld-bus-0 bar"]
    result = extract_sld_bus_numbers(tags)
    assert result == ["sld-bus-0"]

    tags = ["sld-bus-0 sld-bus-1"]
    result = extract_sld_bus_numbers(tags)
    assert result == ["sld-bus-0", "sld-bus-1"]

    tags = ["foo sld-bus-0", "bar sld-bus-1 baz", "no-match-here", "sld-bus-2"]
    result = extract_sld_bus_numbers(tags)
    assert result == ["sld-bus-0", "sld-bus-1", "sld-bus-2"]

    tags = ["foo bar", "baz qux"]
    result = extract_sld_bus_numbers(tags)
    assert result == []

    tags = []
    result = extract_sld_bus_numbers(tags)
    assert result == []

    tags = ["sld_bus", "sld-bus-", "sld-bus-0"]
    result = extract_sld_bus_numbers(tags)
    assert result == ["sld-bus-0"]


def test_get_most_common_vl():
    tags = ["a"]
    assert get_most_common_bus(tags) == "a"

    tags = ["a", "b", "a", "c", "a", "b"]
    assert get_most_common_bus(tags) == "a"

    tags = ["a", "b", "b", "a"]
    # Counter.most_common returns the first encountered in case of tie
    assert get_most_common_bus(tags) in ("a", "b")

    tags = [1, 2, 2, 3, 1, 2]
    assert get_most_common_bus(tags) == 2

    tags = ["x", 1, "x", 2, 1, "x"]
    assert get_most_common_bus(tags) == "x"

    with pytest.raises(IndexError):
        get_most_common_bus([])
