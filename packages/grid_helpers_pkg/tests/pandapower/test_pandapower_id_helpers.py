# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    SEPARATOR,
    get_globally_unique_id,
    parse_globally_unique_id,
    table_id,
    table_ids,
)


@pytest.mark.parametrize(
    "elem_id, elem_type, expected",
    [
        (1, "bus", f"1{SEPARATOR}bus"),
        ("42", "line", f"42{SEPARATOR}line"),
        (0, None, f"0{SEPARATOR}"),
        ("abc", "trafo", f"abc{SEPARATOR}trafo"),
        (5, "", f"5{SEPARATOR}"),
    ],
)
def test_get_globally_unique_id(elem_id, elem_type, expected):
    assert get_globally_unique_id(elem_id, elem_type) == expected


@pytest.mark.parametrize(
    "globally_unique_id, expected_id, expected_type",
    [
        (f"1{SEPARATOR}bus", 1, "bus"),
        (f"42{SEPARATOR}line", 42, "line"),
        (f"0{SEPARATOR}", 0, ""),
        (f"123{SEPARATOR}trafo", 123, "trafo"),
        (f"5{SEPARATOR}", 5, ""),
    ],
)
def test_parse_globally_unique_id(globally_unique_id, expected_id, expected_type):
    elem_id, elem_type = parse_globally_unique_id(globally_unique_id)
    assert elem_id == expected_id
    assert elem_type == expected_type


@pytest.mark.parametrize(
    "globally_unique_id",
    [
        f"abc{SEPARATOR}bus",  # non-integer id
        f"{SEPARATOR}bus",  # missing id
        "1",  # missing separator and type
        "",  # empty string
        f"1{SEPARATOR}bus{SEPARATOR}extra",  # too many separators
    ],
)
def test_parse_globally_unique_id_invalid(globally_unique_id):
    with pytest.raises(Exception):
        parse_globally_unique_id(globally_unique_id)


@pytest.mark.parametrize(
    "globally_unique_id, expected_id",
    [
        (f"1{SEPARATOR}bus", 1),
        (f"42{SEPARATOR}line", 42),
        (f"0{SEPARATOR}", 0),
        (f"123{SEPARATOR}trafo", 123),
        (f"5{SEPARATOR}", 5),
    ],
)
def test_table_id_valid(globally_unique_id, expected_id):
    assert table_id(globally_unique_id) == expected_id


@pytest.mark.parametrize(
    "globally_unique_id",
    [
        f"abc{SEPARATOR}bus",  # non-integer id
        f"{SEPARATOR}bus",  # missing id
        "1",  # missing separator and type
        "",  # empty string
        f"1{SEPARATOR}bus{SEPARATOR}extra",  # too many separators
    ],
)
def test_table_id_invalid(globally_unique_id):
    with pytest.raises(Exception):
        table_id(globally_unique_id)


@pytest.mark.parametrize(
    "globally_unique_ids, expected_ids",
    [
        ([f"1{SEPARATOR}bus", f"2{SEPARATOR}line", f"3{SEPARATOR}trafo"], [1, 2, 3]),
        ([f"0{SEPARATOR}", f"42{SEPARATOR}bus"], [0, 42]),
        ([f"5{SEPARATOR}", f"10{SEPARATOR}"], [5, 10]),
        ([], []),
    ],
)
def test_table_ids_valid(globally_unique_ids, expected_ids):
    assert table_ids(globally_unique_ids) == expected_ids


@pytest.mark.parametrize(
    "globally_unique_ids",
    [
        [f"abc{SEPARATOR}bus", f"1{SEPARATOR}bus"],  # first invalid
        [f"1{SEPARATOR}bus", f"xyz{SEPARATOR}line"],  # second invalid
        [f"{SEPARATOR}bus"],  # missing id
        ["1"],  # missing separator and type
        [""],  # empty string
        [f"1{SEPARATOR}bus{SEPARATOR}extra"],  # too many separators
        [f"1{SEPARATOR}bus", f"2{SEPARATOR}line", f"bad{SEPARATOR}trafo"],  # last invalid
    ],
)
def test_table_ids_invalid(globally_unique_ids):
    with pytest.raises(Exception):
        table_ids(globally_unique_ids)
