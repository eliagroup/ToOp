# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for SLD (Single Line Diagram) processing."""

import re
import typing
from collections import Counter

T = typing.TypeVar("T")


def extract_sld_bus_numbers(tags: list[str]) -> list[str]:
    """Extract all 'sld-bus' tags with their numbers from a list of strings.

    Extracts all css classes e.g. sld-bus-0, sld-bus-1, etc. from a list of strings.

    Parameters
    ----------
    tags : list[str]
        List of strings containing 'sld-bus' tags.

    Returns
    -------
    list[str]
        List of extracted 'sld-bus' tags with their numbers.
    """
    # Extract all 'sld-bus' tags with their numbers from each string in the list
    results = []
    for tag_str in tags:
        matches = re.findall(r"sld-bus-\d+", tag_str)
        results.extend(matches)
    return results


def get_most_common_bus(tags: list[T]) -> T:
    """Get the most common Value of a list.

    Parameters
    ----------
    tags : list[Any]
        List of tags to analyze

    Returns
    -------
    most_common_tag: T
        The most common Value from the list
    """
    tag_counter = Counter(tags)
    most_common_tag, _ = tag_counter.most_common(1)[0]
    return most_common_tag
