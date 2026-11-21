# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to replace powsybl Convergence styles in SVG files.

Author: Benjamin Petrick
Date: 2025-07-15

"""

import re
import xml.etree.ElementTree as StdETree

from beartype.typing import Literal, Optional
from defusedxml import ElementTree
from toop_engine_grid_helpers.powsybl.single_line_diagram.constants import (
    BREAKER_STYLE,
    BRIGHT_MODE_STYLE,
    BUSBAR_SECTION_LABEL_POSITION,
    DARK_MODE_STYLE,
    DISCONNECTOR_STYLE,
    SLD_CSS,
)
from toop_engine_grid_helpers.powsybl.single_line_diagram.sld_helper_functions import (
    extract_sld_bus_numbers,
    get_most_common_bus,
)


def _class_of(element: StdETree.Element) -> str:
    """Return the element's ``class`` attribute, or an empty string when unset."""
    return element.get("class") or ""


def replace_sld_disconnected(element: StdETree.Element) -> None:
    """Replace sld-disconnected with the most common vl tag in the SVG tree.

    Parameters
    ----------
    element: StdETree.Element
        The element to search for sld-intern-cell and sld-extern-cell
    """
    if "sld-intern-cell" in _class_of(element) or "sld-extern-cell" in _class_of(element):
        # iterate over all children of the element
        # get the most common sld-vl color from the children
        class_tags = [cls for child in element if (cls := child.get("class")) is not None]
        vl_tags = extract_sld_bus_numbers(class_tags)
        if len(vl_tags) == 0:
            return
        most_common_vl = get_most_common_bus(vl_tags)
        # replace sld-disconnected with the most common vl tag
        for child in element:
            class_content = child.get("class")
            if class_content is not None and "sld-disconnected" in class_content:
                new_class_content = class_content.replace("sld-disconnected", most_common_vl)
                child.set("class", new_class_content)


def move_transform(child: StdETree.Element, move_by_item: float) -> None:
    """Move the transform attribute of a child element by a given value.

    Parameters
    ----------
    child : xml.etree.ElementTree.Element
        The child element to modify.
    move_by_item : float
        The value to move the transform by.
    """
    transform = child.get("transform")
    # content is "translate(X,Y)"
    # set y
    if transform is not None:
        match = re.search(r"translate\(([^,]+),([^,]+)\)", transform)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            # move y by move_by_item
            y += move_by_item
            # update the transform attribute
            child.set("transform", f"translate({x},{y})")


def move_labels_p_q_i(element: StdETree.Element, move_by: float = 0.0) -> None:
    """Move labels for active power, reactive power, and current.

    Avoids overlapping labels of the three-wt, moves all labels for consistent positioning.

    Parameters
    ----------
    element : StdETree.Element
        The element to search for sld-intern-cell and sld-extern-cell.
    move_by : float, optional
        The distance to move the labels, by default 0.0.
    """
    if "sld-intern-cell" in _class_of(element) or "sld-extern-cell" in _class_of(element):
        if "sld-cell-direction-bottom" in _class_of(element):
            move_by_item = move_by
        else:
            move_by_item = -move_by
        # iterate over all children of the element
        for child in element:
            child_class = child.get("class")
            if child_class is None:
                continue
            if "sld-active-power" in child_class or "sld-current" in child_class or "sld-reactive-power" in child_class:
                move_transform(child, move_by_item)


def replace_disconnector(element: StdETree.Element, disconnector_style: Optional[dict[str, str]] = None) -> None:
    """Replace the powsybl Convergence disconnector with the given style.

    Parameter
    ---------
    element: StdETree.Element
        The element to search for sld-disconnector
    disconnector_style: dict[str, str]
        The disconnector style used as the replacement,
        expects a dict for a rect like:
        {"height": "8", "width": "8", "x": "0", "y": "0", "transform": "rotate(45.0,4.0,4.0)"}
    """
    if disconnector_style is None:
        disconnector_style = DISCONNECTOR_STYLE

    if "sld-disconnector" in _class_of(element):
        # Find the <rect> child inside this <g> element

        for child in element:
            if child.tag.endswith("path"):
                # replace path with path
                child.tag = "{http://www.w3.org/2000/svg}rect"
                child.attrib.clear()
                child.attrib.update(disconnector_style)


def replace_breaker(element: StdETree.Element, breaker_style: Optional[dict[str, str]] = None) -> None:
    """Replace the powsybl Convergence breaker with the given style.

    Parameter
    ---------
    element: StdETree.Element
        The element to search for sld-breaker
    breaker_style: dict[str, str]
        The breaker style, expects a dict for a path like: "d": "M1,1 V19 H19 V1z"
    """
    if breaker_style is None:
        breaker_style = BREAKER_STYLE

    if "sld-breaker" in _class_of(element):
        for child in element:
            if child.tag.endswith("path"):
                child.attrib.update(breaker_style)


def move_busbar_section_label(
    element: StdETree.Element, busbar_section_label_position: Optional[dict[str, str]] = None
) -> None:
    """Move the busbar section label up, to avoid overlapping.

    Parameter
    ---------
    element: StdETree.Element
        The element to search for sld-busbar-section
    busbar_section_label_position: dict[str, str]
        expects values for x and y like: {"x":3, "y":4}
    """
    if busbar_section_label_position is None:
        busbar_section_label_position = BUSBAR_SECTION_LABEL_POSITION

    if "sld-busbar-section" in _class_of(element):
        for child in element:
            child_class = child.get("class")
            if child_class is not None and "sld-label" in child_class:
                # lift sld label of busbar section
                child.attrib.update(busbar_section_label_position)


def switch_dark_to_bright_mode(
    xmlstring: str,
    dark_mode: dict[str, str] = DARK_MODE_STYLE,
    bright_mode: dict[str, str] = BRIGHT_MODE_STYLE,
) -> str:
    """Switches the dark mode to bright mode in the SVG XML string.

    Parameters
    ----------
    xmlstring : str
        The SVG XML string to modify.
    dark_mode : dict[str, str]
        The dark mode styles, a dict with keys that must match the bright_mode and values of
        default dark mode style to be replaced with the bright_mode style.
    bright_mode : dict[str, str]
        The bright mode styles, a dict with keys that mus match the dark_mode and values
        to replace the dark mode values.

    Returns
    -------
    str
        The modified SVG XML string with dark mode styles replaced by bright mode styles.
    """
    for key, bright_mode_value in bright_mode.items():
        # Replace dark_mode styles with bright_mode styles
        xmlstring = xmlstring.replace(dark_mode[key], bright_mode_value)
    return xmlstring


def replace_svg_styles(
    xmlstring: str,
    style: Literal["dark_mode", "bright_mode"] = "dark_mode",
    sld_css: str = SLD_CSS,
    move_p_q_i_labels_by: float = 10.0,
    highlight_color: Optional[str] = "#0050fcff",
    highlight_grid_model_ids: Optional[list[str]] = None,
) -> str:
    """Replace powsybl Convergence with dark or bright mode.

    Parameter
    ---------
    xmlstring: str
        The svg xmls string to load and replace the style
    style: Literal["dark_mode", "bright_mode"]
        Choose the style
    dark_mode: dict[str, str]
        The dark mode styles, a dict with keys that must match the bright_mode and values of
        default dark mode style to be replaced with the bright_mode style.
    bright_mode: dict[str, str]
        The bright mode styles, a dict with keys that mus match the dark_mode and values
        to replace the dark mode values.
    sld_css: str
        The SLD CSS to be added to the SVG.
    move_p_q_i_labels_by: float
        The distance to move the labels for active power, reactive power, and current.
        Default is 10.0.
    highlight_color: Optional[str]
        The color to use for highlighting elements in the SVG.
    highlight_grid_model_ids: Optional[list[str]]
        The list of grid model IDs to highlight in the SVG.
    """
    tree = ElementTree.fromstring(xmlstring)

    # Find all elements with class="sld-disconnector sld-open sld-disconnected"
    for elem in tree.iter():
        if elem.tag == "{http://www.w3.org/2000/svg}style":
            sld_css = set_highlight_color(xmlstring=sld_css, highlight_color=highlight_color)
            elem.text = sld_css
        if elem.get("class") is None:
            continue

        replace_disconnector(elem)
        replace_breaker(elem)
        move_busbar_section_label(elem)
        move_labels_p_q_i(elem, move_by=move_p_q_i_labels_by)
        replace_sld_disconnected(elem)
        if highlight_grid_model_ids is not None:
            set_highlight_color_for_ids(elem, grid_model_ids=highlight_grid_model_ids)

    # write tree to string
    xmlstring = ElementTree.tostring(tree, encoding="unicode", xml_declaration=True)
    # replace some defuse_et artifacts
    if style == "bright_mode":
        xmlstring = switch_dark_to_bright_mode(xmlstring)

    xmlstring = remove_defuse_artifacts(xmlstring=xmlstring)
    return xmlstring


def set_highlight_color_for_ids(
    element: StdETree.Element,
    grid_model_ids: list[str],
) -> None:
    """Replace the SVG style given an id.

    Parameter
    ---------
    xmlstring: str
        The svg xmls string to load and replace the style
    grid_model_ids: list[str]
        The list of grid model IDs to replace in the SVG.
    sld_css: str
        The SLD CSS to be added to the SVG.

    """
    element_class = element.get("class")
    element_id = element.get("id")
    if element_class is None or element_id is None:
        return

    for id in grid_model_ids:
        svg_id = re.sub(r"_(\d+)_", lambda match: chr(int(match.group(1))), element_id)
        if f"id{id}" == svg_id:
            # set the highlight class
            element.set("class", f"{element_class} highlight")


def set_highlight_color(xmlstring: str, highlight_color: Optional[str] = None) -> str:
    """Set the color for the highlight color

    Parameters
    ----------
    xmlstring : str
        The SVG XML string to modify.
    highlight_color : str, optional
        The color to set for the highlight elements, by default "#0050fcff".

    Returns
    -------
    str

    """
    # set the highlight color
    default_color = "#0050fcff"
    effective_color = highlight_color if highlight_color is not None else default_color
    if default_color != effective_color:
        content_str = ".highlight {--sld-vl-color: " + default_color + "}"
        replace_str = ".highlight {--sld-vl-color: " + effective_color + "}"
        xmlstring = xmlstring.replace(content_str, replace_str)

    return xmlstring


def remove_defuse_artifacts(xmlstring: str) -> str:
    """Remove defuse_et artifacts from the SVG XML string.

    xmlstring: str
        The svg xmls string to load and replace the style
    """
    xmlstring = xmlstring.replace(":ns0", "").replace("ns0:", "")
    xmlstring = xmlstring.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    xmlstring = xmlstring.replace('encoding="utf-8"', 'encoding="UTF-8" standalone="no"')
    return xmlstring
