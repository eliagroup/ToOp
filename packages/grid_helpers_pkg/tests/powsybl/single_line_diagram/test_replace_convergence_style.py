# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from xml.etree.ElementTree import Element

import pypowsybl.network as pn
from pypowsybl.network.impl.svg import Svg
from toop_engine_grid_helpers.powsybl.single_line_diagram.constants import (
    BREAKER_STYLE,
    BRIGHT_MODE_STYLE,
    BUSBAR_SECTION_LABEL_POSITION,
    DARK_MODE_STYLE,
    DISCONNECTOR_STYLE,
    SLD_CSS,
)
from toop_engine_grid_helpers.powsybl.single_line_diagram.replace_convergence_style import (
    move_busbar_section_label,
    move_labels_p_q_i,
    move_transform,
    replace_breaker,
    replace_disconnector,
    replace_sld_disconnected,
    replace_svg_styles,
    switch_dark_to_bright_mode,
)


def make_label_element(label_class, transform=None):
    attribs = {"class": label_class}
    if transform:
        attribs["transform"] = transform
    return Element("text", attribs)


def make_element_with_children(parent_class, child_classes):
    parent = Element("g", {"class": parent_class})
    for cls in child_classes:
        child = Element("rect", {"class": cls})
        parent.append(child)
    return parent


def test_replace_sld_disconnected_replaces_tag():
    parent = make_element_with_children("sld-intern-cell", ["sld-disconnected", "sld-bus-0", "sld-bus-0 sld-disconnected"])
    replace_sld_disconnected(parent)
    classes = [child.get("class") for child in parent]
    # All "sld-disconnected" should be replaced with "sld-bus-0"
    assert "sld-bus-0" in classes
    assert "sld-bus-0 sld-bus-0" in classes
    assert all("sld-disconnected" not in c for c in classes if c)

    # no_vl_tags, intended use case
    parent = make_element_with_children("sld-extern-cell", ["sld-disconnected", "foo", "bar"])
    replace_sld_disconnected(parent)
    # No replacement should happen since no vl tags
    classes = [child.get("class") for child in parent]
    assert "sld-disconnected" in classes

    # mixed_children
    parent = make_element_with_children(
        "sld-intern-cell", ["sld-disconnected", "sld-bus-1", "sld-bus-1 sld-disconnected", "sld-bus-0"]
    )
    replace_sld_disconnected(parent)
    classes = [child.get("class") for child in parent]
    # All sld-disconnected replaced with most common vl (first found: sld-bus-1)
    assert "sld-bus-1" in classes
    assert "sld-bus-1 sld-bus-1" in classes
    assert "sld-bus-0" in classes
    assert all("sld-disconnected" not in c for c in classes if c)


def test_move_transform():
    child = Element("rect", {"transform": "translate(10,20)"})
    move_transform(child, 5.0)
    assert child.get("transform") == "translate(10.0,25.0)"

    child = Element("rect", {"transform": "translate(5,15)"})
    move_transform(child, -10.0)
    assert child.get("transform") == "translate(5.0,5.0)"

    # missing transform attribute
    child = Element("rect")
    move_transform(child, 10.0)
    # Should not raise or set transform
    assert child.get("transform") is None

    # wrong transform content -> do not modify
    child = Element("rect", {"transform": "rotate(45)"})
    move_transform(child, 10.0)
    # Should not change the attribute
    assert child.get("transform") == "rotate(45)"

    # test float
    child = Element("rect", {"transform": "translate(3.5,7.25)"})
    move_transform(child, 2.75)
    assert child.get("transform") == "translate(3.5,10.0)"


def test_move_labels_p_q_i_moves_labels_down():
    parent = Element("g", {"class": "sld-intern-cell sld-cell-direction-bottom"})
    label1 = make_label_element("sld-active-power", "translate(0.0,0.0)")
    label2 = make_label_element("sld-current", "translate(1.0,2.0)")
    label3 = make_label_element("sld-reactive-power", "translate(2.0,3.0)")
    not_label = make_label_element("not-a-label", "translate(3.0,4.0)")
    parent.extend([label1, label2, label3, not_label])

    move_labels_p_q_i(parent, move_by=15.0)
    assert label1.get("transform") == "translate(0.0,15.0)"
    assert label2.get("transform") == "translate(1.0,17.0)"
    assert label3.get("transform") == "translate(2.0,18.0)"
    assert not_label.get("transform") == "translate(3.0,4.0)"  # should not change

    parent = Element("g", {"class": "foo-bar"})
    label = make_label_element("sld-active-power", "translate(0.0,0.0)")
    parent.append(label)
    move_labels_p_q_i(parent)
    assert label.get("transform") == "translate(0.0,0.0)"  # should not change
    # No exception means pass


def test_replace_disconnector_replaces():
    # Use a custom style for testing
    custom_style = {"height": "10", "width": "12", "x": "1", "y": "2", "transform": "rotate(90,5,5)"}
    # Create a <g> element with class sld-disconnector and a <path> child
    parent = Element("g", {"class": "sld-disconnector"})
    path = Element("{http://www.w3.org/2000/svg}path", {"d": "M0,0L1,1"})
    parent.append(path)
    # Add another child that should not be touched
    untouched = Element("{http://www.w3.org/2000/svg}circle", {"r": "5"})
    parent.append(untouched)
    replace_disconnector(parent, disconnector_style=custom_style)
    # The path should now be a rect with the custom style
    assert path.tag == "{http://www.w3.org/2000/svg}rect"
    assert path.attrib == custom_style
    # The untouched child should remain unchanged
    assert untouched.tag == "{http://www.w3.org/2000/svg}circle"
    assert untouched.attrib == {"r": "5"}

    custom_style = DISCONNECTOR_STYLE
    # Create a <g> element with class sld-disconnector and a <path> child
    parent = Element("g", {"class": "sld-disconnector"})
    path = Element("{http://www.w3.org/2000/svg}path", {"d": "M0,0L1,1"})
    parent.append(path)
    # Add another child that should not be touched
    untouched = Element("{http://www.w3.org/2000/svg}circle", {"r": "5"})
    parent.append(untouched)
    replace_disconnector(parent, disconnector_style=custom_style)
    # The path should now be a rect with the custom style
    assert path.tag == "{http://www.w3.org/2000/svg}rect"
    assert path.attrib == custom_style

    # test default style
    parent = Element("g", {"class": "sld-disconnector"})
    path = Element("{http://www.w3.org/2000/svg}path", {"d": "M0,0L1,1"})
    parent.append(path)
    replace_disconnector(parent)
    assert path.tag == "{http://www.w3.org/2000/svg}rect"
    assert path.attrib == DISCONNECTOR_STYLE


def test_replace_disconnector_does_nothing_if_no_disconnector_class():
    parent = Element("g", {"class": "not-a-disconnector"})
    path = Element("{http://www.w3.org/2000/svg}path", {"d": "M0,0L1,1"})
    parent.append(path)
    orig_tag = path.tag
    orig_attrib = dict(path.attrib)

    replace_disconnector(parent, disconnector_style={"foo": "bar"})

    # Should not change anything
    assert path.tag == orig_tag
    assert path.attrib == orig_attrib


def test_replace_disconnector_does_nothing_if_no_path_child():
    parent = Element("g", {"class": "sld-disconnector"})
    rect = Element("{http://www.w3.org/2000/svg}rect", {"height": "5"})
    parent.append(rect)
    orig_tag = rect.tag
    orig_attrib = dict(rect.attrib)

    replace_disconnector(parent, disconnector_style={"foo": "bar"})

    # Should not change the rect
    assert rect.tag == orig_tag
    assert rect.attrib == orig_attrib


def test_replace_breaker_replaces():
    # Custom style for testing
    custom_style = {"d": "M1,1 V19 H19 V1z", "stroke": "red"}
    parent = Element("g", {"class": "sld-breaker"})
    path = Element("{http://www.w3.org/2000/svg}path", {"d": "M0,0L1,1", "stroke": "black"})
    parent.append(path)
    untouched = Element("{http://www.w3.org/2000/svg}circle", {"r": "5"})
    parent.append(untouched)

    replace_breaker(parent, breaker_style=custom_style)
    # Path should have updated attributes
    for k, v in custom_style.items():
        assert path.attrib[k] == v
    # Old attributes not in custom_style should remain if not overwritten
    assert "stroke" in path.attrib
    # Untouched child should remain unchanged
    assert untouched.attrib == {"r": "5"}

    parent = Element("g", {"class": "sld-breaker"})
    path = Element("{http://www.w3.org/2000/svg}path", {"d": "foo"})
    parent.append(path)
    replace_breaker(parent)
    for k, v in BREAKER_STYLE.items():
        assert path.attrib[k] == v

    parent = Element("g", {"class": "not-a-breaker"})
    path = Element("{http://www.w3.org/2000/svg}path", {"d": "foo"})
    parent.append(path)
    orig_attrib = dict(path.attrib)
    replace_breaker(parent, breaker_style={"d": "bar"})
    assert path.attrib == orig_attrib

    parent = Element("g", {"class": "sld-breaker"})
    rect = Element("{http://www.w3.org/2000/svg}rect", {"height": "5"})
    parent.append(rect)
    orig_attrib = dict(rect.attrib)
    replace_breaker(parent, breaker_style={"d": "bar"})
    assert rect.attrib == orig_attrib


def test_move_busbar_section_label_updates_label_position():
    # Custom position for testing
    custom_position = {"x": "10", "y": "20"}
    # Element with sld-busbar-section and a child with sld-label
    parent = Element("g", {"class": "sld-busbar-section"})
    label = Element("text", {"class": "sld-label", "x": "1", "y": "2"})
    parent.append(label)
    move_busbar_section_label(parent, busbar_section_label_position=custom_position)
    assert label.attrib["x"] == "10"
    assert label.attrib["y"] == "20"
    custom_position = BUSBAR_SECTION_LABEL_POSITION
    # Test with default position
    parent = Element("g", {"class": "sld-busbar-section"})
    label = Element("text", {"class": "sld-label", "x": "3", "y": "4"})
    parent.append(label)
    move_busbar_section_label(parent)
    assert label.attrib["x"] == custom_position["x"]
    assert label.attrib["y"] == custom_position["y"]

    parent = Element("g", {"class": "sld-busbar-section"})
    not_label = Element("text", {"class": "not-a-label", "x": "1", "y": "2"})
    parent.append(not_label)
    orig_attrib = dict(not_label.attrib)
    move_busbar_section_label(parent, busbar_section_label_position={"x": "9", "y": "8"})
    assert not_label.attrib == orig_attrib

    parent = Element("g", {"class": "not-busbar-section"})
    label = Element("text", {"class": "sld-label", "x": "1", "y": "2"})
    parent.append(label)
    orig_attrib = dict(label.attrib)
    move_busbar_section_label(parent, busbar_section_label_position={"x": "9", "y": "8"})
    assert label.attrib == orig_attrib


def test_switch_dark_to_bright_mode_basic_replacement():
    dark_mode = {"background": "#222", "text": "#fff"}
    bright_mode = {"background": "#333", "text": "#000"}
    xmlstring = "<svg><style>.background{fill:#222;}.text{fill:#fff;}</style></svg>"
    result = switch_dark_to_bright_mode(xmlstring, dark_mode=dark_mode, bright_mode=bright_mode)
    assert "#222" not in result
    assert "#333" in result
    assert "#000" in result
    assert "#222" not in result
    assert "#fff" not in result


def test_replace_svg_styles():
    net = pn.create_four_substations_node_breaker_network()
    component_library = "Convergence"
    sld_param = pn.SldParameters(
        use_name=True,
        component_library=component_library,
        nodes_infos=True,
        display_current_feeder_info=True,
    )
    id = net.get_voltage_levels().index[1]
    svg = net.get_single_line_diagram(id, parameters=sld_param)
    svg._content = replace_svg_styles(xmlstring=svg._content, style="dark_mode")

    assert isinstance(svg, Svg)
    for value in DARK_MODE_STYLE.values():
        assert value in svg._content, f"value '{value}' not found in SVG content"
    assert SLD_CSS in svg._content, "SLD_CSS not found in SVG content"

    svg = net.get_single_line_diagram(id, parameters=sld_param)
    svg._content = replace_svg_styles(xmlstring=svg._content, style="bright_mode")
    assert isinstance(svg, Svg)
    for value in BRIGHT_MODE_STYLE.values():
        assert value in svg._content, f"value '{value}' not found in SVG content"
