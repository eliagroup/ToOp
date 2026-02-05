# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pypowsybl.network as pn
from pypowsybl.network.impl.svg import Svg
from toop_engine_grid_helpers.powsybl.single_line_diagram.constants import (
    BRIGHT_MODE_STYLE,
    DARK_MODE_STYLE,
    SLD_CSS,
)
from toop_engine_grid_helpers.powsybl.single_line_diagram.get_single_line_diagram_custom import (
    get_single_line_diagram_custom,
)


def test_get_single_line_diagram_custom():
    net = pn.create_four_substations_node_breaker_network()
    component_library = "Convergence"
    sld_param = pn.SldParameters(
        use_name=True,
        component_library=component_library,
        nodes_infos=True,
        display_current_feeder_info=True,
    )
    id = net.get_voltage_levels().index[1]
    svg_org = net.get_single_line_diagram(id, parameters=sld_param)

    assert isinstance(svg_org, Svg)
    svg = get_single_line_diagram_custom(net, id, parameters=sld_param, custom_style=None)
    assert isinstance(svg, Svg)
    assert svg._content == svg_org._content, "SVG content should be the same when no custom style is applied"

    svg = get_single_line_diagram_custom(net, id, parameters=sld_param, custom_style="dark_mode")
    assert isinstance(svg, Svg)
    for value in DARK_MODE_STYLE.values():
        assert value in svg._content, f"value '{value}' not found in SVG content"
    assert SLD_CSS in svg._content, "SLD_CSS not found in SVG content"

    svg = get_single_line_diagram_custom(net, id, parameters=sld_param, custom_style="bright_mode")
    assert isinstance(svg, Svg)
    for value in BRIGHT_MODE_STYLE.values():
        assert value in svg._content, f"value '{value}' not found in SVG content"

    svg = get_single_line_diagram_custom(net, id, parameters=None, custom_style="bright_mode")
    assert isinstance(svg, Svg)


def test_get_single_line_diagram_custom_highlight():
    net = pn.create_four_substations_node_breaker_network()
    component_library = "Convergence"
    sld_param = pn.SldParameters(
        use_name=True,
        component_library=component_library,
        nodes_infos=True,
        display_current_feeder_info=True,
    )
    id = net.get_voltage_levels().index[1]
    svg_org = net.get_single_line_diagram(id, parameters=sld_param)

    assert isinstance(svg_org, Svg)
    svg = get_single_line_diagram_custom(net, id, parameters=sld_param, custom_style="dark_mode")
    assert ' highlight"' not in svg._content, "SVG content should not contain the highlight class for switches"
    assert isinstance(svg, Svg)

    switches = net.get_switches()
    switches = switches[switches["open"]]
    switches = switches.index.tolist()[0:4]
    svg_highlight = get_single_line_diagram_custom(
        net, id, parameters=sld_param, custom_style="bright_mode", highlight_grid_model_ids=switches
    )
    assert isinstance(svg_highlight, Svg)
    assert "#0050fcff" in svg_highlight._content, "SVG content should contain the changelog color for switches"
    assert ' highlight"' in svg_highlight._content, "SVG content should contain the highlight class for switches"

    switches = net.get_switches()
    switches = switches[switches["open"]]
    switches = switches.index.tolist()[0:4]
    svg_highlight = get_single_line_diagram_custom(
        net, id, parameters=sld_param, custom_style="dark_mode", highlight_grid_model_ids=switches
    )
    assert isinstance(svg_highlight, Svg)
    assert "#0050fcff" in svg_highlight._content, "SVG content should contain the changelog color for switches"
    assert ' highlight"' in svg_highlight._content, "SVG content should contain the highlight class for switches"

    switches = net.get_switches()
    switches = switches[switches["open"]]
    switches = switches.index.tolist()[0:4]
    svg_highlight = get_single_line_diagram_custom(
        net,
        id,
        parameters=sld_param,
        custom_style="dark_mode",
        highlight_grid_model_ids=switches,
        highlight_color="#cc50fcff",
    )
    assert isinstance(svg_highlight, Svg)
    assert "#cc50fcff" in svg_highlight._content, "SVG content should contain the changelog color for switches"
    assert ' highlight"' in svg_highlight._content, "SVG content should contain the highlight class for switches"

    # insert all changed switches should not throw an error, but change all available switches
    switches = net.get_switches()
    switches = switches.index.tolist()
    svg_highlight = get_single_line_diagram_custom(
        net, id, parameters=sld_param, custom_style="bright_mode", highlight_grid_model_ids=switches
    )
    assert isinstance(svg_highlight, Svg)
    assert "#0050fcff" in svg_highlight._content, "SVG content should contain the changelog color for switches"
    assert ' highlight"' in svg_highlight._content, "SVG content should contain the highlight class for switches"
