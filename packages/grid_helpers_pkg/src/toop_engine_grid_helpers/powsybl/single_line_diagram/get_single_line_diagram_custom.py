# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""A replacement for the powsybl get_single_line_diagram function."""

import pypowsybl.network as pn
from beartype.typing import Literal, Optional
from pypowsybl.network.impl.svg import Svg
from toop_engine_grid_helpers.powsybl.single_line_diagram.replace_convergence_style import replace_svg_styles

custom_style_options = Literal["dark_mode", "bright_mode"]


def get_single_line_diagram_custom(
    net: pn.Network,
    container_id: str,
    parameters: Optional[pn.SldParameters] = None,
    custom_style: Optional[custom_style_options] = "dark_mode",
    highlight_grid_model_ids: Optional[list[str]] = None,
    highlight_color: str = "#0050fcff",
) -> Svg:
    """Get a single line diagram with custom styles.

    A replacement for the powsybl get_single_line_diagram function that allows
    for custom styles to be applied, specifically dark mode or bright mode.

    Parameters
    ----------
    net : pn.Network
        The network for which to get the single line diagram.
        Hint: If you want the SLD for including loadflow results,
        run the loadflow first.
    container_id : str
        The ID of Voltage level for which to get the single line diagram.
    parameters : Optional[pn.SldParameters]
        Parameters for the single line diagram, by default None.
    custom_style : Optional[Literal["dark_mode", "bright_mode"]], optional
        The custom style to apply, either "dark_mode" or "bright_mode". If None,
        the original style will be used, by default "dark_mode".
    highlight_grid_model_ids : Optional[list[str]], optional
        The IDs of the grid model elements to highlight, by default None.
        E.g. get all switches that changed their state from closed to open -> show the difference between the
        original and new states.
    highlight_color : str, optional
        The color to use for highlighting, by default "#0050fcff".

    """
    if parameters is None:
        parameters = pn.SldParameters(
            use_name=True,
            component_library="Convergence",
            nodes_infos=True,
            display_current_feeder_info=True,
        )
    svg = net.get_single_line_diagram(container_id=container_id, parameters=parameters)
    if custom_style in custom_style_options.__args__:
        svg._content = replace_svg_styles(
            xmlstring=svg._content,
            style=custom_style,
            highlight_color=highlight_color,
            highlight_grid_model_ids=highlight_grid_model_ids,
        )
    # else: keep the original style e.g. Convergence
    return svg
