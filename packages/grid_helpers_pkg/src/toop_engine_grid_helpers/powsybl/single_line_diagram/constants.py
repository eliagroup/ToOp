# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines constants used in the SLD (Single Line Diagram) importer module.

Style guide:
Colors for Voltage Levels
Voltage level >=400kV (220, 255, 220) | #DCFFDC | NEW
Voltage level 380kV (160, 255, 166) | #A0FFA6
Voltage level 220kV (60, 163, 40) | #3CA328
Voltage level 150.sld-bus-170kV (44, 122, 30) | #2C7A1E | NEW
Voltage level 110kV (15, 90, 0)  | #0F5A00
Voltage level 66kV (80, 255, 90)  | #50FF5A | NEW
Voltage level <=33kV (12, 244, 30) | #0CF41E
Grounded (182, 132, 18) | #B68412
No voltage (255, 0, 0) | #FF0000
Data invalid (200, 10, 200) | #C80AC8
Blocked (0, 120, 230) | #0078E6
Bypass busbar (255, 255, 25) | #E1E119
Manual entry (255, 255, 25) | #E1E119

Line thickness: 3px
Switches:
Open: Square 18x18,
Outline 2px inside in the color of the voltage level,
Background color: color/base/levels/04
Closed: Square 18x18,
Color of the voltage level
Disconnector:
Open: Square 16.97x16.97 rotated by 45 degrees, outline 2px inside in the color of the voltage level, background color: color/base/levels/04
Closed: Square 16.97x16.97 rotated by 45 degrees, color of the voltage level

"""

# ignore ruff
# ruff: noqa: E501
# a full replacement of the SLD CSS styles
SLD_CSS = r"""<![CDATA[
/* ----------------------------------------------------------------------- */
/* File: tautologies.css ------------------------------------------------- */
.sld-out .sld-arrow-in {visibility: hidden}
.sld-in .sld-arrow-out {visibility: hidden}
.sld-closed .sld-sw-open {visibility: hidden}
.sld-open .sld-sw-closed {visibility: hidden}
.sld-hidden-node {visibility: hidden}
.sld-top-feeder .sld-label {dominant-baseline: auto}
.sld-bottom-feeder .sld-label {dominant-baseline: hanging}
.sld-active-power .sld-label {dominant-baseline: mathematical}
.sld-reactive-power .sld-label {dominant-baseline: mathematical}
.sld-current .sld-label {dominant-baseline: mathematical}
/* ----------------------------------------------------------------------- */
/* File: topologicalBaseVoltages.css ------------------------------------- */
.sld-disconnected {--sld-vl-color: #FF0000}
.sld-vl300to500.sld-bus-0 {--sld-vl-color: #A0FFA6}
.sld-vl300to500.sld-bus-1 {--sld-vl-color: #D56701}
.sld-vl300to500.sld-bus-2 {--sld-vl-color: #3E8AD0}
.sld-vl300to500.sld-bus-3 {--sld-vl-color: #A0FFA6}
.sld-vl300to500.sld-bus-4 {--sld-vl-color: #A0FFA6}
.sld-vl300to500.sld-bus-5 {--sld-vl-color: #A0FFA6}
.sld-vl300to500.sld-bus-6 {--sld-vl-color: #A0FFA6}
.sld-vl300to500.sld-bus-7 {--sld-vl-color: #A0FFA6}
.sld-vl300to500.sld-bus-8 {--sld-vl-color: #A0FFA6}
.sld-vl300to500.sld-bus-9 {--sld-vl-color: #A0FFA6}
.sld-vl300to500.highlight {--sld-vl-color: #0050fcff}
.sld-vl180to300.sld-bus-0 {--sld-vl-color: #3CA328}
.sld-vl180to300.sld-bus-1 {--sld-vl-color: #7C5CC3}
.sld-vl180to300.sld-bus-2 {--sld-vl-color: #A67D00}
.sld-vl180to300.sld-bus-3 {--sld-vl-color: #3CA328}
.sld-vl180to300.sld-bus-4 {--sld-vl-color: #3CA328}
.sld-vl180to300.sld-bus-5 {--sld-vl-color: #3CA328}
.sld-vl180to300.sld-bus-6 {--sld-vl-color: #3CA328}
.sld-vl180to300.sld-bus-7 {--sld-vl-color: #3CA328}
.sld-vl180to300.sld-bus-8 {--sld-vl-color: #3CA328}
.sld-vl180to300.sld-bus-9 {--sld-vl-color: #3CA328}
.sld-vl180to300.highlight {--sld-vl-color: #0050fcff}
.sld-vl120to180.sld-bus-0 {--sld-vl-color: #2C7A1E}
.sld-vl120to180.sld-bus-1 {--sld-vl-color: #72858C}
.sld-vl120to180.sld-bus-2 {--sld-vl-color: #D94814}
.sld-vl120to180.sld-bus-3 {--sld-vl-color: #2C7A1E}
.sld-vl120to180.sld-bus-4 {--sld-vl-color: #2C7A1E}
.sld-vl120to180.sld-bus-5 {--sld-vl-color: #2C7A1E}
.sld-vl120to180.sld-bus-6 {--sld-vl-color: #2C7A1E}
.sld-vl120to180.sld-bus-7 {--sld-vl-color: #2C7A1E}
.sld-vl120to180.sld-bus-8 {--sld-vl-color: #2C7A1E}
.sld-vl120to180.sld-bus-9 {--sld-vl-color: #2C7A1E}
.sld-vl120to180.highlight {--sld-vl-color: #0050fcff}
.sld-vl70to120.sld-bus-0 {--sld-vl-color: #0F5A00}
.sld-vl70to120.sld-bus-1 {--sld-vl-color: #699720}
.sld-vl70to120.sld-bus-2 {--sld-vl-color: #CB2D64}
.sld-vl70to120.sld-bus-3 {--sld-vl-color: #0F5A00}
.sld-vl70to120.sld-bus-4 {--sld-vl-color: #0F5A00}
.sld-vl70to120.sld-bus-5 {--sld-vl-color: #0F5A00}
.sld-vl70to120.sld-bus-6 {--sld-vl-color: #0F5A00}
.sld-vl70to120.sld-bus-7 {--sld-vl-color: #0F5A00}
.sld-vl70to120.sld-bus-8 {--sld-vl-color: #0F5A00}
.sld-vl70to120.sld-bus-9 {--sld-vl-color: #0F5A00}
.sld-vl70to120.highlight {--sld-vl-color: #0050fcff}
.sld-vl50to70.sld-bus-0 {--sld-vl-color: #50FF5A}
.sld-vl50to70.sld-bus-1 {--sld-vl-color: #50FF5A}
.sld-vl50to70.sld-bus-2 {--sld-vl-color: #50FF5A}
.sld-vl50to70.sld-bus-3 {--sld-vl-color: #50FF5A}
.sld-vl50to70.sld-bus-4 {--sld-vl-color: #50FF5A}
.sld-vl50to70.sld-bus-5 {--sld-vl-color: #50FF5A}
.sld-vl50to70.sld-bus-6 {--sld-vl-color: #50FF5A}
.sld-vl50to70.sld-bus-7 {--sld-vl-color: #50FF5A}
.sld-vl50to70.sld-bus-8 {--sld-vl-color: #50FF5A}
.sld-vl50to70.sld-bus-9 {--sld-vl-color: #50FF5A}
.sld-vl50to70.highlight {--sld-vl-color: #0050fcff}
.sld-vl30to50.sld-bus-0 {--sld-vl-color: #0CF41E}
.sld-vl30to50.sld-bus-1 {--sld-vl-color: #0CF41E}
.sld-vl30to50.sld-bus-2 {--sld-vl-color: #0CF41E}
.sld-vl30to50.sld-bus-3 {--sld-vl-color: #0CF41E}
.sld-vl30to50.sld-bus-4 {--sld-vl-color: #0CF41E}
.sld-vl30to50.sld-bus-5 {--sld-vl-color: #0CF41E}
.sld-vl30to50.sld-bus-6 {--sld-vl-color: #0CF41E}
.sld-vl30to50.sld-bus-7 {--sld-vl-color: #0CF41E}
.sld-vl30to50.sld-bus-8 {--sld-vl-color: #0CF41E}
.sld-vl30to50.sld-bus-9 {--sld-vl-color: #0CF41E}
.sld-vl30to50.highlight {--sld-vl-color: #0050fcff}
.sld-vl0to30.sld-bus-0 {--sld-vl-color: #0CF41E}
.sld-vl0to30.sld-bus-1 {--sld-vl-color: #0CF41E}
.sld-vl0to30.sld-bus-2 {--sld-vl-color: #0CF41E}
.sld-vl0to30.sld-bus-3 {--sld-vl-color: #0CF41E}
.sld-vl0to30.sld-bus-4 {--sld-vl-color: #0CF41E}
.sld-vl0to30.sld-bus-5 {--sld-vl-color: #0CF41E}
.sld-vl0to30.sld-bus-6 {--sld-vl-color: #0CF41E}
.sld-vl0to30.sld-bus-7 {--sld-vl-color: #0CF41E}
.sld-vl0to30.sld-bus-8 {--sld-vl-color: #0CF41E}
.sld-vl0to30.sld-bus-9 {--sld-vl-color: #0CF41E}
.sld-vl0to30.highlight {--sld-vl-color: #0050fcff}


/* ----------------------------------------------------------------------- */
/* File : highlightLineStates.css ---------------------------------------- */
.sld-wire.sld-feeder-disconnected {stroke: #FF0000; stroke-width: 1.5px}
.sld-wire.sld-feeder-disconnected-connected {stroke: #FF0000; stroke-width: 1.5px}
/* File : components.css ------------------------------------------------- */
.sld-disconnector.sld-open {stroke-width: 1.5; stroke: var(--sld-vl-color, black); fill: black}
.sld-disconnector.sld-closed {fill: var(--sld-vl-color, black)}
.sld-breaker, .sld-load-break-switch {fill: var(--sld-vl-color, blue)}
.sld-breaker {stroke: var(--sld-vl-color, blue); stroke-width: 1.5}
.sld-bus-connection {fill: var(--sld-vl-color, black)}
.sld-cell-shape-flat .sld-bus-connection {visibility: hidden}
.sld-busbar-section {stroke: var(--sld-vl-color, black); stroke-width: 3; fill: none}
.sld-wire {stroke: var(--sld-vl-color, #c80000); fill: none; stroke-width: 1.5px}
.sld-wire.sld-dangling-line {stroke-width: 2px}
.sld-wire.sld-tie-line {stroke-width: 2px}

/* Stroke --sld-vl-color with fallback #C80AC8 */
.sld-load {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-battery {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-generator {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-two-wt {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-three-wt {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-winding {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-capacitor {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-inductor {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-pst {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-pst-arrow {stroke: #C80AC8; fill: none; stroke-width: 1.5px}
.sld-svc {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}
.sld-vsc {stroke: var(--sld-vl-color, #C80AC8); font-size: 7.43px; fill: none; stroke-width: 1.5px}
.sld-lcc {stroke: var(--sld-vl-color, #C80AC8); font-size: 7.43px; fill: none; stroke-width: 1.5px}
.sld-ground {stroke: var(--sld-vl-color, #C80AC8); fill: none; stroke-width: 1.5px}

.sld-node-infos {stroke: none; fill: var(--sld-vl-color, black)}
.sld-node {stroke: none; fill: black}
.sld-flash {stroke: none; fill: black}
.sld-lock {stroke: none; fill: black}
.sld-unknown {stroke: none; fill: #C80AC8}
/* Fonts */
.sld-label {stroke: none; fill: white; font: 8px serif}
.sld-bus-legend-info {font: 10px serif; fill: white}
.sld-graph-label {font: 12px serif}
/* Specific */
.sld-grid {stroke: #003700; stroke-dasharray: 1,10}
.sld-feeder-info.sld-active-power {fill: #fff}
.sld-feeder-info.sld-reactive-power {fill: #fff}
.sld-feeder-info.sld-current {fill:#fff}
.sld-frame {fill: var(--sld-background-color, #000)}
/* fictitious switches and busbar */
.sld-breaker.sld-fictitious {stroke: var(--sld-vl-color, #C80AC8); stroke-width: 1.5}
.sld-disconnector.sld-fictitious {stroke: maroon}
.sld-load-break-switch.sld-fictitious {stroke: maroon}
.sld-busbar-section.sld-fictitious {stroke: var(--sld-vl-color, #c80000); stroke-width: 1}
/* ground disconnector specific */
.sld-ground-disconnection-attach {stroke: var(--sld-vl-color, #c80000); fill: none}
.sld-open.sld-ground-disconnection-ground {stroke: black; fill: none}
.sld-closed.sld-ground-disconnection-ground {stroke: var(--sld-vl-color, #c80000); fill: none}
.sld-open.sld-ground-disconnection {stroke: var(--sld-vl-color, black); fill: none}
.sld-closed.sld-ground-disconnection {fill: var(--sld-vl-color, black)}
.sld-breaker.sld-open {fill: transparent}
]]>
"""


# Note: DARK_MODE_STYLE keys and BRIGHT_MODE_STYLE keys must match
DARK_MODE_STYLE = {
    "background": r".sld-frame {fill: var(--sld-background-color, #000)}",
    "active_power": r".sld-feeder-info.sld-active-power {fill: #fff}",
    "reactive_power": r".sld-feeder-info.sld-reactive-power {fill: #fff}",
    "current": r".sld-feeder-info.sld-current {fill:#fff}",
    "label": r".sld-label {stroke: none; fill: white; font: 8px serif}",
    "bus-legend-info": r".sld-bus-legend-info {font: 10px serif; fill: white}",
    "sld-open": r".sld-disconnector.sld-open {stroke-width: 1.5; stroke: var(--sld-vl-color, black); fill: black}",
}

BRIGHT_MODE_STYLE = {
    "background": r".sld-frame {fill: var(--sld-background-color, #fff)}",
    "active_power": r".sld-feeder-info.sld-active-power {fill: #000}",
    "reactive_power": r".sld-feeder-info.sld-reactive-power {fill: #000}",
    "current": r".sld-feeder-info.sld-current {fill:#000}",
    "label": r".sld-label {stroke: none; fill: black; font: 8px serif}",
    "bus-legend-info": r".sld-bus-legend-info {font: 10px serif; fill: black}",
    "sld-open": r".sld-disconnector.sld-open {stroke-width: 1.5; stroke: var(--sld-vl-color, black); fill: white}",
}

DISCONNECTOR_STYLE = {"height": "8", "width": "8", "x": "0", "y": "0", "transform": "rotate(45.0,4.0,4.0)"}
BREAKER_STYLE = {"d": "M1,1 V19 H19 V1z"}
BUSBAR_SECTION_LABEL_POSITION = {
    "x": "-5",
    "y": "-7",
}
