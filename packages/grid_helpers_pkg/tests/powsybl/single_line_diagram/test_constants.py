# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from toop_engine_grid_helpers.powsybl.single_line_diagram.replace_convergence_style import BRIGHT_MODE_STYLE, DARK_MODE_STYLE


def test_dark_bright_mode_styles():
    for key in DARK_MODE_STYLE.keys():
        assert key in BRIGHT_MODE_STYLE, f"Key '{key}' not found in BRIGHT_MODE_STYLE"

    for key in BRIGHT_MODE_STYLE.keys():
        assert key in DARK_MODE_STYLE, f"Key '{key}' not found in DARK_MODE_STYLE"
