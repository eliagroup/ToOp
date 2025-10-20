from toop_engine_grid_helpers.powsybl.single_line_diagram.replace_convergence_style import BRIGHT_MODE_STYLE, DARK_MODE_STYLE


def test_dark_bright_mode_styles():
    for key in DARK_MODE_STYLE.keys():
        assert key in BRIGHT_MODE_STYLE, f"Key '{key}' not found in BRIGHT_MODE_STYLE"

    for key in BRIGHT_MODE_STYLE.keys():
        assert key in DARK_MODE_STYLE, f"Key '{key}' not found in DARK_MODE_STYLE"
