# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Plot the artificial limit of the example branches in docs/importer/pypowsybl/artificial_limits.md.

The limits are computed with compute_artificial_limits, so the figure follows the implementation. Run it from the
repository root after changing the limit computation or the example table:

    uv run python scripts/plot_artificial_limits.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from toop_engine_importer.pypowsybl_import.artificial_limits import compute_artificial_limits
from toop_engine_interfaces.messages.preprocess.preprocess_commands import DoubleLimits
from toop_engine_interfaces.nminus1_definition import LoadingConstraint

OUTPUT_PATH = Path(__file__).parent.parent / "docs" / "importer" / "pypowsybl" / "images" / "artificial_limits.png"

PHYSICAL_LIMIT = 1000.0
DOUBLE_LIMITS = DoubleLimits(lower=0.8, upper=0.9, minimum_hold_limit=0.4, hold_margin=0.01)

# The branches of the example table: name, loading constraint and worst current in A
BRANCHES = [
    ("A", LoadingConstraint.RELIEVE, 500.0),
    ("B", LoadingConstraint.RELIEVE, 850.0),
    ("C", LoadingConstraint.RELIEVE, 1100.0),
    ("D", LoadingConstraint.RELIEVE_HOLD, 1100.0),
    ("E", LoadingConstraint.RELIEVE_HOLD, 700.0),
    ("F", LoadingConstraint.RELIEVE_HOLD, 200.0),
    ("G", LoadingConstraint.HOLD, 1800.0),
    ("H", LoadingConstraint.HOLD, 500.0),
    ("I", LoadingConstraint.HOLD, 200.0),
    ("J", LoadingConstraint.CAP, 500.0),
    ("K", LoadingConstraint.CAP, 850.0),
    ("L", LoadingConstraint.CAP, 1100.0),
    ("M", LoadingConstraint.UNCONSTRAINED, 700.0),
]

REFERENCE_LINES = [
    ("physical limit", PHYSICAL_LIMIT, "#4d4d4d"),
    ("upper limit", PHYSICAL_LIMIT * DOUBLE_LIMITS.upper, "#e8483a"),
    ("lower limit", PHYSICAL_LIMIT * DOUBLE_LIMITS.lower, "#e0d800"),
    ("minimum hold limit", PHYSICAL_LIMIT * DOUBLE_LIMITS.minimum_hold_limit, "#1f77b4"),
]

ALLOWED_COLOR = "#cfece4"
OVERLOAD_COLOR = "#e8d0ee"
NO_LIMIT_COLOR = "#dadada"
BAR_WIDTH = 0.5
Y_MAX = 2000.0


def plot_artificial_limits() -> plt.Figure:
    """Plot the allowed and the overloaded loading range of every example branch.

    Returns
    -------
    plt.Figure
        The figure with one bar per example branch, split at its artificial limit.
    """
    names, loading_constraints, worst_currents = zip(*BRANCHES, strict=True)
    limits = compute_artificial_limits(
        flow=pd.Series(worst_currents),
        limit=pd.Series([PHYSICAL_LIMIT] * len(BRANCHES)),
        loading_constraint=pd.Series(loading_constraints),
        double_limits=DOUBLE_LIMITS,
    )

    fig, ax = plt.subplots(figsize=(18, 9.7), dpi=100)
    positions = range(len(BRANCHES))
    for position, limit, worst_current in zip(positions, limits, worst_currents, strict=True):
        if pd.isna(limit):
            ax.bar(position, Y_MAX, width=BAR_WIDTH, color=NO_LIMIT_COLOR)
        else:
            ax.bar(position, limit, width=BAR_WIDTH, color=ALLOWED_COLOR)
            ax.bar(position, Y_MAX - limit, bottom=limit, width=BAR_WIDTH, color=OVERLOAD_COLOR)
        ax.scatter(position, worst_current, marker="D", s=110, facecolor="white", edgecolor="#2b1a33", zorder=3)

    line_end = len(BRANCHES) - 0.5
    for label, value, color in REFERENCE_LINES:
        ax.hlines(value, -0.6, line_end, colors=color, linestyles=(0, (2, 1)), linewidth=2.2, zorder=2)
        ax.text(line_end + 0.1, value, label, va="center", fontsize=14)

    ax.set_xticks(
        list(positions),
        [f"{name}\n{constraint.value}" for name, constraint in zip(names, loading_constraints, strict=True)],
        fontsize=10.5,
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlim(-0.6, len(BRANCHES) + 3.0)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel("branches (loading constraint)", fontsize=13)
    ax.set_ylabel("Current [A]", fontsize=13)
    ax.set_title("Artificial limit by loading constraint", fontsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(color=ALLOWED_COLOR, label="allowed, up to the artificial limit"),
            Patch(color=OVERLOAD_COLOR, label="counts as overload"),
            Patch(color=NO_LIMIT_COLOR, label="no artificial limit"),
            plt.Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                markersize=10,
                markerfacecolor="white",
                markeredgecolor="#2b1a33",
                label="worst current",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(line_end + 0.05, Y_MAX),
        bbox_transform=ax.transData,
        fontsize=12,
        frameon=False,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    plot_artificial_limits().savefig(OUTPUT_PATH)
