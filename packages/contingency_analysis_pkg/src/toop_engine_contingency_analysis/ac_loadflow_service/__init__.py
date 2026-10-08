# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from beartype.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results

__all__ = [
    "get_ac_loadflow_results",
]


def __getattr__(name: str) -> object:
    """Import get_ac_loadflow_results on first use, as it pulls in the pandapower contingency analysis and ray."""
    if name == "get_ac_loadflow_results":
        from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import (  # noqa: PLC0415
            get_ac_loadflow_results,
        )

        return get_ac_loadflow_results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
