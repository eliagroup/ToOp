# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines an abstract runner class.

An abstract runner class encapsulates a loadflow solver and a postprocessing pipeline.
There are currently implementations for pandapower and powsybl as runners.
This module helps with postprocessing of optimizer results.
"""

from importlib import import_module
from importlib.util import find_spec

from .abstract_runner import AbstractLoadflowRunner
from .postprocess_powsybl import PowsyblRunner


def _is_missing_pandapower_dependency(exc: ModuleNotFoundError) -> bool:
    """Return whether the import failed because pandapower is not installed."""
    return (exc.name == "pandapower" or (exc.name is not None and exc.name.startswith("pandapower."))) or (
        "pandapower" in str(exc)
    )


def _has_pandapower_dependency() -> bool:
    """Return whether pandapower can be resolved without importing optional exports."""
    try:
        return find_spec("pandapower") is not None
    except ModuleNotFoundError as exc:
        if not _is_missing_pandapower_dependency(exc):
            raise
        return False


__all__ = ["AbstractLoadflowRunner", "PowsyblRunner"]

if _has_pandapower_dependency():
    PandapowerRunner = import_module(".postprocess_pandapower", package=__name__).PandapowerRunner
    __all__.append("PandapowerRunner")
