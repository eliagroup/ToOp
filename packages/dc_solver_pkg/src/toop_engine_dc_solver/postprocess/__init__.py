"""Defines an abstract runner class.

An abstract runner class encapsulates a loadflow solver and a postprocessing pipeline.
There are currently implementations for pandapower and powsybl as runners.
This module helps with postprocessing of optimizer results.
"""

from beartype.claw import beartype_this_package

# Make sure beartype_this_package is the only imported module
beartype_only_non_dunder_import = all(d.startswith("_") or d == "beartype_this_package" for d in dir())
if beartype_only_non_dunder_import:
    beartype_this_package()  # Leave this at the top. Otherwise the modules imported before wont be beartyped
else:
    raise ImportError(
        "Please make sure that beartype_this_package is the only imported module before calling beartype_this_package"
        "Please check the import statements."
    )

from .abstract_runner import (
    AbstractLoadflowRunner,
)
from .postprocess_pandapower import (
    PandapowerRunner,
)
from .postprocess_powsybl import (
    PowsyblRunner,
)

__all__ = ["AbstractLoadflowRunner", "PandapowerRunner", "PowsyblRunner"]
