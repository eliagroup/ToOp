import sys

from beartype.claw import beartype_this_package
from pandera import Int

beartype_this_package()

if sys.platform == "win32":
    Int.check = lambda self, pandera_dtype, data_container=None: isinstance(pandera_dtype, Int)  # noqa: ARG005
