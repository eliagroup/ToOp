import sys

from pandera import Int

if sys.platform == "win32":
    Int.check = lambda self, pandera_dtype, data_container=None: isinstance(pandera_dtype, Int)  # noqa: ARG005
