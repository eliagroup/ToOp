# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""A reference to a stored loadflow, in a separate file because many locations started to reference it.

Also, we might add helper functions in the future like load/save to azure bucket.
"""

from pydantic import BaseModel


class StoredLoadflowReference(BaseModel):
    """A reference to a stored loadflow result on disk or in an object store.

    Loadflow results are too large to be sent directly over kafka, so they need to be stored somewhere and referenced.
    They are stored and written using the functions in `loadflow_result_helpers_new.py`, which use the `fsspec` library
    to abstract away the filesystem. Hence, these can write to local disk, Azure bucket, ...

    The reference contains the filename relative to the base path or bucket defined in the filesystem, i.e. if a
    DirFileSystem is used with base_path="/path/to/base" and the filename is "loadflows" then the full path is
    "/path/to/base/loadflows/node_results.parquet", "/path/to/base/loadflows/branch_results.parquet", ...
    """

    relative_path: str
    """The folder of the loadflow result relative to the base path or bucket in the filesystem. This points to a directory
    under which the files "node_results.parquet", "branch_results.parquet", "metadata.json", ... are stored.
    """
