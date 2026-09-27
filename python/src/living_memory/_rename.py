"""How a file is replaced whole: written beside it, renamed onto it."""

__all__ = ["replace"]

import os
import tempfile
from pathlib import Path


def replace(path: Path, data: bytes) -> None:
    # A file is replaced whole, so a reader sees the old file or the new
    # one: the bytes are written to a new file in the same folder, then
    # renamed onto the name in one step.
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, written = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with open(handle, "wb") as fp:
        fp.write(data)
    os.replace(written, path)
