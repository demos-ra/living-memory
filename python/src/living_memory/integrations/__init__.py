"""Which reader reads which input.

Modules:
otlp_json -- read MTSV sheets from an OTLP JSON Lines file
providers -- the readers of providers' own formats

Functions:
load -- read MTSV sheets from a file in the format an extension names
lookup -- return the reader of a file extension

Constants:
JSONL -- the file extension of JSON Lines
"""

__all__ = ["load", "lookup", "JSONL"]

import importlib
from types import ModuleType
from typing import Any, BinaryIO

from living_memory.integrations import providers

# JSON Lines, Conventions: "JSON Lines files may be saved with the file
# extension .jsonl".
JSONL = ".jsonl"

_MODULES = {JSONL: "otlp_json"}


def load(suffix: str, fp: BinaryIO, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from a file in the format an extension names.

    suffix -- the file extension, such as ".jsonl"
    fp -- a binary file object

    Raise LookupError for an extension that names no format.
    """
    return lookup(suffix).load(fp)


def lookup(suffix: str) -> ModuleType:
    """Return the module that reads a file extension's format.

    suffix -- the file extension, such as ".jsonl"

    Raise LookupError for an extension that names no format.
    """
    if suffix in _MODULES:
        return importlib.import_module(f"{__name__}.{_MODULES[suffix]}")
    return providers.lookup(suffix)
