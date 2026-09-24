"""Which reader reads which input."""

__all__ = ["load", "lookup", "JSONL"]

import importlib
from types import ModuleType
from typing import Any, BinaryIO

from living_memory.integrations import providers

# JSON Lines, Conventions.
JSONL = ".jsonl"

_MODULES = {JSONL: "otlp_json"}


def load(suffix: str, fp: BinaryIO, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from a file in the format an extension names.

    Raise LookupError for an extension that names no format.
    """
    return lookup(suffix).load(fp)


def lookup(suffix: str) -> ModuleType:
    """Return the module that reads a file extension's format.

    Raise LookupError for an extension that names no format.
    """
    if suffix in _MODULES:
        return importlib.import_module(f"{__name__}.{_MODULES[suffix]}")
    return providers.lookup(suffix)
