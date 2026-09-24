"""Which reader reads which input."""

__all__ = ["load", "lookup", "lookup_directory", "lookup_plugin", "JSONL"]

import importlib
from pathlib import Path
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
    if suffix not in _MODULES:
        raise LookupError(f"no format for {suffix!r}")
    return importlib.import_module(f"{__name__}.{_MODULES[suffix]}")


def lookup_directory(directory: Path) -> ModuleType:
    """Return the module of the product whose directory this is.

    Raise LookupError for a directory of no provider.
    """
    return providers.lookup(directory)


def lookup_plugin(name: str) -> ModuleType:
    """Return the module that installs a plugin.

    Raise LookupError for a name that is no plugin's.
    """
    return providers.lookup_plugin(name)
