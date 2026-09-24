"""Which provider formats exist, each read by a module of its own."""

__all__ = ["lookup", "PROVIDERS"]

import importlib
from types import ModuleType

PROVIDERS: dict[str, str] = {}


def lookup(suffix: str) -> ModuleType:
    """Return the module that reads a provider's format.

    Raise LookupError for an extension that names no provider format.
    """
    if suffix not in PROVIDERS:
        raise LookupError(f"no format for {suffix!r}")
    return importlib.import_module(f"{__name__}.{PROVIDERS[suffix]}")
