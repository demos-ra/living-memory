"""Which provider formats exist.

A provider's own format is read by a module of its own, which converts
it to the same sheets. None exists yet.

Functions:
lookup -- return the module that reads a provider's format

Constants:
PROVIDERS -- each provider format's file extension, and its module
"""

__all__ = ["lookup", "PROVIDERS"]

import importlib
from types import ModuleType

PROVIDERS: dict[str, str] = {}


def lookup(suffix: str) -> ModuleType:
    """Return the module that reads a provider's format.

    suffix -- the format's file extension

    Raise LookupError for an extension that names no provider format.
    """
    if suffix not in PROVIDERS:
        raise LookupError(f"no format for {suffix!r}")
    return importlib.import_module(f"{__name__}.{PROVIDERS[suffix]}")
