"""Which providers exist, each read by a module of its own."""

__all__ = ["lookup", "PROVIDERS"]

import importlib
from pathlib import Path
from types import ModuleType

# The file a provider's directory holds, and the module that reads it:
# anthropic.mtsv › index.1.
PROVIDERS: dict[str, str] = {"index.jsonl": "anthropic"}


def lookup(directory: Path) -> ModuleType:
    """Return the module of the provider whose directory this is.

    Raise LookupError for a directory of no provider.
    """
    for name, module in PROVIDERS.items():
        if (directory / name).is_file():
            return importlib.import_module(f"{__name__}.{module}")
    raise LookupError(f"no provider for {str(directory)!r}")
