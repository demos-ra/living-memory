"""Which providers' products exist, each with modules of its own."""

__all__ = ["lookup", "lookup_plugin"]

import importlib
from pathlib import Path
from types import ModuleType

# A product's directory is known by the file it holds, and is read by
# its module (raw_api_bodies.mtsv › index.1).
_DIRECTORIES = {"index.jsonl": "anthropic.claude_code.raw_api_bodies"}

# A plugin is installed by its product's module (install.mtsv ›
# hook.1).
_PLUGINS = {"claude-code": "anthropic.claude_code.install"}


def lookup(directory: Path) -> ModuleType:
    """Return the module of the product whose directory this is.

    The module reads the directory with load(path), which returns MTSV
    sheets and raises a ValueError for a directory that does not
    conform. Raise LookupError for a directory of no provider.
    """
    for name, module in _DIRECTORIES.items():
        if (directory / name).is_file():
            return importlib.import_module(f"{__name__}.{module}")
    raise LookupError(f"no provider for {str(directory)!r}")


def lookup_plugin(name: str) -> ModuleType:
    """Return the module that installs a plugin.

    The module's plan(environ, platform, home) returns what an
    installation changes and the steps that make the changes, computing
    both and changing nothing. Raise LookupError for a name that is no
    plugin's.
    """
    if name not in _PLUGINS:
        raise LookupError(f"no plugin {name!r}")
    return importlib.import_module(f"{__name__}.{_PLUGINS[name]}")
