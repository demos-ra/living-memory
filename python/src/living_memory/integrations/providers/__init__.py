"""Which providers' products exist, each with modules of its own."""

__all__ = ["lookup", "lookup_plugin", "PLUGINS", "PROVIDERS"]

import importlib
from pathlib import Path
from types import ModuleType

# The file a product's directory holds, and the module that reads it:
# raw_api_bodies.mtsv › index.1.
PROVIDERS: dict[str, str] = {"index.jsonl": "anthropic.claude_code.raw_api_bodies"}

# A plugin's name, and the module that installs it:
# install.mtsv › hook.1.
PLUGINS: dict[str, str] = {"claude-code": "anthropic.claude_code.install"}


def lookup(directory: Path) -> ModuleType:
    """Return the module of the product whose directory this is.

    The module reads the directory with load(path), which returns MTSV
    sheets and raises a ValueError for a directory that does not
    conform. Raise LookupError for a directory of no provider.
    """
    for name, module in PROVIDERS.items():
        if (directory / name).is_file():
            return importlib.import_module(f"{__name__}.{module}")
    raise LookupError(f"no provider for {str(directory)!r}")


def lookup_plugin(name: str) -> ModuleType:
    """Return the module that installs a plugin.

    The module holds SETTINGS, the settings file under the home
    directory, and data_directory(environ, platform, home),
    changes(data, home), directories(data), index_file(data),
    commands(data) and settings(document, data), each computing one
    part of the installation and changing nothing. Raise LookupError
    for a name that is no plugin's.
    """
    if name not in PLUGINS:
        raise LookupError(f"no plugin {name!r}")
    return importlib.import_module(f"{__name__}.{PLUGINS[name]}")
