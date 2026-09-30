"""The integrations: each reads a source, or installs into a host.

An integration is a module of this package, named by its path within it,
as anthropic/claude_code/raw_api_bodies. One that reads a source names
the file extension it reads, EXTENSION, or the file a directory it reads
holds, FILE, and supplies schema(), the schema its module specification
gives, and values(path, held), the input values in the order its source
holds them, after those held: the records of the output's sheet of the
input values, each by its header's names, or None where the output holds
none. Where its source is several inputs, read line by line, it also
supplies lines(path, start), the source's lines after the first start of
them, each decoded; inputs(index, names), the names of the inputs those
lines give values to, each the folder of the output it is written to,
names being those the output holds; values takes the name and the
lines, values(path, name, held, index); read(index, held, start), how
many lines are read, held the records the outputs of those inputs hold;
and it may supply spent(path, name, held, index), the files of the
source its data bank has spent. One that installs into a host names the
host, HOST, and supplies change(), the change it makes stated,
question(), what it asks first, install(options), which makes it, the
options those it runs the command with, and context(hook_input, bank),
the host's context, composed of what the data bank communicates:
bank.names(), bank.filter(input, values, places) and bank.new(input,
places).

Functions:
reader -- return the integration that reads a file or a directory
named -- return the integration of a name that reads a source
installer -- return the integration that installs into a host
"""

__all__ = ["installer", "named", "reader"]

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType


def reader(path: Path) -> ModuleType:
    """Return the integration that reads a file or a directory.

    A file is read by the integration of its extension, and a directory
    by the integration of the file it holds. Raise LookupError where no
    integration reads it.
    """
    for module in _members():
        if path.is_dir():
            if hasattr(module, "FILE") and (path / module.FILE).is_file():
                return module
        elif getattr(module, "EXTENSION", None) == path.suffix:
            return module
    raise LookupError(f"no integration reads {str(path)!r}")


def named(name: str) -> ModuleType:
    """Return the integration of a name that reads a source.

    The name is the integration's path within this package, as
    anthropic/claude_code/raw_api_bodies. Raise LookupError where no
    integration of that name reads a source.
    """
    for module in _members():
        reads = hasattr(module, "FILE") or hasattr(module, "EXTENSION")
        if reads and _name(module) == name:
            return module
    raise LookupError(f"no integration named {name!r} reads a source")


def installer(host: str) -> ModuleType:
    """Return the integration that installs into a host.

    Raise LookupError where no integration installs into it.
    """
    for module in _members():
        if getattr(module, "HOST", None) == host:
            return module
    raise LookupError(f"no integration installs into {host!r}")


def _name(module: ModuleType) -> str:
    # A module's path within this package, '/' between its parts.
    return module.__name__.removeprefix(f"{__name__}.").replace(".", "/")


def _members() -> list[ModuleType]:
    # Every module of the package is found as the package holds it, so
    # that a new integration changes no existing module.
    return [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(__path__, f"{__name__}.")
    ]
