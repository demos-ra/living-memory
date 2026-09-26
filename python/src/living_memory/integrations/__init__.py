"""The integrations: each reads a source, or installs into a host.

An integration is a module of this package. One that reads a source
names the file extension it reads, EXTENSION, or the file a directory it
reads holds, FILE, and supplies schema(), the schema its module
specification gives, and values(path, held), the input values in the
order its source holds them, after those held: the records of the
output's sheet of the input values, each by its header's names, or None
where the output holds none. One that installs into a host names the
host, HOST, and supplies change(), the change it makes stated, and
install(), which makes it.

Functions:
reader -- return the integration that reads a file or a directory
installer -- return the integration that installs into a host
"""

__all__ = ["installer", "reader"]

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


def installer(host: str) -> ModuleType:
    """Return the integration that installs into a host.

    Raise LookupError where no integration installs into it.
    """
    for module in _members():
        if getattr(module, "HOST", None) == host:
            return module
    raise LookupError(f"no integration installs into {host!r}")


def _members() -> list[ModuleType]:
    # Every module of the package is found as the package holds it, so
    # that a new integration changes no existing module.
    return [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(__path__, f"{__name__}.")
    ]
