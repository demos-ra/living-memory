"""What Claude Code is told: to record, and to run the conversion."""

__all__ = ["HOST", "change", "install"]

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from living_memory import _json, _json_schema

# The host this integration installs into (install.mtsv › install.1).
HOST = "claude-code"
# The variable that records the raw API bodies, as a file in a folder
# (install.mtsv › capture.1).
_CAPTURE = "OTEL_LOG_RAW_API_BODIES"
# The recording's folder under the data directory: the application,
# then the path of the module that reads it (install.mtsv › folder.2).
_FOLDER = ("living-memory", "anthropic", "claude_code", "raw_api_bodies")
# The command that converts it, and the program that states where
# it is.
_COMMAND = "living-memory"
_ECHO = "echo"
# The user settings are written back in their own layout.
_INDENT = 2


def change() -> str:
    # The change, stated before it is made (install.mtsv › install.1).
    folder = _folder()
    return (
        f"living-memory --install={HOST} will:\n"
        f"  create {folder}, readable by you alone, where Claude Code writes"
        " each request and response;\n"
        f"  in {_settings()}, set {_CAPTURE}=file:{folder}, which records the"
        " full requests and responses, your prompts, tool details and tool"
        " content included;\n"
        f"  add a Stop hook that converts that folder to MTSV, in {_store(folder)},"
        " after each response;\n"
        "  add a SessionStart hook that states where that MTSV is."
    )


def install() -> None:
    # The folder is made, then the settings are written, adding to what
    # they hold, beside their name and renamed onto it (install.mtsv ›
    # folder.1, install.2).
    folder = _folder()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _settings()
    held = _json.decode(path.read_bytes()) if path.exists() else {}
    _replace(path, _json.encode(_updated(held, folder), _INDENT) + b"\n")


def _data_home() -> Path:
    # On macOS, ~/Library/Application Support; else $XDG_DATA_HOME where
    # it is an absolute path, and ~/.local/share where it is unset,
    # empty or relative (install.mtsv › folder.1).
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    given = os.environ.get("XDG_DATA_HOME", "")
    if given and Path(given).is_absolute():
        return Path(given)
    return Path.home() / ".local" / "share"


def _folder() -> Path:
    return _data_home().joinpath(*_FOLDER)


def _store(folder: Path) -> Path:
    # The command's output: the input's name with .mtsv, beside it.
    return folder.with_suffix(".mtsv")


def _settings() -> Path:
    # The user settings file, for every project (install.mtsv ›
    # capture.2).
    return Path.home() / ".claude" / "settings.json"


def _updated(settings: dict[str, Any], folder: Path) -> dict[str, Any]:
    # The capture variable set, and each hook's matcher group added
    # where the settings do not hold it (install.mtsv › capture.1,
    # install.2).
    env = {**settings.get("env", {}), _CAPTURE: f"file:{folder}"}
    hooks = dict(settings.get("hooks", {}))
    for event, group in _groups(folder).items():
        present = hooks.get(event, [])
        if not any(_json_schema.equal(group, each) for each in present):
            hooks[event] = [*present, group]
    return {**settings, "env": env, "hooks": hooks}


def _groups(folder: Path) -> dict[str, dict[str, Any]]:
    # Each group has no matcher, so it activates on every occurrence:
    # Stop converts the recording in the background; SessionStart
    # states, as a fact, where the output is (install.mtsv › hooks.1,
    # hooks.2).
    store = _store(folder)
    statement = (
        f"Claude Code's conversations are kept as MTSV in {store}, a folder of"
        " one file per sheet, the files joined in the order their names give"
        " being the MTSV file; each conversation's rows carry its session_id."
    )
    convert = {"type": "command", "command": _COMMAND, "args": [str(folder)]}
    state = {"type": "command", "command": _ECHO, "args": [statement]}
    return {
        "Stop": {"hooks": [{**convert, "async": True}]},
        "SessionStart": {"hooks": [state]},
    }


def _replace(path: Path, data: bytes) -> None:
    # A file is replaced whole: written beside its name and renamed onto
    # it (POSIX.1-2017, rename).
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, written = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with open(handle, "wb") as fp:
        fp.write(data)
    os.replace(written, path)
