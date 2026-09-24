"""What Claude Code must be told: where to record, and the plugin."""

__all__ = ["plan"]

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from living_memory._json import decode_strict

# The data directory takes the application's name
# (install.mtsv › directory.1).
_NAME = "living-memory"
# The raw API bodies take the path of the module that reads them, their
# MTSV file the command's output name, and the directories are private
# (install.mtsv › directory.2; XDG-BASEDIR, Referencing this
# specification).
_BODIES = Path("anthropic", "claude_code", "raw_api_bodies")
_MTSV = ".mtsv"
_INDEX = "index.jsonl"
_PRIVATE = 0o700
# The plugin is installed from this marketplace
# (install.mtsv › plugin.1, hook.1).
_PLUGIN = "claude-code"
_MARKETPLACE = "demos-ra/living-memory"
_INSTALLED = _PLUGIN + "@" + _NAME
# The capture is turned on by this variable, in the user settings file
# under the home directory (install.mtsv › capture.1).
_SETTINGS = Path(".claude", "settings.json")
_VARIABLE = "OTEL_LOG_RAW_API_BODIES"

_Step = tuple[str, Any, Any]


def plan(
    environ: Mapping[str, str], system: str, home: Path
) -> tuple[str, list[_Step]]:
    """Return what an installation changes, and the steps that do it.

    Each step is ("directory", path, mode), ("file", path, None),
    ("run", arguments, None) or ("replace", path, rewrite), rewrite
    taking the file's text, None for no file, and returning its new
    text. Raise LookupError for a system with no data directory held.
    """
    # The steps run in the order of directory.2, plugin.1 and capture.1,
    # the capture last (install.mtsv › conformance.2).
    data = _data_directory(environ, system, home)
    settings_file = home / _SETTINGS
    steps: list[_Step] = [
        ("directory", directory, _PRIVATE) for directory in _directories(data)
    ]
    steps.append(("file", data / _BODIES / _INDEX, None))
    steps += [("run", command, None) for command in _commands(data)]
    steps.append(("directory", settings_file.parent, None))
    steps.append(("replace", settings_file, _rewrite(data)))
    return _changes(data, settings_file), steps


def _data_directory(environ: Mapping[str, str], system: str, home: Path) -> Path:
    # On Linux, $XDG_DATA_HOME if it is an absolute path, else
    # ~/.local/share; on macOS, ~/Library/Application Support; no other
    # system's data directory is held (install.mtsv › directory.1;
    # XDG-BASEDIR, Basics).
    if system == "linux":
        base = environ.get("XDG_DATA_HOME", "")
        if not Path(base).is_absolute():
            return home / ".local" / "share" / _NAME
        return Path(base) / _NAME
    if system == "darwin":
        return home / "Library" / "Application Support" / _NAME
    raise LookupError(f"no data directory for {system!r}")


def _directories(data: Path) -> list[Path]:
    # The directories are created each inside the one before
    # (install.mtsv › directory.2).
    found = [data]
    for part in _BODIES.parts:
        found.append(found[-1] / part)
    return found


def _changes(data: Path, settings_file: Path) -> str:
    # Each change is stated before it is made (install.mtsv › prompt.1).
    bodies = data / _BODIES
    lines = [
        f"This will create {bodies}, with an empty {_INDEX}, for the raw API"
        f" bodies, and write {bodies.with_suffix(_MTSV)} after each response.",
        *(" ".join(command) for command in _commands(data)),
        f"This will set {_VARIABLE} to file:{bodies} in {settings_file}:"
        " Claude Code will then save every Messages API request and response,"
        " the whole conversation, to that directory.",
    ]
    return "\n".join(lines)


def _commands(data: Path) -> list[list[str]]:
    # The marketplace is added, then the plugin installed with the data
    # directory as its option (install.mtsv › plugin.1, hook.1).
    return [
        ["claude", "plugin", "marketplace", "add", _MARKETPLACE],
        ["claude", "plugin", "install", _INSTALLED, "--config", f"data_dir={data}"],
    ]


def _rewrite(data: Path) -> Callable[[str | None], str]:
    return lambda document: _settings(document, data)


def _settings(document: str | None, data: Path) -> str:
    # The settings are strict JSON; the variable is added or replaced
    # under env, and every other member kept (install.mtsv › capture.1).
    found = {} if document is None else decode_strict(document)
    if not isinstance(found, dict):
        raise ValueError(f"{_SETTINGS} is not a JSON object")
    env = found.setdefault("env", {})
    if not isinstance(env, dict):
        raise ValueError(f"env in {_SETTINGS} is not a JSON object")
    env[_VARIABLE] = f"file:{data / _BODIES}"
    return json.dumps(found, indent=2, ensure_ascii=False) + "\n"
