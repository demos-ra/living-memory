"""What Claude Code must be told: where to record, and the plugin."""

__all__ = [
    "SETTINGS",
    "changes",
    "commands",
    "data_directory",
    "directories",
    "index_file",
    "settings",
]

import json
from collections.abc import Mapping
from pathlib import Path

# install.mtsv › directory.1: the application's name.
_NAME = "living-memory"
# install.mtsv › directory.2: the path of the module that reads the raw
# API bodies; their MTSV file takes the command's output name.
_BODIES = Path("anthropic", "claude_code", "raw_api_bodies")
_MTSV = ".mtsv"
_INDEX = "index.jsonl"
# install.mtsv › plugin.1, hook.1.
_PLUGIN = "claude-code"
_MARKETPLACE = "demos-ra/living-memory"
_INSTALLED = _PLUGIN + "@" + _NAME
# install.mtsv › capture.1: the user settings file, under the home
# directory.
SETTINGS = Path(".claude", "settings.json")
_VARIABLE = "OTEL_LOG_RAW_API_BODIES"


def data_directory(environ: Mapping[str, str], system: str, home: Path) -> Path:
    """Return the data directory, from environment, platform, home.

    Raise LookupError for a system with no data directory held.
    """
    # install.mtsv › directory.1; XDG Base Directory, Basics, block 10.
    if system == "linux":
        base = environ.get("XDG_DATA_HOME", "")
        if not Path(base).is_absolute():
            return home / ".local" / "share" / _NAME
        return Path(base) / _NAME
    if system == "darwin":
        return home / "Library" / "Application Support" / _NAME
    raise LookupError(f"no data directory for {system!r}")


def directories(data: Path) -> list[Path]:
    """Return the directories to create, each inside the one before."""
    # install.mtsv › directory.2.
    found = [data]
    for part in _BODIES.parts:
        found.append(found[-1] / part)
    return found


def index_file(data: Path) -> Path:
    """Return the raw API bodies' index file in the data directory."""
    # install.mtsv › directory.2.
    return data / _BODIES / _INDEX


def changes(data: Path, home: Path) -> str:
    """Return the statement of each change an installation makes."""
    # install.mtsv › prompt.1, in the order of directory.2, plugin.1 and
    # capture.1.
    bodies = data / _BODIES
    lines = [
        f"This will create {bodies}, with an empty {_INDEX}, for the raw API"
        f" bodies, and write {bodies.with_suffix(_MTSV)} after each response.",
        *(" ".join(command) for command in commands(data)),
        f"This will set {_VARIABLE} to file:{bodies} in {home / SETTINGS}:"
        " Claude Code will then save every Messages API request and response,"
        " the whole conversation, to that directory.",
    ]
    return "\n".join(lines)


def commands(data: Path) -> list[list[str]]:
    """Return the commands adding the marketplace and the plugin."""
    # install.mtsv › plugin.1, hook.1.
    return [
        ["claude", "plugin", "marketplace", "add", _MARKETPLACE],
        ["claude", "plugin", "install", _INSTALLED, "--config", f"data_dir={data}"],
    ]


def settings(document: str | None, data: Path) -> str:
    """Return the user settings with the capture variable set.

    document is the file's text, None for a file that does not exist.
    Raise ValueError for a file that is not a JSON object.
    """
    # install.mtsv › capture.1: strict JSON, every other member kept.
    found = {} if document is None else json.loads(document, parse_constant=_refuse)
    if not isinstance(found, dict):
        raise ValueError(f"{SETTINGS} is not a JSON object")
    env = found.setdefault("env", {})
    if not isinstance(env, dict):
        raise ValueError(f"env in {SETTINGS} is not a JSON object")
    env[_VARIABLE] = f"file:{data / _BODIES}"
    return json.dumps(found, indent=2, ensure_ascii=False) + "\n"


def _refuse(name: str) -> None:
    # RFC 8259, Section 6: NaN and Infinity are not JSON.
    raise ValueError(f"{name} is not a JSON value")
