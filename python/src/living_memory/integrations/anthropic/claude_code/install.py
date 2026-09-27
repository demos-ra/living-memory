"""What Claude Code is told: to record, to run the conversion, and the
context of what it holds."""

__all__ = ["HOST", "change", "context", "install", "question"]

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mtsv

from living_memory import _json, _json_schema, _rename
from living_memory.integrations.anthropic.claude_code import raw_api_bodies

# The host this integration installs into (install.mtsv › install.1).
HOST = "claude-code"
# The variable that records the raw API bodies, as a file in a folder
# (install.mtsv › capture.1).
_CAPTURE = "OTEL_LOG_RAW_API_BODIES"
# The recording's folder under the data directory: the application,
# then the path of the module that reads it (install.mtsv › folder.3).
_FOLDER = ("living-memory", "anthropic", "claude_code", "raw_api_bodies")
# The command that converts it and gives the context.
_COMMAND = "living-memory"
# The host's own word for adding context (install.mtsv › hooks.3).
_ADD_CONTEXT = f"--add-context={HOST}"
# Where Claude Code keeps its home-directory files, and the user
# settings file among them (install.mtsv › capture.2).
_CONFIG = "CLAUDE_CONFIG_DIR"
_SETTINGS = "settings.json"
# The user settings are written back in their own layout.
_INDENT = 2
# The system the command does not run on, having no POSIX file lock
# (install.mtsv › folder.2).
_WINDOWS = "win32"
# The events whose context the hooks give (install.mtsv › hooks.3).
_START = "SessionStart"
_TURN = "UserPromptSubmit"
# The source of a SessionStart after compaction (context.2).
_COMPACT = "compact"
# An input's name: its date, then '/', then its session (raw_api_bodies.mtsv
# › values.7).
_SEPARATOR = "/"
# The columns of a group of requests (raw_api_bodies.mtsv › map.1).
_GROUP = [
    "session_id",
    "query_source",
    "requests",
    "first index_line",
    "last index_line",
    "first timestamp",
    "last timestamp",
]

# What the command gives for one input's output: its sheet files, the
# records of its sheet of the input values, and how many a context has
# given, each read when asked for.
Look = Callable[[str], dict[str, Callable[[], Any]]]


def change() -> str:
    """Return the change install makes, stated before it is made.

    It names the folder created, the variable set and what that
    records, that the request and response files are removed once
    converted unless kept, the three hooks added, and that the recording
    begins with the next session (install.mtsv › install.2, capture.3).
    Raise OSError on Windows, where the command does not run (folder.2).
    """
    _refuse_windows()
    folder = _folder()
    return (
        f"living-memory --install={HOST} will:\n"
        f"  create {folder}, readable by you alone, where Claude Code writes"
        " each request and response;\n"
        f"  in {_settings()}, set {_CAPTURE}=file:{folder}, which records the"
        " full requests and responses, your prompts, tool details and tool"
        " content included, from the next session;\n"
        f"  add a Stop hook that converts that folder to MTSV, in {_store(folder)},"
        " after each response, then removes the request and response files it"
        " has converted, unless you keep them;\n"
        "  add a SessionStart and a UserPromptSubmit hook that give Claude a"
        " map of that MTSV."
    )


def question() -> str:
    """Return the question install asks before it is made.

    Whether to keep the request and response files once converted, No
    by default (install.mtsv › install.5).
    """
    return (
        "Keep Claude Code's request and response files (raw API bodies)"
        " after they are converted?"
    )


def install(options: list[str]) -> None:
    """Make the change that change states.

    options -- the options each hook runs living-memory with, as
        --keep-files where the files are kept

    The recording's folder is made, then the user settings are written,
    adding to what they hold, and replaced whole (install.mtsv ›
    install.3-5, folder.1). Raise OSError on Windows, where the command
    does not run (folder.2).
    """
    _refuse_windows()
    folder = _folder()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _settings()
    held = _json.decode(path.read_bytes()) if path.exists() else {}
    updated = _updated(held, folder, options)
    _rename.replace(path, _json.encode(updated, _INDENT) + b"\n")


def context(hook_input: bytes, output: Path, names: list[str], look: Look) -> tuple:
    """Return the context of a hook's event, and what it has given.

    hook_input -- the hook's input, a JSON text
    output -- the output's folder
    names -- the recording's inputs, as raw_api_bodies.inputs gives them
    look -- what the command gives for an input's output

    From SessionStart, the map of the output; from UserPromptSubmit,
    the session's requests not yet given; each as MTSV text, and for the
    session's input how many of its requests have now been given
    (install.mtsv › context.1-4).
    """
    event = _json.decode(hook_input)
    this = raw_api_bodies.input_of(names, event["session_id"])
    if event["hook_event_name"] == _START:
        sheets = _start(event, output, names, this, look)
    elif event["hook_event_name"] == _TURN and this is not None:
        sheets = _turn(look(this))
    else:
        sheets = []
    marks = {this: len(look(this)["held"]())} if this is not None else {}
    return (mtsv.dumps(sheets) if sheets else ""), marks


def _start(
    event: dict[str, Any], output: Path, names: list[str], this: str | None, look: Look
) -> list[dict[str, Any]]:
    # The store, its requests by date, the sessions of the session's date
    # grouped, and the session's own sheet files and their columns; after
    # compaction, the lineage of its latest request (context.1, context.2).
    held = {name: look(name)["held"]() for name in names}
    dates: dict[str, list[str]] = {}
    for name in names:
        dates.setdefault(name.split(_SEPARATOR)[0], []).append(name)
    day = (this or (names[-1] if names else "")).split(_SEPARATOR)[0]
    sheets = [
        _sheet("store", ["path", "inputs"], [[str(output), str(len(names))]]),
        _sheet(
            "raw_api_bodies by date",
            ["date", "sessions", "requests"],
            [
                [date, str(len(group)), str(sum(len(held[n]) for n in group))]
                for date, group in dates.items()
            ],
        ),
        _sheet(
            "raw_api_bodies by session_id, query_source",
            _GROUP,
            [row for n in dates.get(day, []) for row in raw_api_bodies.groups(held[n])],
        ),
    ]
    if this is None:
        return sheets
    files = look(this)["sheets"]()
    sheets.append(
        _sheet(
            "store by sheet",
            ["file", "sheet name", "records"],
            [[f["file"], f["name"], str(len(f["positions"]))] for f in files],
        )
    )
    sheets.append(
        _sheet(
            "store columns",
            ["file", "position", "field name"],
            [
                [f["file"], str(place), field]
                for f in files
                for place, field in enumerate(f["header"], 1)
            ],
        )
    )
    if event.get("source") == _COMPACT:
        lineage = raw_api_bodies.lineage(held[this])
        sheets.append(
            _sheet("raw_api_bodies not kept", _GROUP, raw_api_bodies.groups(lineage))
        )
    return sheets


def _turn(output: dict[str, Callable[[], Any]]) -> list[dict[str, Any]]:
    # The session's requests not yet given, as the store holds them, and
    # where their records are in each sheet file, counted from 1
    # (context.3).
    held = output["held"]()
    new = held[output["given"]() :]
    if not new:
        return []
    positions = {int(record["pointer"].split("/")[1]) for record in new}
    landed = []
    for f in output["sheets"]():
        places = [i for i, p in enumerate(f["positions"], 1) if p in positions]
        if places:
            landed.append(
                [
                    f["file"],
                    f["name"],
                    str(len(places)),
                    str(places[0]),
                    str(places[-1]),
                ]
            )
    return [
        _sheet("raw_api_bodies", list(new[0]), [list(r.values()) for r in new]),
        _sheet(
            "store by sheet",
            ["file", "sheet name", "records", "first", "last"],
            landed,
        ),
    ]


def _sheet(name: str, header: list[str], records: list[list[str]]) -> dict[str, Any]:
    return {"sheet name": name, "header": header, "records": records}


def _refuse_windows() -> None:
    # The command does not run on Windows, which has no POSIX file lock
    # (install.mtsv › folder.2).
    if sys.platform == _WINDOWS:
        raise OSError("the command does not run on Windows, which has no POSIX lock")


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
    # The user settings file, settings.json in CLAUDE_CONFIG_DIR where
    # it is set, else in ~/.claude (install.mtsv › capture.2).
    home = os.environ.get(_CONFIG, "")
    return (Path(home) if home else Path.home() / ".claude") / _SETTINGS


def _updated(
    settings: dict[str, Any], folder: Path, options: list[str]
) -> dict[str, Any]:
    # The capture variable set, and each hook's matcher group added
    # where the settings do not hold it (install.mtsv › capture.1,
    # install.3).
    env = {**settings.get("env", {}), _CAPTURE: f"file:{folder}"}
    hooks = dict(settings.get("hooks", {}))
    for event, group in _groups(folder, options).items():
        present = hooks.get(event, [])
        if not any(_json_schema.equal(group, each) for each in present):
            hooks[event] = [*present, group]
    return {**settings, "env": env, "hooks": hooks}


def _groups(folder: Path, options: list[str]) -> dict[str, dict[str, Any]]:
    # Each group has no matcher, so it activates on every occurrence:
    # Stop converts the recording in the background; SessionStart and
    # UserPromptSubmit give the context (install.mtsv › hooks.1-3).
    convert = {"type": "command", "command": _COMMAND, "args": [*options, str(folder)]}
    add = {
        "type": "command",
        "command": _COMMAND,
        "args": [*options, _ADD_CONTEXT, str(folder)],
    }
    return {
        "Stop": {"hooks": [{**convert, "async": True}]},
        _START: {"hooks": [add]},
        _TURN: {"hooks": [add]},
    }
