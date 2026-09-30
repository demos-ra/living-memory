"""What Claude Code is told: to record, to run the conversion, and the
context of what the data bank communicates."""

__all__ = ["HOST", "change", "context", "groups", "install", "lineage", "question"]

import os
import sys
from pathlib import Path
from typing import Any

import mtsv

from living_memory import _json, _rename
from living_memory.integrations.anthropic.claude_code import raw_api_bodies

# The host this integration installs into (install.mtsv › install.1).
HOST = "claude-code"
# The variable that records the raw API bodies, as a file in a folder
# (install.mtsv › capture.1).
_CAPTURE = "OTEL_LOG_RAW_API_BODIES"
# The recording's folder under the data directory: the application,
# then the path of the module that reads it (install.mtsv › folder.3).
_FOLDER = ("living-memory", "anthropic", "claude_code", "raw_api_bodies")
# The reader of the recording, the path of the module that reads it
# (install.mtsv › hooks.5, folder.3).
_READER = "/".join(_FOLDER[1:])
# The command that converts it and gives the context.
_COMMAND = "living-memory"
# The host's own word for adding context (install.mtsv › hooks.3).
_ADD_CONTEXT = f"--add-context={HOST}"
_FROM = f"--from={_READER}"
# Added context is capped at 10,000 characters (install.mtsv ›
# context.5).
_CAP = 10_000
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
# An input's name: its date, then '/', then its session
# (raw_api_bodies.mtsv › values.7).
_SEPARATOR = "/"
# The place of the sheet of the input values (living-memory.mtsv ›
# order.1).
_VALUES = 0
# The columns of a group of requests, a role name qualifying each domain
# name (install.mtsv › context.6).
_GROUP = [
    "session_id",
    "query_source",
    "requests",
    "first.index_line",
    "last.index_line",
    "first.timestamp",
    "last.timestamp",
]


def change() -> str:
    """Return the change install makes, stated before it is made.

    It names the folder created, the variable set and what that
    records, that the request and response files are removed once
    stored unless kept, the three hooks added, and that the recording
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
        "  add a Stop hook that stores that folder as MTSV in a data bank"
        " beside it after each response, then removes the request and"
        " response files it has stored, unless you keep them;\n"
        "  add a SessionStart and a UserPromptSubmit hook that give Claude"
        " what the data bank communicates."
    )


def question() -> str:
    """Return the question install asks before it is made.

    Whether to keep the request and response files once stored, No by
    default (install.mtsv › install.5).
    """
    return (
        "Keep Claude Code's request and response files (raw API bodies)"
        " after they are stored?"
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


def context(hook_input: bytes, bank: Any) -> str:
    """Return the context of a hook's event, as MTSV text.

    hook_input -- the hook's input, a JSON text
    bank -- the data bank, which gives only its communications: names(),
        filter(input, values, places) and new(input, places)

    From SessionStart, the data bank, the requests by date and grouped,
    and the session's sheets and fields; from UserPromptSubmit, what is
    new of the session's requests, grouped where it would pass the cap
    (install.mtsv › context.1-7).
    """
    event = _json.decode(hook_input)
    names = {s["sheet name"]: s for s in mtsv.loads(bank.names())}
    inputs = [record[0] for record in names["inputs"]["records"]]
    this = raw_api_bodies.input_of(inputs, event["session_id"])
    if event["hook_event_name"] == _START:
        sheets = _start(event, bank, names, this)
    elif event["hook_event_name"] == _TURN and this is not None:
        sheets = _turn(bank, this)
    else:
        sheets = []
    return mtsv.dumps(sheets) if sheets else ""


def _start(
    event: dict[str, Any], bank: Any, names: dict[str, Any], this: str | None
) -> list[dict[str, Any]]:
    # The data bank, the requests by date, the requests of the session's
    # date grouped, and the session's sheets and fields; after
    # compaction, the lineage of its latest request; and what is new
    # begins after the last value stored (context.1, context.2,
    # context.4).
    counts = [(record[0], int(record[1])) for record in names["inputs"]["records"]]
    dates: dict[str, list[tuple[str, int]]] = {}
    for name, values in counts:
        dates.setdefault(name.split(_SEPARATOR)[0], []).append((name, values))
    day = (this or (counts[-1][0] if counts else "")).split(_SEPARATOR)[0]
    today = [
        row for name, _ in dates.get(day, []) for row in groups(_values(bank, name))
    ]
    sheets = [
        _sheet("data bank", ["source", "reader"], [[str(_folder()), _READER]]),
        _sheet(
            "raw_api_bodies by date",
            ["date", "sessions", "requests"],
            [
                [date, str(len(group)), str(sum(values for _, values in group))]
                for date, group in dates.items()
            ],
        ),
        _sheet("raw_api_bodies by session_id, query_source", _GROUP, today),
    ]
    if this is None:
        return sheets
    for sheet_name in ("sheets", "fields"):
        own = [r for r in names[sheet_name]["records"] if r[0] == this]
        sheets.append({**names[sheet_name], "records": own})
    if event.get("source") == _COMPACT:
        chain = lineage(_values(bank, this))
        sheets.append(_sheet("raw_api_bodies not kept", _GROUP, groups(chain)))
    bank.new(this, [])
    return sheets


def _turn(bank: Any, this: str) -> list[dict[str, Any]]:
    # What is new of the session's requests, of the sheet of the input
    # values alone, or its groups where it would pass the cap
    # (context.3, context.5).
    new = mtsv.loads(bank.new(this, [_VALUES]))
    if not new:
        return []
    if len(mtsv.dumps(new)) <= _CAP:
        return new
    records = [dict(zip(new[0]["header"], r)) for r in new[0]["records"]]
    return [
        _sheet("raw_api_bodies by session_id, query_source", _GROUP, groups(records))
    ]


def groups(records: list[dict[str, str]]) -> list[list[str]]:
    """Return requests grouped by session_id and query_source.

    Each group, in the order of its first index line: session_id,
    query_source, how many requests, and the first and last index_line
    and timestamp (install.mtsv › context.6).
    """
    found: dict[tuple[str, str], list[dict[str, str]]] = {}
    for record in records:
        key = (record["session_id"], record["query_source"])
        found.setdefault(key, []).append(record)
    return [
        [
            *key,
            str(len(group)),
            group[0]["index_line"],
            group[-1]["index_line"],
            group[0]["timestamp"],
            group[-1]["timestamp"],
        ]
        for key, group in found.items()
    ]


def lineage(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the lineage of an input's latest request, earliest first.

    That request, then the request it extends, and so on until a request
    that extends none (install.mtsv › context.7).
    """
    by_line = {record["index_line"]: record for record in records}
    found = []
    line = records[-1]["index_line"] if records else "0"
    while line in by_line:
        found.append(by_line[line])
        line = by_line[line]["extends"]
    return found[::-1]


def _values(bank: Any, name: str) -> list[dict[str, str]]:
    # An input's records of the sheet of the input values, as the data
    # bank communicates them, each by its header's names.
    found = mtsv.loads(bank.filter(name, None, [_VALUES]))
    return [dict(zip(s["header"], r)) for s in found for r in s["records"]]


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


def _settings() -> Path:
    # The user settings file, settings.json in CLAUDE_CONFIG_DIR where
    # it is set, else in ~/.claude (install.mtsv › capture.2).
    home = os.environ.get(_CONFIG, "")
    return (Path(home) if home else Path.home() / ".claude") / _SETTINGS


def _updated(
    settings: dict[str, Any], folder: Path, options: list[str]
) -> dict[str, Any]:
    # The capture variable set, and each hook's matcher group added, or
    # put where the first group it added before stands, the others it
    # added left out (install.mtsv › capture.1, install.3).
    env = {**settings.get("env", {}), _CAPTURE: f"file:{folder}"}
    hooks = dict(settings.get("hooks", {}))
    for event, group in _groups(folder, options).items():
        present = hooks.get(event, [])
        own = [_own(each, folder) for each in present]
        others = [each for each, mine in zip(present, own) if not mine]
        at = own.index(True) if True in own else len(others)
        hooks[event] = [*others[:at], group, *others[at:]]
    return {**settings, "env": env, "hooks": hooks}


def _own(group: dict[str, Any], folder: Path) -> bool:
    # A group install added: one whose command hook runs living-memory
    # on the recording's folder (install.mtsv › install.3).
    return any(
        hook.get("command") == _COMMAND and hook.get("args", [])[-1:] == [str(folder)]
        for hook in group.get("hooks", [])
    )


def _groups(folder: Path, options: list[str]) -> dict[str, dict[str, Any]]:
    # Each group has no matcher, so it activates on every occurrence:
    # Stop converts the recording in the background; SessionStart and
    # UserPromptSubmit give the context; each names its reader
    # (install.mtsv › hooks.1-3, hooks.5).
    convert = {
        "type": "command",
        "command": _COMMAND,
        "args": [*options, _FROM, str(folder)],
    }
    add = {
        "type": "command",
        "command": _COMMAND,
        "args": [*options, _FROM, _ADD_CONTEXT, str(folder)],
    }
    return {
        "Stop": {"hooks": [{**convert, "async": True}]},
        _START: {"hooks": [add]},
        _TURN: {"hooks": [add]},
    }
