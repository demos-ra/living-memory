"""Claude Code's recording of the raw API bodies, as input values."""

__all__ = ["FILE", "schema", "values"]

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from living_memory import _json, _json_pointer, _json_schema
from living_memory._json import Number
from living_memory.integrations.anthropic import messages

# In file mode each successful response appends one line to
# <dir>/index.jsonl (raw_api_bodies.mtsv › recording.1).
FILE = "index.jsonl"
# The sheet of the input values is named after the recording.
_TITLE = "raw_api_bodies"
# The request's members that hold a list of units, each compared with
# the parent's at its pointer, and the response's content, the message
# that follows (raw_api_bodies.mtsv › values.1, values.3).
_LISTS = ("system", "tools")
_RESPONSE = "content"
# A value's names for its index line and for the index line of the
# request it extends, names the living-memory specification's Fields do
# not give (raw_api_bodies.mtsv › values.2, values.4).
_LINE = "index_line"
_EXTENDS = "extends"
# The index line's members that name the request (values.4).
_INDEX = ("session_id", "query_source", "model", "timestamp")
# Index lines count from 1, so 0 names no parent (JSON Lines;
# raw_api_bodies.mtsv › values.2).
_NO_PARENT = 0


@dataclass(frozen=True)
class _Request:
    # A request as it is compared: its index line and thread, its
    # messages as they are kept, then its response as the message of its
    # role and content that follows, and its system blocks and tools,
    # each as it is kept.
    line: int
    thread: tuple[str, str]
    sequence: list[Any]
    lists: dict[str, list[Any]]


def schema() -> bytes:
    """Return the schema of the input values, as a JSON text.

    A value names its request by index_line, session_id, query_source,
    model and timestamp; names as extends the request it extends and as
    kept how many of its messages it keeps; counts its system blocks and
    tools; and holds the units the request it extends does not, each at
    its pointer in the request, and the response's members kept beside
    its content (raw_api_bodies.mtsv › values.2-4).
    """
    count = {"type": "integer", "minimum": Number("0")}
    pointer = {
        "type": "object",
        "properties": {"pointer": {"type": "string"}},
        "required": ["pointer"],
        "additionalProperties": False,
    }
    unit = {
        "type": "object",
        "properties": {"request": pointer, **messages.units()},
        "required": ["request"],
        "additionalProperties": False,
    }
    counts = {
        "type": "object",
        "properties": {name: count for name in _LISTS},
        "required": list(_LISTS),
        "additionalProperties": False,
    }
    root = {
        "title": _TITLE,
        "type": "object",
        "properties": {
            _LINE: {"type": "integer", "minimum": Number("1")},
            **{name: {"type": "string"} for name in _INDEX},
            _EXTENDS: count,
            "kept": count,
            "count": counts,
            "units": {"type": "array", "items": unit},
            **messages.ending(),
        },
        "required": [_LINE, *_INDEX, _EXTENDS, "kept", "count", "units"],
        "additionalProperties": False,
        "definitions": messages.definitions(),
    }
    return _json.encode(root)


def values(path: Path, held: list[dict[str, str]] | None) -> Iterator[bytes]:
    """Yield the input values of a recording, each a JSON text.

    path -- the recording's folder, which holds index.jsonl
    held -- the records of the output's sheet of the input values, each
        by its header's names, or None where the output holds none

    One value for each request, in the index's order, from the lines
    after the last one held, until a line whose files are not yet
    written (raw_api_bodies.mtsv › recording.4, recording.5,
    values.1-6).
    """
    lines = _index(path)
    after = max((int(record[_LINE]) for record in held or []), default=0)
    tips = _tips(lines, after, path)
    for number, entry in lines:
        if number <= after:
            continue
        found = _bodies(entry, path)
        if found is None:
            return
        if "request_file" not in entry:
            continue
        request, response = found
        current = _request(number, entry, request, response)
        parent = _parent(current, tips)
        yield _value(entry, (request, response), current, parent)
        tips[current.thread] = current


def _index(path: Path) -> list[tuple[int, dict[str, Any]]]:
    # The index's lines, each a JSON value, numbered from 1; a
    # terminator after the last one is optional (JSON Lines).
    data = (path / FILE).read_bytes()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return [(number, _json.decode(line)) for number, line in enumerate(lines, 1)]


def _file(entry: dict[str, Any], field: str, path: Path) -> Path:
    # A path that is not absolute is in the recording's folder
    # (raw_api_bodies.mtsv › recording.3).
    named = Path(entry[field])
    return named if named.is_absolute() else path / named


def _bodies(entry: dict[str, Any], path: Path) -> tuple[Any, Any] | None:
    # A line's request and response, or None where a file it names is
    # not yet written; a line that names no request file has none
    # (raw_api_bodies.mtsv › recording.4, recording.5).
    files = [
        _file(entry, f, path) for f in ("request_file", "response_file") if f in entry
    ]
    if not all(file.is_file() for file in files):
        return None
    read = [_json.decode(file.read_bytes()) for file in files]
    return (read[0], read[1]) if len(read) == 2 else (None, read[0])


def _request(
    number: int, entry: dict[str, Any], request: Any, response: Any
) -> _Request:
    # A request's messages, each whole as messages.mtsv keeps it, then
    # its response as the message that follows, of its role, assistant,
    # and its content, which is all it holds; and its system blocks and
    # tools, each as it is kept (values.3).
    said = [messages.kept("messages", m) for m in request.get("messages", [])]
    answer = {
        "role": "assistant",
        "content": messages.kept(_RESPONSE, response[_RESPONSE]),
    }
    lists = {
        name: [messages.kept(name, unit) for unit in request.get(name) or []]
        for name in _LISTS
        if isinstance(request.get(name), list)
    }
    thread = (entry["session_id"], entry["query_source"])
    return _Request(number, thread, [*said, answer], lists)


def _tips(
    lines: list[tuple[int, dict[str, Any]]], after: int, path: Path
) -> dict[tuple[str, str], _Request]:
    # The latest request of each thread up to the last line held, read
    # again from its files (values.6).
    latest: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for number, entry in lines:
        if number <= after and "request_file" in entry:
            latest[(entry["session_id"], entry["query_source"])] = (number, entry)
    found = {}
    for thread, (number, entry) in latest.items():
        request, response = _bodies(entry, path)
        found[thread] = _request(number, entry, request, response)
    return found


def _parent(current: _Request, tips: dict[tuple[str, str], _Request]) -> tuple:
    # Of the latest request of each thread of the session, the one whose
    # messages, then response, the request's messages agree with
    # furthest, the later where two agree as far; none where none agrees
    # with its first message (values.2).
    own = current.sequence[:-1]
    best: tuple[int, int, _Request | None] = (0, _NO_PARENT, None)
    for tip in tips.values():
        if tip.thread[0] != current.thread[0]:
            continue
        agree = 0
        for mine, theirs in zip(own, tip.sequence):
            if not _json_schema.equal(mine, theirs):
                break
            agree += 1
        best = max(best, (agree, tip.line, tip), key=lambda b: (b[0], b[1]))
    return best if best[0] else (0, _NO_PARENT, None)


def _value(
    entry: dict[str, Any],
    bodies: tuple[Any, Any],
    current: _Request,
    parent: tuple,
) -> bytes:
    # One input value: the request's names, its parent and how many of
    # its messages it keeps, its counts, the units the parent does not
    # hold, and the response's members kept beside its content
    # (values.2-4).
    request, response = bodies
    kept, parent_line, tip = parent
    units: list[dict[str, Any]] = []
    system = request.get("system")
    if isinstance(system, str):
        units.append(_unit("/system", "system", system))
    for name in _LISTS:
        before = tip.lists.get(name, []) if tip else []
        for position, unit in enumerate(current.lists.get(name, [])):
            if position >= len(before) or not _json_schema.equal(
                unit, before[position]
            ):
                units.append(
                    _unit(_json_pointer.pointer(f"/{name}", position), name, unit)
                )
    said = request.get("messages", [])
    for position in range(kept, len(said)):
        message = messages.kept("messages", said[position])
        units.append(
            _unit(_json_pointer.pointer("/messages", position), "messages", message)
        )
    answer = messages.kept(_RESPONSE, response[_RESPONSE])
    units.append(
        _unit(_json_pointer.pointer("/messages", len(said)), _RESPONSE, answer)
    )
    value: dict[str, Any] = {
        _LINE: Number(str(current.line)),
        **{name: entry[name] for name in _INDEX},
        _EXTENDS: Number(str(parent_line)),
        "kept": Number(str(kept)),
        "count": {
            name: Number(str(len(current.lists.get(name, [])))) for name in _LISTS
        },
        "units": units,
    }
    for name in messages.ending():
        if response.get(name) not in (None, [], {}):
            value[name] = response[name]
    return _json.encode(value)


def _unit(pointer: str, member: str, unit: Any) -> dict[str, Any]:
    # A unit at its JSON Pointer in the request, under the member that
    # holds it (values.3).
    return {"request": {"pointer": pointer}, member: unit}
