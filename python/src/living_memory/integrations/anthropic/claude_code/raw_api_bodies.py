"""Claude Code's recording of the raw API bodies, as input values."""

__all__ = ["FILE", "schema", "values"]

from collections.abc import Iterator
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
# The request's kept members, in the order the request holds them, and
# the response's (raw_api_bodies.mtsv › values.1).
_REQUEST = ("system", "tools", "messages")
_RESPONSE = "content"
# Each value's members that name it, before the unit it holds.
_NAMES = ("session_id", "query_source", "request", "version", "from", "line")

# A unit: the member that holds it, its JSON Pointer in the request, and
# the unit itself.
_Unit = tuple[str, str, Any]


def schema() -> bytes:
    """Return the schema of the input values, as a JSON text.

    A value names its unit, by its conversation, its pointer in the
    request, its version, from and line, and holds the unit under the
    member that holds it (raw_api_bodies.mtsv › values.1, values.2).
    """
    count = {"type": "integer", "minimum": Number("0")}
    root = {
        "title": _TITLE,
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "query_source": {"type": "string"},
            "request": {
                "type": "object",
                "properties": {"pointer": {"type": "string"}},
                "required": ["pointer"],
                "additionalProperties": False,
            },
            "version": count,
            "from": count,
            "line": {"type": "integer", "minimum": Number("1")},
            **messages.units(),
        },
        "required": list(_NAMES),
        "additionalProperties": False,
        "definitions": messages.definitions(),
    }
    return _json.encode(root)


def values(path: Path, held: list[dict[str, str]] | None) -> Iterator[bytes]:
    """Yield the input values of a recording, each a JSON text.

    path -- the recording's folder, which holds index.jsonl
    held -- the records of the output's sheet of the input values, each
        by its header's names, or None where the output holds none

    Each kept unit once, in the order its conversation first gives it,
    from the index lines after the last one held, until a line whose
    files are not yet written (raw_api_bodies.mtsv › recording.4,
    recording.5, values.1-6).
    """
    lines = _index(path)
    after = max((int(record["line"]) for record in held or []), default=0)
    versions = _versions(held or [])
    last = _last(lines, after, path)
    for number, entry in lines:
        if number <= after:
            continue
        found = _bodies(entry, path)
        if found is None:
            return
        if "request_file" not in entry:
            continue
        request, response = found
        conversation = (entry["session_id"], entry["query_source"])
        units = _units(request, response)
        previous = last.get(conversation, [])
        for member, pointer, unit in units:
            if _held((member, pointer, unit), previous):
                continue
            key = (*conversation, pointer)
            versions[key] = versions.get(key, -1) + 1
            given = len(request["messages"])
            if member == _RESPONSE:
                given += 1
            yield _value(
                conversation, (member, pointer, unit), (versions[key], given, number)
            )
        last[conversation] = units


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


def _units(request: dict[str, Any], response: dict[str, Any]) -> list[_Unit]:
    # The request's system blocks, or its system prompt where it is a
    # string, its tools and its messages, each by its JSON Pointer; then
    # the response's content, as the message that follows, each unit as
    # the bodies hold it (RFC 6901; raw_api_bodies.mtsv › recording.2,
    # values.1).
    units: list[_Unit] = []
    for member in _REQUEST:
        held = request.get(member)
        if isinstance(held, list):
            for position, unit in enumerate(held):
                pointer = _json_pointer.pointer(f"/{member}", position)
                units.append((member, pointer, unit))
        elif held is not None:
            units.append((member, f"/{member}", held))
    following = _json_pointer.pointer("/messages", len(request["messages"]))
    units.append((_RESPONSE, following, response[_RESPONSE]))
    return units


def _held(given: _Unit, previous: list[_Unit]) -> bool:
    # A unit equal to the one the conversation's last request or
    # response gave at its pointer is held; a message re-sent after a
    # response is compared by its role and content (raw_api_bodies.mtsv
    # › values.3, values.4).
    member, pointer, unit = given
    for earlier_member, earlier_pointer, earlier in previous:
        if earlier_pointer != pointer:
            continue
        if earlier_member == member:
            return _json_schema.equal(unit, earlier)
        if earlier_member == _RESPONSE and member == "messages":
            return unit.get("role") == "assistant" and _json_schema.equal(
                unit.get("content"), earlier
            )
    return False


def _versions(held: list[dict[str, str]]) -> dict[tuple[str, str, str], int]:
    # The latest version held at each address (raw_api_bodies.mtsv ›
    # values.2, values.5).
    found: dict[tuple[str, str, str], int] = {}
    for record in held:
        key = (record["session_id"], record["query_source"], record["request.pointer"])
        found[key] = max(found.get(key, -1), int(record["version"]))
    return found


def _last(
    lines: list[tuple[int, dict[str, Any]]], after: int, path: Path
) -> dict[tuple[str, str], list[_Unit]]:
    # Each conversation's last request and response up to the last line
    # held, read again, so the first new line is compared with them
    # (raw_api_bodies.mtsv › values.6).
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for number, entry in lines:
        if number <= after and "request_file" in entry:
            latest[(entry["session_id"], entry["query_source"])] = entry
    found = {}
    for conversation, entry in latest.items():
        request, response = _bodies(entry, path)
        found[conversation] = _units(request, response)
    return found


def _value(conversation: tuple[str, str], unit: _Unit, counts: tuple) -> bytes:
    # One input value: its names, then the unit under its member
    # (raw_api_bodies.mtsv › values.2).
    member, pointer, held = unit
    version, given, line = counts
    value = {
        "session_id": conversation[0],
        "query_source": conversation[1],
        "request": {"pointer": pointer},
        "version": Number(str(version)),
        "from": Number(str(given)),
        "line": Number(str(line)),
        member: held,
    }
    return _json.encode(value)
