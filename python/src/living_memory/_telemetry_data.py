"""The OTLP data tree, and where the eight are among its attributes.

The TracesData and LogsData of OTLP opentelemetry/proto/trace/v1/
trace.proto, logs/v1/logs.proto, common/v1/common.proto and resource/v1/
resource.proto, in OTLP JSON encoding.

Classes:
Message -- the fields of an OTLP message, and its child sheets

Functions:
entries -- return the rows of one line's TracesData or LogsData
message_sheets -- return the sheets of a message, in order
message_entries -- return the rows of a message and its children
attribute_entries -- return the rows of a list of attributes
set_entries -- return the rows of one of the eight
any_value -- return the JSON value an AnyValue maps to

Constants:
RESOURCE_SPANS -- ResourceSpans, and the traces tree below it
RESOURCE_LOGS -- ResourceLogs, and the logs tree below it
SETS -- the eight, by attribute key: each set's module
SHEETS -- every sheet of the output, in order, with its header
"""

__all__ = [
    "Message",
    "entries",
    "message_sheets",
    "message_entries",
    "attribute_entries",
    "set_entries",
    "any_value",
    "RESOURCE_SPANS",
    "RESOURCE_LOGS",
    "SETS",
    "SHEETS",
]

from dataclasses import dataclass
from typing import Any

from living_memory import (
    _input_messages,
    _memory_records,
    _output_messages,
    _relations,
    _retrieval_documents,
    _system_instructions,
    _tool_call_arguments,
    _tool_call_result,
    _tool_definitions,
)
from living_memory._relations import Number, cell, pointer, text

Entry = tuple[str, list[str]]

# spec › attributes.1: keyed by the owner's address, the attribute's
# key, and a pointer within its value.
_ATTRIBUTES = ["address", "key", "pointer", "type", "value"]
_KEY_LINES = ["address", "key", "line", "value"]
_VALUE_LINES = ["address", "key", "pointer", "line", "value"]
# spec › envelope.1, line.1: a structure row is keyed by its address; a
# line by the address of its text value and its position.
_LINES = ["address", "line", "value"]


@dataclass(frozen=True)
class Message:
    """The fields of an OTLP message, and its child sheets.

    columns -- the simple fields, by OTLP JSON key (spec › record.1)
    lines -- the columns the proto types as string (spec › line.1)
    parts -- the child sheets, in order, each a kind and a key:
        "one" and "many" a message of one row or of a row per element
        (spec › resource.1, span.1), "strings" a list of simple values
        (spec › resource.2), "attributes" the attributes (spec ›
        attributes.1), "eight" the attributes where the eight are found
        (spec › set.1), "body" a value of any shape (spec › body.1)
    messages -- the message of each "one" and "many" part, by its key
    """

    columns: tuple[str, ...] = ()
    lines: frozenset[str] = frozenset()
    parts: tuple[tuple[str, str], ...] = ()
    messages: tuple[tuple[str, "Message"], ...] = ()


_ENTITY_REF = Message(
    columns=("schemaUrl", "type"),
    lines=frozenset({"schemaUrl", "type"}),
    parts=(("strings", "idKeys"), ("strings", "descriptionKeys")),
)
_RESOURCE = Message(
    columns=("droppedAttributesCount",),
    parts=(("attributes", "attributes"), ("many", "entityRefs")),
    messages=(("entityRefs", _ENTITY_REF),),
)
_SCOPE = Message(
    columns=("name", "version", "droppedAttributesCount"),
    lines=frozenset({"name", "version"}),
    parts=(("attributes", "attributes"),),
)
_STATUS = Message(columns=("message", "code"), lines=frozenset({"message"}))
_EVENT = Message(
    columns=("timeUnixNano", "name", "droppedAttributesCount"),
    lines=frozenset({"name"}),
    parts=(("attributes", "attributes"),),
)
_LINK = Message(
    columns=("traceId", "spanId", "traceState", "droppedAttributesCount", "flags"),
    lines=frozenset({"traceState"}),
    parts=(("attributes", "attributes"),),
)
_SPAN = Message(
    columns=(
        "traceId",
        "spanId",
        "traceState",
        "parentSpanId",
        "flags",
        "name",
        "kind",
        "startTimeUnixNano",
        "endTimeUnixNano",
        "droppedAttributesCount",
        "droppedEventsCount",
        "droppedLinksCount",
    ),
    lines=frozenset({"traceState", "name"}),
    parts=(
        ("eight", "attributes"),
        ("one", "status"),
        ("many", "events"),
        ("many", "links"),
    ),
    messages=(("status", _STATUS), ("events", _EVENT), ("links", _LINK)),
)
_SCOPE_SPANS = Message(
    columns=("schemaUrl",),
    lines=frozenset({"schemaUrl"}),
    parts=(("one", "scope"), ("many", "spans")),
    messages=(("scope", _SCOPE), ("spans", _SPAN)),
)
RESOURCE_SPANS = Message(
    columns=("schemaUrl",),
    lines=frozenset({"schemaUrl"}),
    parts=(("one", "resource"), ("many", "scopeSpans")),
    messages=(("resource", _RESOURCE), ("scopeSpans", _SCOPE_SPANS)),
)
_LOG_RECORD = Message(
    columns=(
        "timeUnixNano",
        "observedTimeUnixNano",
        "severityNumber",
        "severityText",
        "droppedAttributesCount",
        "flags",
        "traceId",
        "spanId",
        "eventName",
    ),
    lines=frozenset({"severityText", "eventName"}),
    parts=(("eight", "attributes"), ("body", "body")),
)
_SCOPE_LOGS = Message(
    columns=("schemaUrl",),
    lines=frozenset({"schemaUrl"}),
    parts=(("one", "scope"), ("many", "logRecords")),
    messages=(("scope", _SCOPE), ("logRecords", _LOG_RECORD)),
)
RESOURCE_LOGS = Message(
    columns=("schemaUrl",),
    lines=frozenset({"schemaUrl"}),
    parts=(("one", "resource"), ("many", "scopeLogs")),
    messages=(("resource", _RESOURCE), ("scopeLogs", _SCOPE_LOGS)),
)

# spec › set.1: each of the eight is found by its attribute key.
SETS = {
    module.ATTRIBUTE: module
    for module in (
        _system_instructions,
        _tool_definitions,
        _input_messages,
        _output_messages,
        _tool_call_arguments,
        _tool_call_result,
        _memory_records,
        _retrieval_documents,
    )
}


def message_sheets(sheet: str, message: Message) -> list[tuple[str, list[str]]]:
    """Return the sheets of a message, in order, with their headers.

    sheet -- the message's sheet name
    message -- the message

    spec › file.4: each sheet is followed by its lines sheets, then by
    its child sheets. spec › nested.3: a structure sheet's name is the
    full path of its OTLP JSON keys.
    """
    sheets = [(sheet, ["address", *message.columns])]
    for name in message.columns:
        if name in message.lines:
            sheets.append((f"{sheet}.{name}", _LINES))
    children = dict(message.messages)
    for kind, key in message.parts:
        name = f"{sheet}.{key}"
        if kind in ("one", "many"):
            sheets += message_sheets(name, children[key])
        elif kind in ("attributes", "eight"):
            sheets.append((name, _ATTRIBUTES))
            sheets.append((f"{name}.key", _KEY_LINES))
            sheets.append((f"{name}.value", _VALUE_LINES))
        elif kind == "strings":
            sheets.append((name, ["address", "value"]))
            sheets.append((f"{name}.value", _LINES))
        else:
            sheets += _relations.node_sheets(name)
    return sheets


_TREES = (("resourceSpans", RESOURCE_SPANS), ("resourceLogs", RESOURCE_LOGS))

# spec › file.4: the traces tree, then the logs tree, then the eight's
# sheets.
SHEETS = (
    message_sheets("resourceSpans", RESOURCE_SPANS)
    + message_sheets("resourceLogs", RESOURCE_LOGS)
    + [sheet for module in SETS.values() for sheet in module.SHEETS]
)


def entries(index: int, data: Any) -> list[Entry]:
    """Return the rows of one line's TracesData or LogsData.

    index -- the zero-based index of the line in the file
    data -- the line's decoded JSON value

    spec › envelope.1: every row carries an address whose first token is
    the index of its line. spec › file.1: a MetricsData line holds none
    of the eight and yields no rows; spec › file.3, a field with an
    unknown name is ignored.
    """
    if not isinstance(data, dict):
        return []
    found: list[Entry] = []
    for key, message in _TREES:
        found += _many(key, message, pointer(f"/{index}", key), data.get(key))
    return found


def message_entries(
    sheet: str, message: Message, address: str, value: Any
) -> list[Entry]:
    """Return the rows of a message and of its child sheets.

    sheet -- the message's sheet name
    message -- the message
    address -- the message's address
    value -- the decoded JSON object; anything else has no rows

    spec › record.1: the simple fields are written as the input holds
    them. spec › resource.1: an absent object has no row.
    """
    if not isinstance(value, dict):
        return []
    row = [address] + [cell(text(value.get(name))) for name in message.columns]
    found: list[Entry] = [(sheet, row)]
    for name in message.columns:
        if name in message.lines:
            found += _relations.text_entries(
                f"{sheet}.{name}", [pointer(address, name)], value.get(name)
            )
    children = dict(message.messages)
    for kind, key in message.parts:
        name = f"{sheet}.{key}"
        member = value.get(key)
        if kind == "one":
            found += message_entries(name, children[key], pointer(address, key), member)
        elif kind == "many":
            found += _many(name, children[key], pointer(address, key), member)
        elif kind in ("attributes", "eight"):
            found += attribute_entries(name, address, member, kind == "eight")
        elif kind == "strings":
            found += _strings(name, pointer(address, key), member)
        elif kind == "body" and key in value:
            found += _relations.node_entries(name, [address], any_value(member), "")
    return found


def attribute_entries(sheet: str, address: str, value: Any, eight: bool) -> list[Entry]:
    """Return the rows of attributes, and of the eight among them.

    sheet -- the attributes sheet's name
    address -- the owner's address
    value -- the decoded list of KeyValue; anything else has no rows
    eight -- whether the eight are found among these attributes

    spec › attributes.1: a row per node of each attribute's value, keyed
    by the owner's address, the attribute's key and a pointer. spec ›
    set.1: on spans and log records, and there only, the eight are
    written to their own sheets and never to the attributes sheet.
    """
    if not isinstance(value, list):
        return []
    found: list[Entry] = []
    for pair in value:
        if not isinstance(pair, dict):
            continue
        key = pair.get("key") if isinstance(pair.get("key"), str) else None
        if eight and key in SETS:
            found += set_entries(key, address, pair.get("value"))
            continue
        found += _relations.text_entries(f"{sheet}.key", [address, cell(key)], key)
        found += _relations.node_entries(
            sheet, [address, cell(key)], any_value(pair.get("value")), ""
        )
    return found


def set_entries(key: str, address: str, value: Any) -> list[Entry]:
    """Return the rows of one of the eight.

    key -- the attribute's key
    address -- the address of the span or log record
    value -- the decoded AnyValue

    spec › set.1: structured, or a JSON string; both forms are read. A
    JSON string that is not JSON yields no rows.
    """
    if isinstance(value, dict) and isinstance(value.get("stringValue"), str):
        try:
            decoded = _relations.decode(value["stringValue"])
        except ValueError:
            return []
    else:
        decoded = any_value(value)
    return SETS[key].entries(address, decoded)


def any_value(value: Any) -> Any:
    """Return the JSON value an AnyValue maps to.

    value -- the decoded AnyValue

    OTEL-COMMON, Attribute representation for non-OTLP protocols: a
    string a JSON string, a boolean a JSON boolean, an integer or
    floating point number a JSON number, a byte array a Base64 JSON
    string, an empty value null, an array a JSON array, a map a JSON
    object. OTLP common.proto: string_value_strindex is read as absent.
    """
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("stringValue"), str):
        return value["stringValue"]
    if isinstance(value.get("boolValue"), bool):
        return value["boolValue"]
    for name in ("intValue", "doubleValue"):
        if isinstance(value.get(name), str):
            return Number(value[name])
    if isinstance(value.get("arrayValue"), dict):
        elements = value["arrayValue"].get("values")
        return [any_value(element) for element in _list(elements)]
    if isinstance(value.get("kvlistValue"), dict):
        pairs = value["kvlistValue"].get("values")
        return {
            pair["key"]: any_value(pair.get("value"))
            for pair in _list(pairs)
            if isinstance(pair, dict) and isinstance(pair.get("key"), str)
        }
    if isinstance(value.get("bytesValue"), str):
        return value["bytesValue"]
    return None


def _many(sheet: str, message: Message, at: str, value: Any) -> list[Entry]:
    """Return the rows of a list of messages, a row per element.

    sheet -- the messages' sheet name
    message -- the message of each element
    at -- the address of the list
    value -- the decoded list; anything else has no rows
    """
    found: list[Entry] = []
    for index, element in enumerate(_list(value)):
        found += message_entries(sheet, message, pointer(at, index), element)
    return found


def _strings(sheet: str, at: str, value: Any) -> list[Entry]:
    """Return the rows of a list of simple values, one value per row.

    sheet -- the list's sheet name
    at -- the address of the list
    value -- the decoded list; anything else has no rows

    spec › resource.2: keyed by the address of the value.
    """
    found: list[Entry] = []
    for index, element in enumerate(_list(value)):
        address = pointer(at, index)
        found.append((sheet, [address, cell(text(element))]))
        found += _relations.text_entries(f"{sheet}.value", [address], element)
    return found


def _list(value: Any) -> list[Any]:
    """Return a decoded list, or no elements for anything else."""
    return value if isinstance(value, list) else []
