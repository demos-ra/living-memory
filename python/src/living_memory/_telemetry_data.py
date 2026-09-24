"""The OTLP data tree.

OTLP opentelemetry/proto/trace/v1/trace.proto, logs/v1/logs.proto,
metrics/v1/metrics.proto, common/v1/common.proto and
resource/v1/resource.proto, in the OTLP JSON encoding.
"""

__all__ = ["check", "kind_of_data", "left_behind", "rows", "sheets"]

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from living_memory import _any_value, _fields, _protojson, _sets
from living_memory._json import array
from living_memory._json_pointer import pointer
from living_memory._relations import Key, line_rows, node_rows, node_sheets

_Row = tuple[str, list[str]]
_Sheet = tuple[str, list[str]]
_Field = tuple[str, str]
_Check = Callable[[Any, str], None]

# An attributes sheet and its lines sheets have these columns
# (spec › attributes.1, line.1).
_ATTRIBUTES = ["address", "key", "pointer", "type", "value"]
_KEY_LINES = ["address", "key", "line", "value"]
_VALUE_LINES = ["address", "key", "pointer", "line", "value"]
# A structure's lines sheet has these columns (spec › envelope.1,
# line.1).
_LINES = ["address", "line", "value"]

# A KeyValue has these members and types (OTLP common.proto).
_KEY_VALUE = {"key": "string", "value": "AnyValue", "keyStrindex": "int32"}


def sheets() -> list[_Sheet]:
    # The Sheets run the traces tree, then the logs tree, then the
    # eight's sheets (spec › file.4).
    found = [sheet for part in _TREES for sheet in part.sheets(part.key)]
    return found + _sets.sheets()


def check(data: Any) -> None:
    # A line is non-conforming if it is not one kind of data in the OTLP
    # JSON encoding, or holds one of the eight that does not validate
    # (spec › file.5, set.3).
    if not isinstance(data, dict):
        raise ValueError("a line is a TracesData, MetricsData or LogsData object")
    kinds = [part.key for part in _KINDS if data.get(part.key) is not None]
    if len(kinds) > 1:
        raise ValueError(f"a line holds one kind of data, not {' and '.join(kinds)}")
    for part in _KINDS:
        part.check(data.get(part.key), f"/{part.key}")


def kind_of_data(data: dict[str, Any]) -> str:
    # A line's kind of data is that of the one tree it holds
    # (spec › file.5).
    kinds = (part.key for part in _KINDS if data.get(part.key) is not None)
    return next(kinds, "")


def left_behind(data: dict[str, Any]) -> set[str]:
    # A line's unknown members, its fields used only by Profiling, and
    # MetricsData are held but not carried (spec › file.1, file.3).
    known = {part.key for part in _TREES}
    found = {name for name in data if name not in known}
    for part in _TREES:
        found |= part.left_behind(data.get(part.key))
    if data.get(_METRICS.key) is None:
        found.discard(_METRICS.key)
    return found


def rows(index: int, data: dict[str, Any]) -> list[_Row]:
    # Every row carries its address, the line's zero-based index its
    # first token (spec › envelope.1, file.1).
    found: list[_Row] = []
    for part in _TREES:
        found += part.rows(part.key, f"/{index}", data.get(part.key))
    return found


@dataclass(frozen=True)
class _Message:
    # In an OTLP message, simple fields are columns, a string field has
    # a lines sheet, and parts are child sheets; lists of simple values
    # are checked, never written (spec › record.1, line.1, file.5).
    fields: tuple[_Field, ...] = ()
    parts: tuple["_Part", ...] = ()
    lists: tuple[_Field, ...] = ()

    def sheets(self, sheet: str) -> list[_Sheet]:
        # A sheet is named by the full path of its OTLP JSON keys
        # (spec › file.4, nested.3).
        sheets = [(sheet, ["address", *(name for name, _ in self.fields)])]
        for name, proto in self.fields:
            if proto == "string":
                sheets.append((f"{sheet}.{name}", _LINES))
        for part in self.parts:
            sheets += part.sheets(f"{sheet}.{part.key}")
        return sheets

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        # The row is keyed by the message's address (spec › record.1,
        # resource.1).
        if not isinstance(value, dict):
            return []
        row = [address] + [_fields.text(value.get(name)) for name, _ in self.fields]
        rows: list[_Row] = [(sheet, row)]
        for name, proto in self.fields:
            if proto == "string":
                key = Key((address,)).extend(name)
                rows += line_rows(f"{sheet}.{name}", key, value.get(name))
        for part in self.parts:
            rows += part.rows(f"{sheet}.{part.key}", address, value.get(part.key))
        return rows

    def check(self, value: Any, at: str) -> None:
        # A message that is null is unset (PROTOJSON, Null values).
        if value is None:
            return
        _check_object(value, at)
        for name, proto in self.fields:
            _check_scalar(value.get(name), (name, proto), pointer(at, name))
        for name, proto in self.lists:
            element = _scalar_check((name, proto))
            _check_repeated(value.get(name), pointer(at, name), element)
        for part in self.parts:
            part.check(value.get(part.key), pointer(at, part.key))

    def left_behind(self, value: Any) -> set[str]:
        if not isinstance(value, dict):
            return set()
        known = {name for name, _ in self.fields} | {part.key for part in self.parts}
        found = {name for name in value if name not in known}
        for part in self.parts:
            found |= part.left_behind(value.get(part.key))
        return found


@dataclass(frozen=True)
class _Child:
    # A member holding one message is a child sheet of one row
    # (spec › resource.1).
    key: str
    message: _Message

    def sheets(self, sheet: str) -> list[_Sheet]:
        return self.message.sheets(sheet)

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        return self.message.rows(sheet, pointer(address, self.key), value)

    def check(self, value: Any, at: str) -> None:
        self.message.check(value, at)

    def left_behind(self, value: Any) -> set[str]:
        return self.message.left_behind(value)


@dataclass(frozen=True)
class _Children:
    # A repeated message is a child sheet, one row per element
    # (spec › resource.1, scope.1, span.1).
    key: str
    message: _Message

    def sheets(self, sheet: str) -> list[_Sheet]:
        return self.message.sheets(sheet)

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        at = pointer(address, self.key)
        rows: list[_Row] = []
        for index, element in enumerate(array(value)):
            rows += self.message.rows(sheet, pointer(at, index), element)
        return rows

    def check(self, value: Any, at: str) -> None:
        _check_repeated(value, at, self.message.check)

    def left_behind(self, value: Any) -> set[str]:
        found: set[str] = set()
        for element in array(value):
            found |= self.message.left_behind(element)
        return found


@dataclass(frozen=True)
class _SimpleValues:
    # A list of simple values is a child sheet, one value per row
    # (spec › resource.2).
    key: str

    def sheets(self, sheet: str) -> list[_Sheet]:
        return [(sheet, ["address", "value"]), (f"{sheet}.value", _LINES)]

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        at = pointer(address, self.key)
        rows: list[_Row] = []
        for index, element in enumerate(array(value)):
            key = Key((pointer(at, index),))
            rows.append((sheet, [*key.fields, _fields.text(element)]))
            rows += line_rows(f"{sheet}.value", key, element)
        return rows

    def check(self, value: Any, at: str) -> None:
        _check_repeated(value, at, _scalar_check((self.key, "string")))

    def left_behind(self, value: Any) -> set[str]:
        return set()


@dataclass(frozen=True)
class _Attributes:
    # The attributes of an owner are a sheet of it; on spans and log
    # records, and there only, the eight go to their own sheets
    # (spec › attributes.1, set.1). A key-value list inside a value is
    # not an attribute collection, and its keys may be empty.
    key: str
    holds_sets: bool = False
    collection: bool = True

    def sheets(self, sheet: str) -> list[_Sheet]:
        return [
            (sheet, _ATTRIBUTES),
            (f"{sheet}.key", _KEY_LINES),
            (f"{sheet}.value", _VALUE_LINES),
        ]

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        rows: list[_Row] = []
        for pair in array(value):
            if not isinstance(pair, dict):
                continue
            name = _fields.text(pair.get("key"))
            if self.holds_sets and _sets.is_set(name):
                rows += _sets.rows(name, address, pair.get("value"))
                continue
            rows += line_rows(f"{sheet}.key", Key((address, name)), name)
            key = Key((address, name, ""))
            rows += node_rows(sheet, key, _any_value.represent(pair.get("value")))
        return rows

    def check(self, value: Any, at: str) -> None:
        _check_repeated(value, at, _pair_check(self))
        _check_keys(value, at)

    def left_behind(self, value: Any) -> set[str]:
        found: set[str] = set()
        for pair in array(value):
            found |= _pair_left(pair)
        return found


@dataclass(frozen=True)
class _Body:
    # A log record's body is written as one node sheet (spec › body.1).
    key: str

    def sheets(self, sheet: str) -> list[_Sheet]:
        return node_sheets(sheet)

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        if value is None:
            return []
        return node_rows(sheet, Key((address, "")), _any_value.represent(value))

    def check(self, value: Any, at: str) -> None:
        _check_any_value(value, at)

    def left_behind(self, value: Any) -> set[str]:
        return _any_value_left(value)


_Part = _Child | _Children | _SimpleValues | _Attributes | _Body

# A KeyValueList holds a list of key-value pairs (OTLP common.proto).
_KEY_VALUE_LIST = _Attributes("values", collection=False)

_ENTITY_REF = _Message(
    fields=(("schemaUrl", "string"), ("type", "string")),
    parts=(_SimpleValues("idKeys"), _SimpleValues("descriptionKeys")),
)
_RESOURCE = _Message(
    fields=(("droppedAttributesCount", "uint32"),),
    parts=(_Attributes("attributes"), _Children("entityRefs", _ENTITY_REF)),
)
_SCOPE = _Message(
    fields=(
        ("name", "string"),
        ("version", "string"),
        ("droppedAttributesCount", "uint32"),
    ),
    parts=(_Attributes("attributes"),),
)
_STATUS = _Message(fields=(("message", "string"), ("code", "enum")))
_EVENT = _Message(
    fields=(
        ("timeUnixNano", "fixed64"),
        ("name", "string"),
        ("droppedAttributesCount", "uint32"),
    ),
    parts=(_Attributes("attributes"),),
)
_LINK = _Message(
    fields=(
        ("traceId", "bytes"),
        ("spanId", "bytes"),
        ("traceState", "string"),
        ("droppedAttributesCount", "uint32"),
        ("flags", "fixed32"),
    ),
    parts=(_Attributes("attributes"),),
)
_SPAN = _Message(
    fields=(
        ("traceId", "bytes"),
        ("spanId", "bytes"),
        ("traceState", "string"),
        ("parentSpanId", "bytes"),
        ("flags", "fixed32"),
        ("name", "string"),
        ("kind", "enum"),
        ("startTimeUnixNano", "fixed64"),
        ("endTimeUnixNano", "fixed64"),
        ("droppedAttributesCount", "uint32"),
        ("droppedEventsCount", "uint32"),
        ("droppedLinksCount", "uint32"),
    ),
    parts=(
        _Attributes("attributes", holds_sets=True),
        _Child("status", _STATUS),
        _Children("events", _EVENT),
        _Children("links", _LINK),
    ),
)
_SCOPE_SPANS = _Message(
    fields=(("schemaUrl", "string"),),
    parts=(_Child("scope", _SCOPE), _Children("spans", _SPAN)),
)
_RESOURCE_SPANS = _Message(
    fields=(("schemaUrl", "string"),),
    parts=(_Child("resource", _RESOURCE), _Children("scopeSpans", _SCOPE_SPANS)),
)
_LOG_RECORD = _Message(
    fields=(
        ("timeUnixNano", "fixed64"),
        ("observedTimeUnixNano", "fixed64"),
        ("severityNumber", "enum"),
        ("severityText", "string"),
        ("droppedAttributesCount", "uint32"),
        ("flags", "fixed32"),
        ("traceId", "bytes"),
        ("spanId", "bytes"),
        ("eventName", "string"),
    ),
    parts=(_Attributes("attributes", holds_sets=True), _Body("body")),
)
_SCOPE_LOGS = _Message(
    fields=(("schemaUrl", "string"),),
    parts=(_Child("scope", _SCOPE), _Children("logRecords", _LOG_RECORD)),
)
_RESOURCE_LOGS = _Message(
    fields=(("schemaUrl", "string"),),
    parts=(_Child("resource", _RESOURCE), _Children("scopeLogs", _SCOPE_LOGS)),
)

# The messages of opentelemetry/proto/metrics/v1/metrics.proto are
# checked and never written: the eight are not recorded on metrics
# (spec › file.1, file.5).
_EXEMPLAR = _Message(
    fields=(
        ("timeUnixNano", "fixed64"),
        ("asDouble", "double"),
        ("asInt", "sfixed64"),
        ("spanId", "bytes"),
        ("traceId", "bytes"),
    ),
    parts=(_Attributes("filteredAttributes"),),
)
_NUMBER_POINT = _Message(
    fields=(
        ("startTimeUnixNano", "fixed64"),
        ("timeUnixNano", "fixed64"),
        ("asDouble", "double"),
        ("asInt", "sfixed64"),
        ("flags", "uint32"),
    ),
    parts=(_Attributes("attributes"), _Children("exemplars", _EXEMPLAR)),
)
_HISTOGRAM_POINT = _Message(
    fields=(
        ("startTimeUnixNano", "fixed64"),
        ("timeUnixNano", "fixed64"),
        ("count", "fixed64"),
        ("sum", "double"),
        ("flags", "uint32"),
        ("min", "double"),
        ("max", "double"),
    ),
    parts=(_Attributes("attributes"), _Children("exemplars", _EXEMPLAR)),
    lists=(("bucketCounts", "fixed64"), ("explicitBounds", "double")),
)
_BUCKETS = _Message(fields=(("offset", "sint32"),), lists=(("bucketCounts", "uint64"),))
_EXPONENTIAL_HISTOGRAM_POINT = _Message(
    fields=(
        ("startTimeUnixNano", "fixed64"),
        ("timeUnixNano", "fixed64"),
        ("count", "fixed64"),
        ("sum", "double"),
        ("scale", "sint32"),
        ("zeroCount", "fixed64"),
        ("flags", "uint32"),
        ("min", "double"),
        ("max", "double"),
        ("zeroThreshold", "double"),
    ),
    parts=(
        _Attributes("attributes"),
        _Child("positive", _BUCKETS),
        _Child("negative", _BUCKETS),
        _Children("exemplars", _EXEMPLAR),
    ),
)
_VALUE_AT_QUANTILE = _Message(fields=(("quantile", "double"), ("value", "double")))
_SUMMARY_POINT = _Message(
    fields=(
        ("startTimeUnixNano", "fixed64"),
        ("timeUnixNano", "fixed64"),
        ("count", "fixed64"),
        ("sum", "double"),
        ("flags", "uint32"),
    ),
    parts=(
        _Attributes("attributes"),
        _Children("quantileValues", _VALUE_AT_QUANTILE),
    ),
)
_GAUGE = _Message(parts=(_Children("dataPoints", _NUMBER_POINT),))
_SUM = _Message(
    fields=(("aggregationTemporality", "enum"), ("isMonotonic", "bool")),
    parts=(_Children("dataPoints", _NUMBER_POINT),),
)
_HISTOGRAM = _Message(
    fields=(("aggregationTemporality", "enum"),),
    parts=(_Children("dataPoints", _HISTOGRAM_POINT),),
)
_EXPONENTIAL_HISTOGRAM = _Message(
    fields=(("aggregationTemporality", "enum"),),
    parts=(_Children("dataPoints", _EXPONENTIAL_HISTOGRAM_POINT),),
)
_SUMMARY = _Message(parts=(_Children("dataPoints", _SUMMARY_POINT),))
_METRIC = _Message(
    fields=(("name", "string"), ("description", "string"), ("unit", "string")),
    parts=(
        _Attributes("metadata"),
        _Child("gauge", _GAUGE),
        _Child("sum", _SUM),
        _Child("histogram", _HISTOGRAM),
        _Child("exponentialHistogram", _EXPONENTIAL_HISTOGRAM),
        _Child("summary", _SUMMARY),
    ),
)
_SCOPE_METRICS = _Message(
    fields=(("schemaUrl", "string"),),
    parts=(_Child("scope", _SCOPE), _Children("metrics", _METRIC)),
)
_RESOURCE_METRICS = _Message(
    fields=(("schemaUrl", "string"),),
    parts=(_Child("resource", _RESOURCE), _Children("scopeMetrics", _SCOPE_METRICS)),
)

# The traces tree is written, then the logs tree (spec › file.1,
# file.4).
_TREES = (
    _Children("resourceSpans", _RESOURCE_SPANS),
    _Children("resourceLogs", _RESOURCE_LOGS),
)
# A line holds TracesData, MetricsData or LogsData
# (OTEL-FILE-EXPORTER, Telemetry data requirements).
_METRICS = _Children("resourceMetrics", _RESOURCE_METRICS)
_KINDS = (_TREES[0], _METRICS, _TREES[1])


def _scalar_check(scalar: _Field) -> _Check:
    return lambda value, at: _check_scalar(value, scalar, at)


def _pair_check(attributes: _Attributes) -> _Check:
    return lambda value, at: _check_pair(value, at, attributes)


def _check_pair(value: Any, at: str, attributes: _Attributes) -> None:
    # A KeyValue's key and value are checked; an attribute's key is not
    # empty, unless the pair names it by key_strindex, which is not
    # fatal (OTEL-COMMON, Attribute; OTLP common.proto); and one of the
    # eight is validated against its schema (spec › file.5, set.3).
    _check_object(value, at)
    for name, proto in _KEY_VALUE.items():
        if proto != "AnyValue":
            _check_scalar(value.get(name), (name, proto), pointer(at, name))
    _check_any_value(value.get("value"), pointer(at, "value"))
    key = value.get("key")
    by_index = value.get("keyStrindex") is not None
    if attributes.collection and not key and not by_index:
        raise ValueError(f"{pointer(at, 'key')}: an attribute's key is empty")
    if attributes.holds_sets and _sets.is_set(key):
        _sets.check(key, value.get("value"), pointer(at, "value"))


def _check_any_value(value: Any, at: str) -> None:
    # Each member of an AnyValue has its type (OTLP common.proto;
    # PROTOJSON, Representation of each type).
    if value is None:
        return
    _check_object(value, at)
    for name, proto in _any_value.members():
        member = value.get(name)
        where = pointer(at, name)
        if member is None:
            continue
        if proto == "ArrayValue":
            _check_object(member, where)
            values = pointer(where, "values")
            _check_repeated(member.get("values"), values, _check_any_value)
        elif proto == "KeyValueList":
            _check_object(member, where)
            _KEY_VALUE_LIST.check(member.get("values"), pointer(where, "values"))
        else:
            _check_scalar(member, (name, proto), where)


def _check_repeated(value: Any, at: str, check_element: _Check) -> None:
    # A repeated field is an array holding no null (PROTOJSON,
    # Representation of each type; Null values).
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(f"{at}: a repeated field is an array")
    for index, element in enumerate(value):
        where = pointer(at, index)
        if element is None:
            raise ValueError(f"{where}: null is not allowed within a repeated field")
        check_element(element, where)


def _check_keys(value: Any, at: str) -> None:
    # The keys of one list of attributes are unique (OTLP common.proto;
    # spec › file.5).
    keys = [pair.get("key") for pair in array(value) if isinstance(pair, dict)]
    for index, key in enumerate(keys):
        if key in keys[:index]:
            raise ValueError(f"{pointer(at, index)}: the key {key!r} is repeated")


def _check_object(value: Any, at: str) -> None:
    # A message is a JSON object (PROTOJSON, Representation of each
    # type).
    if not isinstance(value, dict):
        raise ValueError(f"{at}: a message is an object")


def _check_scalar(value: Any, scalar: _Field, at: str) -> None:
    # A field that is null is unset; any other value has its field's
    # type (PROTOJSON, Null values).
    name, proto = scalar
    if value is not None and not _protojson.valid(value, proto, name):
        raise ValueError(f"{at}: not a valid {proto} in the OTLP JSON encoding")


def _pair_left(value: Any) -> set[str]:
    known = {name for name in _KEY_VALUE}
    if not isinstance(value, dict):
        return set()
    found = {name for name in value if name not in known}
    if value.get("keyStrindex") is not None:
        found.add("keyStrindex")
    return found | _any_value_left(value.get("value"))


def _any_value_left(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    known = {name for name, _ in _any_value.members()}
    found = {name for name in value if name not in known}
    if value.get("stringValueStrindex") is not None:
        found.add("stringValueStrindex")
    for element in _any_value.values(value.get("arrayValue")):
        found |= _any_value_left(element)
    for pair in _any_value.values(value.get("kvlistValue")):
        found |= _pair_left(pair)
    return found
