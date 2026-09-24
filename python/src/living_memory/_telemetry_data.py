"""The OTLP data tree, and where the eight are among its attributes.

OTLP opentelemetry/proto/trace/v1/trace.proto, logs/v1/logs.proto,
metrics/v1/metrics.proto, common/v1/common.proto and
resource/v1/resource.proto, in the OTLP JSON encoding.
"""

__all__ = ["check", "kind_of_data", "left_behind", "rows", "SHEETS"]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from living_memory import (
    _input_messages,
    _json_schema,
    _memory_records,
    _output_messages,
    _protojson,
    _retrieval_documents,
    _system_instructions,
    _tool_call_arguments,
    _tool_call_result,
    _tool_definitions,
)
from living_memory._json import Number, decode
from living_memory._protojson import NON_FINITE
from living_memory._relations import (
    Key,
    line_rows,
    node_rows,
    node_sheets,
    pointer,
    text,
)

_Row = tuple[str, list[str]]
_Sheet = tuple[str, list[str]]
_Field = tuple[str, str]
_Check = Callable[[Any, str], None]

# spec › attributes.1.
_ATTRIBUTES = ["address", "key", "pointer", "type", "value"]
_KEY_LINES = ["address", "key", "line", "value"]
_VALUE_LINES = ["address", "key", "pointer", "line", "value"]
# spec › envelope.1, line.1.
_LINES = ["address", "line", "value"]

# spec › set.1.
_SETS = {
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

# OTLP common.proto: AnyValue and KeyValue.
_ANY_VALUE = {
    "stringValue": "string",
    "boolValue": "bool",
    "intValue": "int64",
    "doubleValue": "double",
    "arrayValue": "ArrayValue",
    "kvlistValue": "KeyValueList",
    "bytesValue": "bytes",
    "stringValueStrindex": "int32",
}
_KEY_VALUE = {"key": "string", "value": "AnyValue", "keyStrindex": "int32"}


@dataclass(frozen=True)
class _Message:
    # spec › record.1, line.1; lists, spec › file.5.
    fields: tuple[_Field, ...] = ()
    parts: tuple["_Part", ...] = ()
    lists: tuple[_Field, ...] = ()

    def sheets(self, sheet: str) -> list[_Sheet]:
        # spec › file.4, nested.3.
        sheets = [(sheet, ["address", *(name for name, _ in self.fields)])]
        for name, proto in self.fields:
            if proto == "string":
                sheets.append((f"{sheet}.{name}", _LINES))
        for part in self.parts:
            sheets += part.sheets(f"{sheet}.{part.key}")
        return sheets

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        # spec › record.1, resource.1.
        if not isinstance(value, dict):
            return []
        row = [address] + [text(value.get(name)) for name, _ in self.fields]
        rows: list[_Row] = [(sheet, row)]
        for name, proto in self.fields:
            if proto == "string":
                key = Key((address,)).extend(name)
                rows += line_rows(f"{sheet}.{name}", key, value.get(name))
        for part in self.parts:
            rows += part.rows(f"{sheet}.{part.key}", address, value.get(part.key))
        return rows

    def check(self, value: Any, at: str) -> None:
        # PROTOJSON, Null values.
        if value is None:
            return
        _check_object(value, at)
        for scalar in self.fields:
            _check_scalar(value.get(scalar[0]), scalar, pointer(at, scalar[0]))
        for scalar in self.lists:
            where = pointer(at, scalar[0])
            _check_repeated(value.get(scalar[0]), where, _scalar_check(scalar))
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
    # spec › resource.1.
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
    # spec › resource.1, scope.1, span.1.
    key: str
    message: _Message

    def sheets(self, sheet: str) -> list[_Sheet]:
        return self.message.sheets(sheet)

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        at = pointer(address, self.key)
        rows: list[_Row] = []
        for index, element in enumerate(_list(value)):
            rows += self.message.rows(sheet, pointer(at, index), element)
        return rows

    def check(self, value: Any, at: str) -> None:
        _check_repeated(value, at, self.message.check)

    def left_behind(self, value: Any) -> set[str]:
        found: set[str] = set()
        for element in _list(value):
            found |= self.message.left_behind(element)
        return found


@dataclass(frozen=True)
class _SimpleValues:
    # spec › resource.2.
    key: str

    def sheets(self, sheet: str) -> list[_Sheet]:
        return [(sheet, ["address", "value"]), (f"{sheet}.value", _LINES)]

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        at = pointer(address, self.key)
        rows: list[_Row] = []
        for index, element in enumerate(_list(value)):
            key = Key((pointer(at, index),))
            rows.append((sheet, [*key.fields, text(element)]))
            rows += line_rows(f"{sheet}.value", key, element)
        return rows

    def check(self, value: Any, at: str) -> None:
        _check_repeated(value, at, _scalar_check((self.key, "string")))

    def left_behind(self, value: Any) -> set[str]:
        return set()


@dataclass(frozen=True)
class _Attributes:
    # spec › attributes.1; sets, spec › set.1.
    key: str
    sets: dict[str, Any] = field(default_factory=dict)

    def sheets(self, sheet: str) -> list[_Sheet]:
        return [
            (sheet, _ATTRIBUTES),
            (f"{sheet}.key", _KEY_LINES),
            (f"{sheet}.value", _VALUE_LINES),
        ]

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        rows: list[_Row] = []
        for pair in _list(value):
            if not isinstance(pair, dict):
                continue
            name = text(pair.get("key"))
            if name in self.sets:
                rows += self.sets[name].rows(address, _set_value(pair.get("value")))
                continue
            rows += line_rows(f"{sheet}.key", Key((address, name)), name)
            key = Key((address, name, ""))
            rows += node_rows(sheet, key, _any_value(pair.get("value")))
        return rows

    def check(self, value: Any, at: str) -> None:
        _check_repeated(value, at, _pair_check(self.sets))
        _check_keys(value, at)

    def left_behind(self, value: Any) -> set[str]:
        found: set[str] = set()
        for pair in _list(value):
            found |= _pair_left(pair)
        return found


@dataclass(frozen=True)
class _Body:
    # spec › body.1.
    key: str

    def sheets(self, sheet: str) -> list[_Sheet]:
        return node_sheets(sheet)

    def rows(self, sheet: str, address: str, value: Any) -> list[_Row]:
        if value is None:
            return []
        return node_rows(sheet, Key((address, "")), _any_value(value))

    def check(self, value: Any, at: str) -> None:
        _check_any_value(value, at)

    def left_behind(self, value: Any) -> set[str]:
        return _any_value_left(value)


_Part = _Child | _Children | _SimpleValues | _Attributes | _Body

# OTLP common.proto: KeyValueList.
_KEY_VALUE_LIST = _Attributes("values")

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
        _Attributes("attributes", _SETS),
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
    parts=(_Attributes("attributes", _SETS), _Body("body")),
)
_SCOPE_LOGS = _Message(
    fields=(("schemaUrl", "string"),),
    parts=(_Child("scope", _SCOPE), _Children("logRecords", _LOG_RECORD)),
)
_RESOURCE_LOGS = _Message(
    fields=(("schemaUrl", "string"),),
    parts=(_Child("resource", _RESOURCE), _Children("scopeLogs", _SCOPE_LOGS)),
)

# OTLP opentelemetry/proto/metrics/v1/metrics.proto: checked, never
# written (spec › file.1, file.5).
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
_BUCKETS = _Message(
    fields=(("offset", "sint32"),), lists=(("bucketCounts", "uint64"),)
)
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
_VALUE_AT_QUANTILE = _Message(
    fields=(("quantile", "double"), ("value", "double"))
)
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

# spec › file.1.
_TREES = (
    _Children("resourceSpans", _RESOURCE_SPANS),
    _Children("resourceLogs", _RESOURCE_LOGS),
)
# OTEL-FILE-EXPORTER, Telemetry data requirements.
_METRICS = _Children("resourceMetrics", _RESOURCE_METRICS)
_KINDS = (_TREES[0], _METRICS, _TREES[1])


def check(data: Any) -> None:
    # spec › file.5, set.3.
    if not isinstance(data, dict):
        raise ValueError("a line is a TracesData, MetricsData or LogsData object")
    kinds = [part.key for part in _KINDS if data.get(part.key) is not None]
    if len(kinds) > 1:
        raise ValueError(f"a line holds one kind of data, not {' and '.join(kinds)}")
    for part in _KINDS:
        part.check(data.get(part.key), f"/{part.key}")


def kind_of_data(data: dict[str, Any]) -> str:
    # spec › file.5.
    kinds = (part.key for part in _KINDS if data.get(part.key) is not None)
    return next(kinds, "")


def left_behind(data: dict[str, Any]) -> set[str]:
    # spec › file.1, file.3.
    known = {part.key for part in _TREES}
    found = {name for name in data if name not in known}
    for part in _TREES:
        found |= part.left_behind(data.get(part.key))
    if data.get(_METRICS.key) is None:
        found.discard(_METRICS.key)
    return found


def rows(index: int, data: dict[str, Any]) -> list[_Row]:
    # spec › envelope.1, file.1.
    found: list[_Row] = []
    for part in _TREES:
        found += part.rows(part.key, f"/{index}", data.get(part.key))
    return found


def _set_value(value: Any) -> Any:
    # spec › set.1.
    name, member = _winner(value)
    if name == "stringValue":
        return decode(member)
    return _any_value(value)


def _any_value(value: Any) -> Any:
    # OTEL-COMMON, Attribute representation for non-OTLP protocols; OTLP
    # common.proto.
    name, member = _winner(value)
    if name in ("", "stringValueStrindex"):
        return None
    if name == "intValue":
        return Number(member)
    if name == "doubleValue":
        if not isinstance(member, Number) and member in NON_FINITE:
            return str(member)
        return Number(member)
    if name == "arrayValue":
        return [_any_value(element) for element in _values(member)]
    if name == "kvlistValue":
        return {
            text(pair.get("key")): _any_value(pair.get("value"))
            for pair in _values(member)
            if isinstance(pair, dict)
        }
    return member


def _scalar_check(scalar: _Field) -> _Check:
    return lambda value, at: _check_scalar(value, scalar, at)


def _pair_check(sets: dict[str, Any]) -> _Check:
    return lambda value, at: _check_pair(value, at, sets)


def _check_pair(value: Any, at: str, sets: dict[str, Any]) -> None:
    # spec › set.3.
    _check_object(value, at)
    for scalar in _KEY_VALUE.items():
        if scalar[1] != "AnyValue":
            _check_scalar(value.get(scalar[0]), scalar, pointer(at, scalar[0]))
    _check_any_value(value.get("value"), pointer(at, "value"))
    key = value.get("key")
    if key in sets:
        _check_set(key, value.get("value"), pointer(at, "value"))


def _check_set(key: str, value: Any, at: str) -> None:
    # spec › set.3.
    try:
        instance = _set_value(value)
    except ValueError as error:
        message = f"{at}: the JSON string of {key} is not JSON: {error}"
        raise ValueError(message) from error
    schema = _SETS[key].SCHEMA
    if not _json_schema.validates(instance, schema, schema):
        raise ValueError(f"{at}: {key} does not validate against its schema")


def _check_any_value(value: Any, at: str) -> None:
    if value is None:
        return
    _check_object(value, at)
    for scalar in _ANY_VALUE.items():
        name, proto = scalar
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
            _check_scalar(member, scalar, where)


def _check_repeated(value: Any, at: str, check_element: _Check) -> None:
    # PROTOJSON, Representation of each type.
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
    # OTLP common.proto; spec › file.5.
    keys = [pair.get("key") for pair in _list(value) if isinstance(pair, dict)]
    for index, key in enumerate(keys):
        if key in keys[:index]:
            raise ValueError(f"{pointer(at, index)}: the key {key!r} is repeated")


def _check_object(value: Any, at: str) -> None:
    # PROTOJSON, Representation of each type.
    if not isinstance(value, dict):
        raise ValueError(f"{at}: a message is an object")


def _check_scalar(value: Any, scalar: _Field, at: str) -> None:
    # PROTOJSON, Null values.
    name, proto = scalar
    if value is not None and not _protojson.valid(value, proto, name):
        raise ValueError(f"{at}: not a valid {proto} in the OTLP JSON encoding")


def _pair_left(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    found = {name for name in value if name not in _KEY_VALUE}
    if value.get("keyStrindex") is not None:
        found.add("keyStrindex")
    return found | _any_value_left(value.get("value"))


def _any_value_left(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    found = {name for name in value if name not in _ANY_VALUE}
    if value.get("stringValueStrindex") is not None:
        found.add("stringValueStrindex")
    for element in _values(value.get("arrayValue")):
        found |= _any_value_left(element)
    for pair in _values(value.get("kvlistValue")):
        found |= _pair_left(pair)
    return found


def _winner(value: Any) -> tuple[str, Any]:
    # PROTOJSON, Duplicate keys; Null values.
    found: tuple[str, Any] = ("", None)
    if isinstance(value, dict):
        for name, member in value.items():
            if name in _ANY_VALUE and member is not None:
                found = (name, member)
    return found


def _values(value: Any) -> list[Any]:
    return _list(value.get("values")) if isinstance(value, dict) else []


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


# spec › file.4.
SHEETS = [sheet for part in _TREES for sheet in part.sheets(part.key)] + [
    sheet for module in _SETS.values() for sheet in module.SHEETS
]
