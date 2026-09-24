"""How a call is written as the GenAI inference details event."""

__all__ = ["attribute", "logs_data", "namespaced", "typed"]

from typing import Any

from living_memory import _any_value
from living_memory._json import Number, is_unicode

# A call's input and output details are stored by this event
# (OTEL-GENAI, Event: gen_ai.client.inference.operation.details).
_EVENT_NAME = "gen_ai.client.inference.operation.details"


def logs_data(attributes: list[dict[str, Any]]) -> dict[str, Any]:
    # One LogsData holding one resourceLogs with one scopeLogs with one
    # log record, an event by its eventName (spec › event.1).
    log_record = {"eventName": _EVENT_NAME, "attributes": attributes}
    return {"resourceLogs": [{"scopeLogs": [{"logRecords": [log_record]}]}]}


def attribute(key: str, value: Any, value_type: str) -> dict[str, Any]:
    # An attribute takes the Value Type its table gives it, any being an
    # AnyValue; a value of another type is non-conforming
    # (spec › event.3, event.6).
    if value_type == "string" and _is_string(value):
        return {"key": key, "value": {"stringValue": value}}
    if value_type == "boolean" and isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if value_type == "int" and _any_value.is_int64(value):
        return {"key": key, "value": _any_value.convert(value)}
    if value_type == "double" and _any_value.is_double(value):
        return {"key": key, "value": {"doubleValue": value}}
    if value_type == "string[]" and _strings(value):
        return {"key": key, "value": _any_value.convert(value)}
    if value_type == "any":
        return {"key": key, "value": _any_value.convert(value)}
    raise ValueError(f"{key} is not a {value_type}")


def typed(key: str, value: Any, value_type: str) -> list[dict[str, Any]]:
    # An attribute is written only where its value is held, and no
    # fallback value is written (spec › event.2).
    if value is None:
        return []
    return [attribute(key, value, value_type)]


def namespaced(namespace: str, members: dict[str, Any]) -> list[dict[str, Any]]:
    # A field with no GenAI attribute is written under the provider's
    # own namespace (spec › event.5).
    return [
        attribute(namespace + name, value, "any") for name, value in members.items()
    ]


def _is_string(value: Any) -> bool:
    # A string of the string Value Type is a valid Unicode sequence
    # (spec › event.3).
    return (
        isinstance(value, str) and not isinstance(value, Number) and is_unicode(value)
    )


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(_is_string(v) for v in value)
