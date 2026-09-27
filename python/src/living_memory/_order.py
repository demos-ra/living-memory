"""The order of sheets, records and columns."""

__all__ = ["columns", "instances", "members", "records", "sheets"]

from collections.abc import Callable, Sequence
from typing import Any


def sheets(
    root: Any, children: Callable[[Any], Sequence[Any]], last: Sequence[Any]
) -> list[tuple[Any, ...]]:
    # The sheet of the input values first, each sheet directly followed
    # by its subordinate sheets and theirs in turn, each once, in the
    # place of its first reference, and each given with the sheets above
    # it; the sheets the whole file shares, where reached, come last,
    # in their order (MTSV draft, Data Model; spec › order.1).
    found: list[tuple[Any, ...]] = []
    shared: dict[int, tuple[Any, ...]] = {}
    seen: set[int] = set()
    stack = [(root,)]
    while stack:
        path = stack.pop()
        if id(path[-1]) in seen:
            continue
        seen.add(id(path[-1]))
        if len(path) > 1 and any(path[-1] is one for one in last):
            shared[id(path[-1])] = path
        else:
            found.append(path)
        stack += [(*path, child) for child in reversed(children(path[-1]))]
    return found + [shared[id(one)] for one in last if id(one) in shared]


def records(
    ordered: Sequence[Any], placed: Sequence[Any], sheet: Callable[[Any], Any]
) -> list[list[Any]]:
    # Each sheet's records in the order the input holds them, never
    # sorted (spec › order.2).
    held: dict[int, list[Any]] = {id(one): [] for one in ordered}
    for one in placed:
        held[id(sheet(one))].append(one)
    return [held[id(one)] for one in ordered]


def columns(keys: Sequence[Any], domains: Sequence[Any]) -> list[Any]:
    # The key columns first, then the simple domains in the order of the
    # schema's tree (Codd, 1.3. A Relational View of Data; spec ›
    # order.3).
    return [*keys, *domains]


def members(value: Any) -> list[tuple[str | int, Any]]:
    # An object's members as written, and an array's elements as the
    # array orders them (RFC 8259, 4. Objects; 5. Arrays; spec ›
    # order.2).
    if isinstance(value, dict):
        return list(value.items())
    if isinstance(value, list):
        return list(enumerate(value))
    return []


def instances(value: Any) -> list[tuple[tuple[str | int, ...], Any]]:
    # Each instance within a value, the value itself first, and an
    # instance before the instances within it, each with the tokens
    # that lead to it; the places still to visit are kept in a list, so
    # any depth of nesting is walked (spec › order.2, value.3).
    found: list[tuple[tuple[str | int, ...], Any]] = []
    waiting: list[tuple[tuple[str | int, ...], Any]] = [((), value)]
    while waiting:
        tokens, instance = waiting.pop()
        found.append((tokens, instance))
        inner = [((*tokens, token), child) for token, child in members(instance)]
        waiting += reversed(inner)
    return found
