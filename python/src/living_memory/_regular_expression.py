"""Which regular expressions a schema may hold, and what they match."""

from __future__ import annotations

__all__ = ["compile", "search"]

import math
from dataclasses import dataclass

# Each simple quantifier schema authors should limit themselves to is
# given by the least and most occurrences it allows; it, and each range
# quantifier, may be lazy (JSON Schema Validation, 4.3. Regular
# Expressions).
_QUANTIFIERS = {"*": (0, math.inf), "+": (1, math.inf), "?": (0, 1)}
# These tokens are outside the subset, and make the schema
# non-conforming (spec › module.1).
_OUTSIDE = ".\\]}"
_DIGITS = "0123456789"


def compile(pattern: str) -> _Group:
    # A pattern holds only individual characters, simple and
    # complemented character classes and ranges, the quantifiers, the
    # anchors ^ and $, and simple grouping and alternation; any other
    # token makes the schema non-conforming (JSON Schema Validation,
    # 4.3. Regular Expressions; spec › module.1).
    group, at = _alternation(pattern, 0)
    if at < len(pattern):
        raise ValueError(f"an unmatched ) in {pattern!r}")
    return group


def search(pattern: str, text: str) -> bool:
    # A pattern matches a string where it matches from any position in
    # it, not implicitly anchored at either end (JSON Schema Validation,
    # 4.3. Regular Expressions; 6.3.3. pattern).
    return bool(compile(pattern).step(text, frozenset(range(len(text) + 1))))


def _alternation(pattern: str, at: int) -> tuple[_Group, int]:
    sequences = []
    sequence, at = _sequence(pattern, at)
    sequences.append(sequence)
    while pattern[at : at + 1] == "|":
        sequence, at = _sequence(pattern, at + 1)
        sequences.append(sequence)
    return _Group(tuple(sequences)), at


def _sequence(pattern: str, at: int) -> tuple[tuple[object, ...], int]:
    # A sequence runs to an alternation or the end of its group; a
    # quantifier follows a character, a class or a group.
    nodes: list[object] = []
    while at < len(pattern) and pattern[at] not in "|)":
        node, at = _atom(pattern, at)
        if pattern[at : at + 1] in ("*", "+", "?", "{"):
            if isinstance(node, (_Start, _End)):
                raise ValueError(f"a quantifier with nothing to repeat in {pattern!r}")
            least, most, at = _bounds(pattern, at)
            if pattern[at : at + 1] == "?":
                at += 1
            node = _Repeat(node, least, most)
        nodes.append(node)
    return tuple(nodes), at


def _atom(pattern: str, at: int) -> tuple[object, int]:
    char = pattern[at]
    if char == "(":
        if pattern.startswith("(?", at):
            raise ValueError(f"not a simple group in {pattern!r}")
        group, end = _alternation(pattern, at + 1)
        if pattern[end : end + 1] != ")":
            raise ValueError(f"an unmatched ( in {pattern!r}")
        return group, end + 1
    if char == "[":
        return _class(pattern, at)
    if char == "^":
        return _Start(), at + 1
    if char == "$":
        return _End(), at + 1
    if char in _OUTSIDE:
        raise ValueError(f"{char!r} is not among the tokens of {pattern!r}")
    if char in _QUANTIFIERS or char == "{":
        raise ValueError(f"a quantifier with nothing to repeat in {pattern!r}")
    return _Char(char), at + 1


def _bounds(pattern: str, at: int) -> tuple[int, float, int]:
    # A quantifier is a simple one, or {x}, {x,y} or {x,}, where x and
    # y are decimal digits.
    char = pattern[at]
    if char in _QUANTIFIERS:
        least, most = _QUANTIFIERS[char]
        return least, most, at + 1
    end = pattern.find("}", at)
    low, comma, high = pattern[at + 1 : end].partition(",")
    if end == -1 or not _decimal(low) or (high and not _decimal(high)):
        raise ValueError(f"a quantifier with nothing to repeat in {pattern!r}")
    least = int(low)
    most = int(high) if high else math.inf if comma else least
    if most < least:
        raise ValueError(f"a range quantifier out of order in {pattern!r}")
    return least, most, end + 1


def _class(pattern: str, at: int) -> tuple[_Class, int]:
    # A class is [abc], [a-z], [^abc] or [^a-z], holding no escape and
    # no class; a '-' first or last in the class is itself.
    end = pattern.find("]", at + 1)
    inside = pattern[at + 1 : end] if end != -1 else ""
    complemented = inside.startswith("^")
    members = inside[1:] if complemented else inside
    if end == -1 or not members or "\\" in members or "[" in members:
        raise ValueError(f"not a simple character class in {pattern!r}")
    ranges = []
    i = 0
    while i < len(members):
        if i + 2 < len(members) and members[i + 1] == "-":
            if members[i] > members[i + 2]:
                raise ValueError(f"a range out of order in {pattern!r}")
            ranges.append((members[i], members[i + 2]))
            i += 3
        else:
            ranges.append((members[i], members[i]))
            i += 1
    return _Class(complemented, tuple(ranges)), end + 1


def _decimal(digits: str) -> bool:
    return bool(digits) and all(digit in _DIGITS for digit in digits)


# Each node takes the positions a match may have reached in the text
# and returns the positions it may reach after the node.


@dataclass(frozen=True)
class _Char:
    char: str

    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        return frozenset(p + 1 for p in positions if text[p : p + 1] == self.char)


@dataclass(frozen=True)
class _Class:
    complemented: bool
    ranges: tuple[tuple[str, str], ...]

    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        return frozenset(
            p + 1 for p in positions if p < len(text) and self._holds(text[p])
        )

    def _holds(self, char: str) -> bool:
        inside = any(low <= char <= high for low, high in self.ranges)
        return inside != self.complemented


@dataclass(frozen=True)
class _Start:
    # ^ matches at the beginning of input.
    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        return positions & {0}


@dataclass(frozen=True)
class _End:
    # $ matches at the end of input, and not before a final line break.
    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        return positions & {len(text)}


@dataclass(frozen=True)
class _Group:
    # A group matches where any of its alternatives does, each a
    # sequence of nodes matched one after another.
    sequences: tuple[tuple[object, ...], ...]

    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        reached: frozenset[int] = frozenset()
        for sequence in self.sequences:
            current = positions
            for node in sequence:
                current = node.step(text, current)
            reached |= current
        return reached


@dataclass(frozen=True)
class _Repeat:
    # A quantified node matches from least to most times; a lazy form
    # reaches the same positions, and so matches where its greedy form
    # does.
    node: object
    least: int
    most: float

    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        current = positions
        reached = positions if self.least == 0 else frozenset()
        count = 0
        while current and count < self.most:
            current = self.node.step(text, current)
            count += 1
            if count >= self.least:
                # Once past least, positions already reached lead only
                # to positions already reached.
                if current <= reached:
                    break
                reached |= current
        return reached
