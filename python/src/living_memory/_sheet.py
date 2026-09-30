"""The name and the header of each sheet."""

__all__ = ["header", "name"]

from living_memory import _order, _relation
from living_memory._relation import Relation, Segment

# These sheets are named by their own name alone: the sheet of the input
# values, a kind's, the sheet of instances and the sheet of runs; the
# last two by names the specification fixes (spec › sheet.1-3).
_OWN = ("root", "kind", "instances", "runs")
_FIXED = ("instances", "runs")
# The mark the names the specification fixes begin with, which a name
# taken or composed from the schema's names is written with doubled
# where it begins with it (CSVW Metadata, 5.6 Columns; spec › sheet.3).
_MARK = "_"


def name(path: tuple[Relation, ...]) -> str:
    # The sheet of the input values is named by the root schema's title,
    # a kind's by its schema's title, else its $ref's last token, and
    # the sheets of instances and runs _instances and _runs; every other
    # by its parent's name and the parts of its path: the properties
    # between them written as the instance's own, then a property's
    # name, a keyword with its pattern or position where it holds
    # several, or a subschema's keyword with its title, else a
    # dependency's key, else its position where the keyword holds
    # several; a name composed of the schema's names is written once
    # composed (spec › sheet.1-4).
    last = _relation.segment(path[-1])
    if last.kind in _FIXED:
        return last.name
    composed = ""
    for one in path:
        composed = _composed(_relation.segment(one), composed)
    return _escaped(composed)


def _composed(segment: Segment, base: str) -> str:
    # A branch of a location placed branch by branch is named after
    # that location, as its sheet would be named (spec › sheet.4).
    chain = [segment]
    while chain[-1].within is not None:
        chain.append(chain[-1].within)
    for one in reversed(chain):
        if one.kind in _OWN:
            base = one.name
            continue
        parts = [*one.prefix, one.name]
        if one.kind == "branch" and one.title is not None:
            parts.append(one.title)
        elif one.kind == "branch" and one.name == "dependencies":
            parts.append(str(one.key))
        elif one.several:
            parts.append(str(one.key))
        base = ".".join([base, *parts])
    return base


def header(one: Relation) -> list[str]:
    # Each column is labelled with the name of its domain, a key column
    # and a column the specification fixes by its own name (Codd, 1.3.
    # A Relational View of Data; spec › sheet.3, sheet.5).
    return [
        column if isinstance(column, str) else _label(column)
        for column in _order.columns(_relation.keys(one), _relation.domains(one))
    ]


def _label(domain: _relation.Domain) -> str:
    return domain.label if domain.fixed else _escaped(domain.label)


def _escaped(text: str) -> str:
    # A name taken or composed from the schema's names that begins with
    # '_' is written with that '_' doubled (RFC 4180, 2. Definition of
    # the CSV Format; spec › sheet.3).
    return _MARK + text if text.startswith(_MARK) else text
