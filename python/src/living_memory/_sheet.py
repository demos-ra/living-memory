"""The name and the header of each sheet."""

__all__ = ["header", "name"]

from living_memory import _order, _relation
from living_memory._relation import Relation, Segment

# These sheets are named by their own name alone: the sheet of the input
# values, a kind's, the sheet of instances and the sheet of runs (spec ›
# sheet.1-3).
_OWN = ("root", "kind", "instances", "runs")


def name(path: tuple[Relation, ...]) -> str:
    # The sheet of the input values is named by the root schema's title,
    # a kind's by its schema's title, else its $ref's last token, and
    # the sheets of instances and runs instances and runs; every other
    # by its parent's name and the parts of its path: the properties
    # between them written as the instance's own, then a property's
    # name, a keyword with its pattern or position where it holds
    # several, or a subschema's keyword with its title, else a
    # dependency's key, else its position where the keyword holds
    # several (spec › sheet.1-4).
    return _name(_relation.segment(path[-1]), path[:-1])


def _name(segment: Segment, above: tuple[Relation, ...]) -> str:
    # A branch of a location placed branch by branch is named after
    # that location, as its sheet would be named (spec › sheet.4).
    if segment.kind in _OWN:
        return segment.name
    if segment.within is not None:
        base = _name(segment.within, above)
    else:
        base = name(above)
    parts = [*segment.prefix, segment.name]
    if segment.kind == "branch" and segment.title is not None:
        parts.append(segment.title)
    elif segment.kind == "branch" and segment.name == "dependencies":
        parts.append(str(segment.key))
    elif segment.several:
        parts.append(str(segment.key))
    return ".".join([base, *parts])


def header(one: Relation) -> list[str]:
    # Each column is labelled with the name of its domain, a key column
    # by its own name (Codd, 1.3. A Relational View of Data; spec ›
    # sheet.5).
    return [
        column if isinstance(column, str) else column.label
        for column in _order.columns(_relation.keys(one), _relation.domains(one))
    ]
