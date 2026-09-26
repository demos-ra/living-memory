"""Which relation and column each value is written to."""

from __future__ import annotations

__all__ = [
    "Domain",
    "Layout",
    "Placed",
    "Relation",
    "Segment",
    "children",
    "content",
    "domains",
    "key",
    "keys",
    "last",
    "layout",
    "place",
    "relation",
    "root",
    "segment",
]

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from living_memory import _field, _json, _json_pointer, _json_schema, _key, _module
from living_memory import _order, _separators

# A column holds one of these primitive types (spec › relation.2).
_SIMPLE = frozenset({"string", "number", "boolean", "null"})
# These keywords apply a subschema to the same location that is written
# as a subordinate sheet of its own (spec › relation.3).
_BRANCHES = ("dependencies", "if", "then", "else", "anyOf", "oneOf")
# These keywords can hold several subschemas, each then named by its
# pattern or position (spec › sheet.1).
_SEVERAL = ("patternProperties", "anyOf", "oneOf")
# These keywords give the branches a location of several types is
# placed by (spec › relation.3).
_UNIONS = ("anyOf", "oneOf")


@dataclass(frozen=True)
class Segment:
    # What names a relation: its kind of sheet, the properties between
    # it and its parent written as the instance's own, its own name, a
    # key, title or count where its keyword needs one, and, for a
    # branch of a location placed branch by branch, that location.
    kind: str
    prefix: tuple[str, ...]
    name: str
    key: str | int | None = None
    title: str | None = None
    several: bool = False
    within: Segment | None = None


@dataclass(frozen=True, eq=False)
class Domain:
    # A column: its label, and whether its field is a value's text, an
    # instance's type or a run of text.
    label: str
    role: str


@dataclass(frozen=True, eq=False)
class Relation:
    # A relation is a sheet: what names it, its key columns, its simple
    # domains and the sheets it refers to, in the order module.2 takes
    # them; a kind's sheet is referred to by its schema's pointer.
    segment: Segment
    keys: tuple[str, ...]
    domains: tuple[Domain, ...]
    children: tuple[Relation | str, ...]


@dataclass(frozen=True, eq=False)
class Placed:
    # One instance written to a relation: its key values and the value
    # each of its columns holds.
    relation: Relation
    keys: Mapping[str, str]
    content: Mapping[Domain, Any]


@dataclass(frozen=True)
class Layout:
    # The relation of the input values and the node that writes them,
    # each kind's node and the sheets it gives by its schema's pointer,
    # and the sheets the whole file shares, which come last.
    root: Relation
    node: Any
    kinds: Mapping[str, tuple[Any, _Refs]]
    shared: tuple[Relation, ...]


# The sheets a place gives: relations, and kinds by their pointers.
_Refs = tuple[Relation | str, ...]
_Written = tuple[list[Placed], list[str]]
_Filled = tuple[dict[Domain, Any], list[Placed], list[str]]


def layout(schema: Any) -> Layout:
    # The input values are one relation, named by the root schema's
    # title; an object's or an array's, else the sheet of instances.
    # Each schema a $ref references is a kind, one relation wherever it
    # is referred to; a kind may refer to itself, so each is built
    # after every node that refers to it by its pointer is (spec ›
    # relation.1, relation.4).
    built: dict[str, tuple[Any, _Refs]] = {}
    kinds = MappingProxyType(built)
    allowed = _allowed(schema, schema)
    molten = allowed not in ({"object"}, {"array"})
    text, instances = _shared(schema["title"] if molten else None)
    scope = _Scope(_File(schema, kinds, text, instances), (), None, frozenset())
    for at, target in _kinds(schema, schema, "").items():
        title = target.get("title") if isinstance(target, dict) else None
        name = _json_pointer.tokens(at)[-1] if title is None else title
        built[at] = _child(_Sub(target, at), scope, Segment("kind", (), name))
    named = Segment("root", (), schema["title"])
    if allowed == {"object"}:
        node, one = _object(_Sub(schema, ""), scope, named)
    elif allowed == {"array"}:
        node, one = _array(_Sub(schema, ""), scope, named)
    else:
        node, one = instances, instances.relation
    return Layout(one, node, kinds, (instances.relation, text.relation))


def place(sheet_layout: Layout, value: Any, position: int) -> _Written:
    # An input value is placed, each value once, with the pointers of
    # the text not carried (spec › relation.5, field.3).
    return sheet_layout.node.write(value, _key.root(position), None)


def root(sheet_layout: Layout) -> Relation:
    return sheet_layout.root


def children(sheet_layout: Layout, one: Relation) -> tuple[Relation, ...]:
    # The sheets a relation refers to, a kind's by its pointer.
    return _expand(sheet_layout, one.children)


def _expand(sheet_layout: Layout, refs: _Refs) -> tuple[Relation, ...]:
    # A kind's pointer stands for the sheets the kind gives.
    return tuple(
        found
        for ref in refs
        for found in (
            _expand(sheet_layout, sheet_layout.kinds[ref][1])
            if isinstance(ref, str)
            else (ref,)
        )
    )


def last(sheet_layout: Layout) -> tuple[Relation, ...]:
    return sheet_layout.shared


def segment(one: Relation) -> Segment:
    return one.segment


def keys(one: Relation) -> tuple[str, ...]:
    return one.keys


def domains(one: Relation) -> tuple[Domain, ...]:
    return one.domains


def relation(placed: Placed) -> Relation:
    return placed.relation


def key(placed: Placed, column: str) -> str:
    return placed.keys[column]


def content(placed: Placed) -> Mapping[Domain, Any]:
    return placed.content


@dataclass(frozen=True)
class _Sub:
    # A subschema and its place in the schema.
    schema: Any
    at: str


@dataclass(frozen=True)
class _File:
    # What the whole file shares: the root schema every $ref resolves
    # against, the kinds, the sheet of runs and the sheet of instances.
    root: Any
    kinds: Mapping[str, tuple[Any, _Refs]]
    text: _Text
    instances: _Instances


@dataclass(frozen=True)
class _Scope:
    # What a frame is built within: the file, the properties between the
    # frame and its sheet written as the instance's own, the names the
    # location's own schema gives, and those that give no column here
    # (spec › relation.5).
    file: _File
    prefix: tuple[str, ...]
    location: frozenset[str] | None
    excluded: frozenset[str]


@dataclass(frozen=True)
class _Rule:
    # A subschema applied to the same location: its keyword, its key or
    # position, whether its keyword holds several, and what decides
    # whether it collects an instance.
    keyword: str
    key: str | int | None
    several: bool
    condition: Any


@dataclass(frozen=True)
class _Place:
    # Where a value is: its pointer, and the pointer of the record that
    # holds it.
    at: str
    record: str


def _shared(title: str | None) -> tuple[_Text, _Instances]:
    # The file's one sheet of runs, and its one sheet of instances,
    # which is the sheet of the input values where it is named by the
    # root schema's title (spec › relation.3, relation.4, key.1).
    run = Domain("value", "run")
    runs = Relation(Segment("runs", (), "runs"), _key.RUN, (run,), ())
    text = _Text(runs, run)
    kind, value = Domain("type", "type"), Domain("value", "value")
    if title is None:
        named, keyed = Segment("instances", (), "instances"), _key.SUBORDINATE
    else:
        named, keyed = Segment("root", (), title), _key.ROOT
    one = Relation(named, keyed, (kind, value), (runs,))
    return text, _Instances(one, kind, value, text)


def _kinds(schema: Any, root_schema: Any, at: str) -> dict[str, Any]:
    # Each schema a $ref references, by its pointer, in the order the
    # schema holds the $refs (spec › module.2, relation.1).
    found: dict[str, Any] = {}
    if isinstance(schema, dict) and "$ref" in schema:
        target, target_at = _module.resolve(schema, root_schema, at)
        found[target_at] = target
    for child_at, child in _json_schema.subschemas(schema, at):
        found |= _kinds(child, root_schema, child_at)
    return found


def _refers(schema: Any) -> bool:
    return isinstance(schema, dict) and "$ref" in schema


def _allowed(schema: Any, root_schema: Any) -> set[str]:
    # The types a location allows are its type keyword's, all six
    # without one, narrowed by allOf and by the union of anyOf and
    # oneOf; an integer is a number (spec › relation.1, value.2).
    schema = _module.resolve(schema, root_schema, "")[0]
    if schema is False:
        return set()
    if "type" in schema:
        listed = schema["type"]
        names = listed if isinstance(listed, list) else [listed]
        types = {"number" if name == "integer" else name for name in names}
    else:
        types = set(_json.TYPES)
    for child in schema.get("allOf", []):
        types &= _allowed(child, root_schema)
    for word in ("anyOf", "oneOf"):
        if word in schema:
            types &= set().union(*(_allowed(c, root_schema) for c in schema[word]))
    return types


def _child(sub: _Sub, scope: _Scope, named: Segment) -> tuple[Any, _Refs]:
    # A child location's instances go to its kind's sheet where a $ref
    # applies to it; else they are the records of its own sheet: an
    # object's, an array's, one simple type's with the column value;
    # else they are placed branch by branch where anyOf or oneOf gives
    # the branches, and else molten (spec › relation.1-4).
    if _refers(sub.schema):
        return _kind(sub, scope)
    allowed = _allowed(sub.schema, scope.file.root)
    if allowed == {"object"}:
        node, one = _object(sub, scope, named)
    elif allowed == {"array"}:
        node, one = _array(sub, scope, named)
    elif len(allowed) == 1 and allowed <= _SIMPLE:
        node, one = _value(allowed, named, scope)
    elif _branched(sub, scope):
        return _union(sub, scope, named)
    else:
        node, one = scope.file.instances, scope.file.instances.relation
    return node, (one,)


def _kind(sub: _Sub, scope: _Scope) -> tuple[_Kind, _Refs]:
    # A $ref's instances go to the sheet of the kind it references,
    # which the relation refers to by the kind's pointer (spec ›
    # relation.1).
    at = _module.resolve(sub.schema, scope.file.root, sub.at)[1]
    return _Kind(at, scope.file.kinds), (at,)


def _branched(sub: _Sub, scope: _Scope) -> bool:
    # A location whose anyOf or oneOf gives its branches (spec ›
    # relation.3).
    schema = _module.resolve(sub.schema, scope.file.root, sub.at)[0]
    return any(schema.get(word) for word in _UNIONS)


def _union(sub: _Sub, scope: _Scope, named: Segment) -> tuple[_Union, _Refs]:
    # A location of several types is placed branch by branch: each
    # instance as the location would be if its schema were the branch,
    # a property's being taken as not required; each branch named by
    # its keyword, then its title, else its position where the keyword
    # holds several (spec › relation.3, sheet.1).
    schema, at = _module.resolve(sub.schema, scope.file.root, sub.at)
    taken = [t for t in _module.taken(schema, at) if t[0] in _UNIONS]
    counts = {w: sum(1 for x, _, _, _ in taken if x == w) for w in _UNIONS}
    branches: list[tuple[Any, Any]] = []
    below: list[Relation | str] = []
    for word, k, child, child_at in taken:
        title = child.get("title") if isinstance(child, dict) else None
        several = counts[word] > 1
        branch = Segment("branch", (), word, k, title, several, named)
        if _is_property(named):
            node, refs = _optional(_Sub(child, child_at), scope, branch)
        else:
            node, refs = _child(_Sub(child, child_at), scope, branch)
        branches.append((child, node))
        below += refs
    return _Union(tuple(branches), scope.file.root), tuple(below)


def _is_property(named: Segment) -> bool:
    # Whether a location is a property, or a branch of one.
    if named.within is not None:
        return _is_property(named.within)
    return named.kind == "property"


def _object(sub: _Sub, scope: _Scope, named: Segment) -> tuple[_Object, Relation]:
    # Each object is one record, its frame giving its columns and the
    # sheets it refers to (spec › relation.2).
    inner = _Scope(scope.file, (), None, frozenset())
    frame, found, below = _frame(sub, inner)
    one = Relation(named, _keys(named), found, below)
    return _Object(one, frame), one


def _keys(named: Segment) -> tuple[str, ...]:
    # The sheet of the input values is keyed by pointer alone, every
    # other by parent, then pointer (spec › key.1).
    return _key.ROOT if named.kind == "root" else _key.SUBORDINATE


def _array(sub: _Sub, scope: _Scope, named: Segment) -> tuple[_Array, Relation]:
    # Each array is one record; one schema of items holds every element,
    # and else each schema of items and additionalItems the elements at
    # its positions (spec › relation.3).
    schema, at = _module.resolve(sub.schema, scope.file.root, sub.at)
    taken = _module.taken(schema, at)
    listed = sum(1 for word, k, _, _ in taken if word == "items" and k is not None)
    items: list[Any] = []
    rest = None
    below: list[Relation | str] = []
    for word, k, child, child_at in taken:
        if word != "items" and (word != "additionalItems" or child is False):
            continue
        several = word == "items" and listed > 1
        item = Segment("keyword", (), word, k, several=several)
        node, refs = _child(_Sub(child, child_at), scope, item)
        below += refs
        if k is None:
            rest = node
        else:
            items.append(node)
    whole = Relation(named, _keys(named), (), tuple(below))
    return _Array(whole, tuple(items), rest), whole


def _value(allowed: set[str], named: Segment, scope: _Scope) -> tuple[_Value, Relation]:
    # A child instance of one simple type is one record, in the column
    # value, with its string's runs where it can be a string (spec ›
    # relation.2, relation.3).
    domain = Domain("value", "value")
    text, below = _text(allowed, scope)
    one = Relation(named, _keys(named), (domain,), below)
    return _Value(one, domain, text), one


def _text(
    allowed: set[str], scope: _Scope
) -> tuple[_Text | None, tuple[Relation, ...]]:
    # A place that can hold a string refers to the file's sheet of runs
    # (spec › relation.3).
    if "string" not in allowed:
        return None, ()
    return scope.file.text, (scope.file.text.relation,)


def _frame(
    sub: _Sub, scope: _Scope
) -> tuple[_Frame, tuple[Domain, ...], tuple[Relation | str, ...]]:
    # An object location's members are placed by its schema: a simple
    # required property a column, a required object's members and
    # allOf's the instance's own, and every other subschema a sheet of
    # its own or its kind's, in the order module.2 takes them (spec ›
    # relation.1-5).
    schema, at = _module.resolve(sub.schema, scope.file.root, sub.at)
    own = frozenset(schema.get("properties", {}))
    location = own if scope.location is None else scope.location
    required = schema.get("required", [])
    taken = _module.taken(schema, at)
    counts = {w: sum(1 for x, _, _, _ in taken if x == w) for w in _SEVERAL}
    members: dict[str, Any] = {}
    matching: list[tuple[str, Any]] = []
    additional = None
    same: list[Any] = []
    found: list[Domain] = []
    below: list[Relation | str] = []
    earlier = location
    for word, k, child, child_at in taken:
        child_sub = _Sub(child, child_at)
        if word == "properties":
            allowed = _allowed(child, scope.file.root)
            domain = k in required and not _refers(child)
            if domain and len(allowed) == 1 and allowed <= _SIMPLE:
                if k in scope.excluded:
                    continue
                member, columns, sheets = _column(k, allowed, scope)
            elif domain and allowed == {"object"}:
                if k in scope.excluded:
                    continue
                member, columns, sheets = _absorbed(k, child_sub, scope)
            else:
                member, columns, sheets = _property(k, child_sub, scope)
            members[k] = member
            found += columns
            below += sheets
        elif word == "patternProperties":
            several = counts[word] > 1
            item = Segment("keyword", scope.prefix, word, k, several=several)
            node, refs = _child(child_sub, scope, item)
            matching.append((k, node))
            below += refs
        elif word == "additionalProperties" and child is not False:
            item = Segment("keyword", scope.prefix, word)
            additional, refs = _child(child_sub, scope, item)
            below += refs
        elif word == "allOf":
            inner = _Scope(scope.file, scope.prefix, location, earlier)
            merged, columns, sheets = _frame(child_sub, inner)
            earlier |= merged.own
            same.append(_Merged(merged))
            found += columns
            below += sheets
        elif word in _BRANCHES and _allowed(child, scope.file.root) == {"object"}:
            conditions = {"dependencies": k, "then": schema.get("if")}
            conditions["else"] = schema.get("if")
            several = counts.get(word, 0) > 1
            rule = _Rule(word, k, several, conditions.get(word, child))
            branch, refs = _branch(rule, child_sub, scope)
            same.append(branch)
            below += refs
    frame = _Frame(schema, own, members, tuple(matching), additional, tuple(same))
    return frame, tuple(found), tuple(below)


def _column(name: str, allowed: set[str], scope: _Scope) -> tuple[Any, ...]:
    # A required property of one simple type is a column of the sheet
    # that holds it, labelled by the properties between them and its
    # name (spec › relation.2, sheet.2).
    domain = Domain(".".join((*scope.prefix, name)), "value")
    text, below = _text(allowed, scope)
    return _Column(domain, text), (domain,), below


def _absorbed(name: str, sub: _Sub, scope: _Scope) -> tuple[Any, ...]:
    # A required object's members are the instance's own (spec ›
    # relation.2).
    inner = _Scope(scope.file, (*scope.prefix, name), None, frozenset())
    return _frame(sub, inner)


def _property(name: str, sub: _Sub, scope: _Scope) -> tuple[Any, ...]:
    # A property that is no column is placed as a property its schema
    # does not require (spec › relation.3, relation.4).
    node, refs = _optional(sub, scope, Segment("property", scope.prefix, name))
    return node, (), refs


def _optional(sub: _Sub, scope: _Scope, named: Segment) -> tuple[Any, _Refs]:
    # A kind goes to its kind's sheet; an array and an optional object
    # are sheets of their own, an array whose items is one schema
    # giving the sheet of its elements; a property of several types is
    # placed branch by branch where anyOf or oneOf gives the branches;
    # any other is molten (spec › relation.1, relation.3, relation.4).
    allowed = _allowed(sub.schema, scope.file.root)
    if allowed == {"array"} and not _refers(sub.schema):
        items = sub.schema.get("items", True)
        if not isinstance(items, list):
            items_at = _json_pointer.pointer(sub.at, "items")
            node, refs = _child(_Sub(items, items_at), scope, named)
            return _Elements(node), refs
    if _refers(sub.schema) or allowed in ({"object"}, {"array"}):
        return _child(sub, scope, named)
    if len(allowed) > 1 and _branched(sub, scope):
        return _union(sub, scope, named)
    return scope.file.instances, (scope.file.instances.relation,)


def _branch(rule: _Rule, sub: _Sub, scope: _Scope) -> tuple[_Branch, _Refs]:
    # A subschema applied to the same location that allows exactly one
    # type, object, is its kind's sheet where a $ref applies it, else a
    # subordinate sheet of that location's sheet, named by its title,
    # else a dependency's key, else its position where its keyword
    # holds several (spec › relation.1, relation.3, sheet.1).
    if _refers(sub.schema):
        node, refs = _kind(sub, scope)
        return _Branch(rule, node, scope.file.root), refs
    title = sub.schema.get("title")
    named = Segment("branch", scope.prefix, rule.keyword, rule.key, title, rule.several)
    node, one = _object(sub, scope, named)
    return _Branch(rule, node, scope.file.root), (one,)


def _member(at: str, token: str | int) -> tuple[str, list[str]]:
    # A member name within a pointer holds no HT, LF, FF or CR, nor what
    # is not text; what is left out is reported by the pointer as
    # written (spec › field.3, key.1).
    if isinstance(token, int):
        return _key.member(at, token), []
    kept, dropped = _field.name_carried(token)
    return _key.member(at, kept), [_key.member(at, token)] if dropped else []


def _pointer(at: str, tokens: tuple[str | int, ...]) -> tuple[str, list[str]]:
    # An instance's pointer through the tokens that lead to it; only the
    # last token's name is its own to report (spec › field.3, key.1).
    missed: list[str] = []
    for token in tokens:
        at, missed = _member(at, token)
    return at, missed


def _cell(
    value: Any, at: str, text: _Text | None
) -> tuple[Any, list[Placed], list[str]]:
    # A string keeps what a field can hold, what is left out reported by
    # its pointer; one holding FF, a line break or HT has its runs
    # written to the sheet of runs (spec › relation.3, field.3).
    if _json.type(value) != "string":
        return value, [], []
    kept, dropped = _field.carried(value)
    reports = [at] if dropped else []
    if text is None or not _separators.holds_separator(kept):
        return kept, [], reports
    return kept, text.write(kept, at), reports


@dataclass(frozen=True)
class _Text:
    # A string's runs of text: split at each FF into pages, each page at
    # each line break into lines, each line at each HT; each run one
    # record with its zero-based page, line and position (RFC 20, 5.2
    # Control Characters; spec › relation.3).
    relation: Relation
    run: Domain

    def write(self, text: str, at: str) -> list[Placed]:
        return [
            Placed(
                self.relation,
                _key.run(at, (page, line, position)),
                MappingProxyType({self.run: run}),
            )
            for page, page_text in enumerate(_separators.pages(text))
            for line, line_text in enumerate(_separators.lines(page_text))
            for position, run in enumerate(_separators.runs(line_text))
        ]


@dataclass(frozen=True)
class _Kind:
    # A place a $ref applies a kind to writes to the kind's one sheet,
    # found by its schema's pointer (spec › relation.1).
    at: str
    kinds: Mapping[str, tuple[Any, _Refs]]

    @property
    def node(self) -> Any:
        return self.kinds[self.at][0]

    @property
    def relation(self) -> Relation:
        return self.node.relation

    @property
    def frame(self) -> _Frame:
        return self.node.frame

    def write(self, value: Any, at: str, parent: str | None) -> _Written:
        return self.node.write(value, at, parent)


@dataclass(frozen=True)
class _Union:
    # A location of several types: each instance is placed by the first
    # branch, in module.2's order, that it is valid against, each value
    # once (spec › relation.3, relation.5).
    branches: tuple[tuple[Any, Any], ...]
    root: Any

    def write(self, value: Any, at: str, parent: str | None) -> _Written:
        for schema, node in self.branches:
            if _json_schema.validates(value, schema, self.root):
                return node.write(value, at, parent)
        return [], []


@dataclass(frozen=True)
class _Column:
    # A simple domain of the record that holds it (spec › relation.2).
    domain: Domain
    text: _Text | None


@dataclass(frozen=True)
class _Value:
    # A child instance of one simple type is one record, in the column
    # value (spec › relation.2).
    relation: Relation
    domain: Domain
    text: _Text | None

    def write(self, value: Any, at: str, parent: str | None) -> _Written:
        kept, runs, reports = _cell(value, at, self.text)
        values = MappingProxyType({self.domain: kept})
        record = Placed(
            self.relation, _key.values(self.relation.keys, parent, at), values
        )
        return [record, *runs], reports


@dataclass(frozen=True)
class _Instances:
    # Each instance within the value is one record, in the order
    # order.2 gives them, its parent the pointer of the instance that
    # holds it (spec › relation.4, key.1).
    relation: Relation
    kind: Domain
    value: Domain
    text: _Text

    def write(self, value: Any, at: str, parent: str | None) -> _Written:
        placed: list[Placed] = []
        reports: list[str] = []
        for tokens, instance in _order.instances(value):
            where, missed = _pointer(at, tokens)
            holder = _pointer(at, tokens[:-1])[0] if tokens else parent
            kept, runs, cell = _cell(instance, where, self.text)
            values = MappingProxyType({self.kind: instance, self.value: kept})
            keyed = _key.values(self.relation.keys, holder, where)
            placed += [Placed(self.relation, keyed, values), *runs]
            reports += missed + cell
        return placed, reports


@dataclass(frozen=True)
class _Array:
    # Each array is one record; each element goes to the sheet of the
    # items schema at its position, else to the rest's (spec ›
    # relation.3).
    relation: Relation
    items: tuple[Any, ...]
    rest: Any

    def write(self, value: Any, at: str, parent: str | None) -> _Written:
        keyed = _key.values(self.relation.keys, parent, at)
        placed = [Placed(self.relation, keyed, MappingProxyType({}))]
        reports: list[str] = []
        for position, element in _order.members(value):
            node = self.items[position] if position < len(self.items) else self.rest
            if node is not None:
                found, missed = node.write(element, _key.member(at, position), at)
                placed += found
                reports += missed
        return placed, reports


@dataclass(frozen=True)
class _Elements:
    # A property's array whose items is one schema gives its elements as
    # the records of the property's sheet, keyed by the record that
    # holds the array (spec › relation.3).
    node: Any

    def write(self, value: Any, at: str, parent: str | None) -> _Written:
        placed: list[Placed] = []
        reports: list[str] = []
        for position, element in _order.members(value):
            found, missed = self.node.write(element, _key.member(at, position), parent)
            placed += found
            reports += missed
        return placed, reports


@dataclass(frozen=True)
class _Object:
    # Each object is one record of its sheet, before the records its
    # members give (spec › relation.2, order.2).
    relation: Relation
    frame: _Frame

    def write(self, value: Any, at: str, parent: str | None) -> _Written:
        values, placed, reports = self.frame.write(value, _Place(at, at), None)
        keyed = _key.values(self.relation.keys, parent, at)
        record = Placed(self.relation, keyed, MappingProxyType(values))
        return [record, *placed], reports


@dataclass(frozen=True)
class _Frame:
    # The placement of an object location's members: its properties'
    # columns and sheets, its patterns', its additionalProperties', and
    # the subschemas applied to the same location, in module.2's order.
    schema: Mapping[str, Any]
    own: frozenset[str]
    members: Mapping[str, Any]
    patterns: tuple[tuple[str, Any], ...]
    additional: Any
    same: tuple[Any, ...]

    def covers(self, name: str) -> bool:
        return _json_schema.covers(name, self.schema)

    def takes(self) -> bool:
        return self.additional is not None

    def write(
        self, value: dict[str, Any], where: _Place, names: frozenset[str] | None
    ) -> _Filled:
        # The members placed here are written in order.2's order; each
        # subschema that collects the instance writes those placed on
        # it.
        collecting = [same for same in self.same if same.collects(value)]
        owners = self._owners(value, collecting, names)
        values: dict[Domain, Any] = {}
        placed: list[Placed] = []
        reports: list[str] = []
        for name, member in _order.members(value):
            if owners.get(name) is not self:
                continue
            at, missed = _member(where.at, name)
            found, written, cell = self._write(name, member, _Place(at, where.record))
            values |= found
            placed += written
            reports += missed + cell
        for same in collecting:
            held = frozenset(n for n, owner in owners.items() if owner is same)
            found, written, cell = same.write(value, where, held)
            values |= found
            placed += written
            reports += cell
        return values, placed, reports

    def _owners(
        self, value: dict[str, Any], collecting: list[Any], names: frozenset[str] | None
    ) -> dict[str, Any]:
        # Each member is written once: by the location's own schema
        # where it covers it, else by the first subschema, in
        # module.2's order, that collects the instance and covers it,
        # else by the first additionalProperties that applies (spec ›
        # relation.5).
        owners: dict[str, Any] = {}
        for name in value:
            if names is not None and name not in names:
                continue
            covering = [same for same in collecting if same.covers(name)]
            taking = [same for same in collecting if same.takes()]
            if self.covers(name):
                owners[name] = self
            elif covering:
                owners[name] = covering[0]
            elif self.additional is not None:
                owners[name] = self
            elif taking:
                owners[name] = taking[0]
        return owners

    def _write(self, name: str, value: Any, where: _Place) -> _Filled:
        # A member goes to its property's column, frame or sheet, else
        # to the first matching pattern's sheet, else to
        # additionalProperties'.
        member = self.members.get(name)
        if member is None:
            matched = (n for p, n in self.patterns if _json_schema.search(p, name))
            member = next(matched, self.additional)
        if isinstance(member, _Column):
            kept, runs, reports = _cell(value, where.at, member.text)
            return {member.domain: kept}, runs, reports
        if isinstance(member, _Frame):
            return member.write(value, where, None)
        if member is None:
            return {}, [], []
        placed, reports = member.write(value, where.at, where.record)
        return {}, placed, reports


@dataclass(frozen=True)
class _Merged:
    # A subschema allOf applies: its members are the instance's own, and
    # it collects every instance (spec › relation.2).
    frame: _Frame

    def collects(self, value: Any) -> bool:
        return True

    def covers(self, name: str) -> bool:
        return self.frame.covers(name)

    def takes(self) -> bool:
        return self.frame.takes()

    def write(self, value: Any, where: _Place, names: frozenset[str]) -> _Filled:
        return self.frame.write(value, where, names)


@dataclass(frozen=True)
class _Branch:
    # A subschema applied to the same location collects the instance's
    # annotations under a dependency's key, where then's if is valid,
    # where else's if is not, and else where the instance is valid
    # against it; each instance it collects is one record of its sheet,
    # or of its kind's (JSON Schema Validation, 3.3.1. Annotations and
    # Validation Outcomes; spec › relation.1, relation.3).
    rule: _Rule
    sheet: Any
    root: Any

    def collects(self, value: Any) -> bool:
        if self.rule.keyword == "dependencies":
            return self.rule.condition in value
        valid = _json_schema.validates(value, self.rule.condition, self.root)
        return not valid if self.rule.keyword == "else" else valid

    def covers(self, name: str) -> bool:
        return self.sheet.frame.covers(name)

    def takes(self) -> bool:
        return self.sheet.frame.takes()

    def write(self, value: Any, where: _Place, names: frozenset[str]) -> _Filled:
        own = _Place(where.at, where.at)
        values, placed, reports = self.sheet.frame.write(value, own, names)
        keyed = _key.values(self.sheet.relation.keys, where.record, where.at)
        record = Placed(self.sheet.relation, keyed, MappingProxyType(values))
        return {}, [record, *placed], reports
