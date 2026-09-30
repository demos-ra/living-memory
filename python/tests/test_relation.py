"""Tests of _relation: which relation and column each value takes."""

import unittest

from living_memory import _relation as module
from living_memory._json import decode

# Deeper than any recursion Python allows by default.
DEEP = 3000


def placed(schema: bytes, *values: bytes) -> tuple[module.Layout, list, list]:
    sheet_layout = module.layout(decode(schema))
    found, reports = [], []
    for position, value in enumerate(values):
        records, missed = module.place(sheet_layout, decode(value), position)
        found += records
        reports += missed
    return sheet_layout, found, reports


def keyed(found: list, one: module.Relation) -> list[dict]:
    return [
        {c: module.key(p, c) for c in module.keys(one)}
        for p in found
        if module.relation(p) is one
    ]


def key(position: str, instance: str, parent: str, pointer: str) -> dict:
    return {
        "_input value": position,
        "_instance": instance,
        "_parent": parent,
        "_pointer": pointer,
    }


def below(sheet_layout: module.Layout) -> list[module.Relation]:
    return list(module.children(sheet_layout, module.root(sheet_layout)))


class TestLayout(unittest.TestCase):
    # relation.5-15: a column, a sheet of its own, or the sheet of
    # instances, each sheet referred to in the order schema.11 takes it.
    def test_places(self):
        schema = (
            b'{"title":"t","type":"object","required":["a","p"],"properties":'
            b'{"a":{"type":"string"},"p":{"type":"object","required":["b"],'
            b'"properties":{"b":{"type":"number"}},"additionalProperties":false},'
            b'"l":{"type":"array","items":{"type":"number"}},"n":{"type":"number"}},'
            b'"additionalProperties":false}'
        )
        sheet_layout = module.layout(decode(schema))
        root = module.root(sheet_layout)
        self.assertEqual([d.label for d in module.domains(root)], ["a", "p.b"])
        sheets = below(sheet_layout)
        found = [(module.segment(c).kind, module.segment(c).name) for c in sheets]
        expected = [("runs", "_runs"), ("property", "l"), ("instances", "_instances")]
        self.assertEqual(found, expected)
        self.assertEqual(module.last(sheet_layout), (sheets[2], sheets[0]))

    # relation.2: a kind is one relation wherever it is referred to, and
    # a kind that holds itself refers to its own relation.
    def test_kinds(self):
        schema = (
            b'{"title":"t","type":"object","required":["a","b"],"properties":'
            b'{"a":{"$ref":"#/definitions/n"},"b":{"$ref":"#/definitions/n"}},'
            b'"additionalProperties":false,"definitions":{"n":{"type":"object",'
            b'"properties":{"c":{"$ref":"#/definitions/n"}},'
            b'"additionalProperties":false}}}'
        )
        sheet_layout = module.layout(decode(schema))
        first, second = below(sheet_layout)
        self.assertIs(first, second)
        self.assertEqual(module.segment(first), module.Segment("kind", (), "n"))
        self.assertEqual(module.children(sheet_layout, first), (first,))

    # relation.1: where a $ref references the root schema, the sheet of
    # the input values is its kind's one sheet, keyed as key.3 keys it.
    def test_root_kind(self):
        schema = (
            b'{"title":"t","type":"object","properties":{"n":{"$ref":"#"}},'
            b'"additionalProperties":false}'
        )
        sheet_layout, found, _ = placed(schema, b'{"n":{}}')
        root = module.root(sheet_layout)
        self.assertEqual(module.segment(root), module.Segment("kind", (), "t"))
        self.assertEqual(below(sheet_layout), [root])
        self.assertEqual(
            keyed(found, root), [key("0", "0", "", ""), key("0", "1", "0", "/n")]
        )

    # relation.3: a $ref that allOf applies adds to its instance's own
    # schema, and is no kind.
    def test_all_of_ref_no_kind(self):
        schema = (
            b'{"title":"t","type":"object","allOf":[{"$ref":"#/definitions/e"}],'
            b'"definitions":{"e":{"type":"object","properties":{"m":'
            b'{"type":"number"}},"required":["m"]}}}'
        )
        sheet_layout = module.layout(decode(schema))
        self.assertEqual(dict(sheet_layout.kinds), {})
        root = module.root(sheet_layout)
        self.assertEqual([d.label for d in module.domains(root)], ["m"])

    # relation.15, key.2: a root schema of other than one type, object
    # or array, is the sheet of instances, keyed as key.3 keys it.
    def test_root_instances(self):
        sheet_layout = module.layout(decode(b'{"title":"t","type":"string"}'))
        root = module.root(sheet_layout)
        self.assertEqual([d.label for d in module.domains(root)], ["_type", "_value"])
        self.assertIs(module.last(sheet_layout)[0], root)
        self.assertEqual(module.keys(root), tuple(key("", "", "", "")))

    # schema.1: a schema of any depth of nesting is laid out.
    def test_any_depth(self):
        schema = b'{"type":"object","properties":{"n":' * DEEP + b"{}" + b"}}" * DEEP
        sheet_layout = module.layout(decode(b'{"title":"t",' + schema[1:]))
        depth, one = 0, module.root(sheet_layout)
        while any(
            module.segment(c).name == "n" for c in module.children(sheet_layout, one)
        ):
            one = next(
                c
                for c in module.children(sheet_layout, one)
                if module.segment(c).name == "n"
            )
            depth += 1
        self.assertEqual(depth, DEEP - 1)


class TestPlace(unittest.TestCase):
    # relation.16, relation.17, key.3: a member two collecting branches
    # name is written by the first, and its field in the later one is
    # empty; a branch of a required object's location is keyed as the
    # record that holds it.
    def test_each_value_once(self):
        schema = (
            b'{"title":"t","type":"object","required":["p"],'
            b'"properties":{"p":{"anyOf":['
            b'{"title":"a","type":"object","properties":{"k":{"type":"string"}},'
            b'"required":["k"]},{"title":"b","type":"object","properties":{"k":'
            b'{"type":"string"},"m":{"type":"number"}},"required":["k","m"],'
            b'"additionalProperties":false}]}},"additionalProperties":false}'
        )
        sheet_layout, found, _ = placed(schema, b'{"p":{"k":"x","m":1}}')
        branches = [
            c for c in below(sheet_layout) if module.segment(c).kind == "branch"
        ]
        first, second = [
            next(p for p in found if module.relation(p) is one) for one in branches
        ]
        self.assertEqual([d.label for d in module.content(first)], ["k"])
        self.assertEqual([d.label for d in module.content(second)], ["m"])
        self.assertEqual(keyed(found, branches[0]), [key("0", "0", "", "")])

    # relation.14, key.3, key.5, field.4, field.5: a string's runs are
    # records of the sheet of runs, keyed by its instance; a pointer
    # keeps what it can hold of a name; what a field cannot hold is
    # reported by its pointer within the input.
    def test_runs_and_reports(self):
        sheet_layout, found, reports = placed(
            b'{"title":"t"}', b'{"a\\tb":"x\\ny\\rz"}'
        )
        instances, text = module.last(sheet_layout)
        runs = [p for p in found if module.relation(p) is text]
        self.assertEqual(
            [list(module.content(p).values()) for p in runs], [["x"], ["yz"]]
        )
        self.assertEqual(
            keyed(found, text)[1],
            {
                "_input value": "0",
                "_instance": "1",
                "_page": "0",
                "_line": "1",
                "_position": "0",
            },
        )
        self.assertEqual(keyed(found, instances)[1], key("0", "1", "0", "/ab"))
        self.assertEqual(reports, ["/0/a\tb", "/0/a\tb"])

    # relation.12: a property of several types is placed branch by
    # branch: a string molten, an array by its elements.
    def test_branch_by_branch(self):
        schema = (
            b'{"title":"t","type":"object","required":["p"],"properties":{"p":'
            b'{"anyOf":[{"type":"string"},'
            b'{"type":"array","items":{"type":"number"}}]}},'
            b'"additionalProperties":false}'
        )
        sheet_layout, found, _ = placed(schema, b'{"p":"x"}', b'{"p":[1]}')
        instances, elements = module.last(sheet_layout)[0], below(sheet_layout)[1]
        self.assertEqual(module.segment(elements).within.name, "p")
        self.assertEqual(keyed(found, instances), [key("0", "1", "0", "/p")])
        self.assertEqual(keyed(found, elements), [key("1", "2", "0", "/p/0")])

    # relation.2, relation.12: a kind whose own schema gives nothing is
    # placed branch by branch, with no sheet of its own, also where a
    # subschema applies it to the same location.
    def test_union_kind_same_location(self):
        schema = (
            b'{"title":"t","type":"object","required":["a"],"properties":'
            b'{"a":{"type":"string"}},"oneOf":[{"$ref":"#/definitions/u"}],'
            b'"definitions":{"u":{"anyOf":[{"$ref":"#/definitions/x"},'
            b'{"$ref":"#/definitions/y"}]},"x":{"type":"object","properties":'
            b'{"x":{"type":"number"}},"required":["x"]},"y":{"type":"object",'
            b'"properties":{"y":{"type":"number"}},"required":["y"]}}}'
        )
        sheet_layout, found, _ = placed(schema, b'{"a":"s","x":1}')
        names = {module.segment(c).name: c for c in below(sheet_layout)}
        self.assertNotIn("u", names)
        self.assertEqual(keyed(found, names["x"]), [key("0", "0", "", "")])
        self.assertEqual(keyed(found, names["y"]), [])

    # key.3: an instance's parent is the instance that holds it, and its
    # pointer is evaluated from it.
    def test_instance_parent(self):
        schema = b'{"title":"t","type":"object","additionalProperties":{}}'
        sheet_layout, found, _ = placed(schema, b'{"o":{"p":1}}')
        instances = module.last(sheet_layout)[0]
        self.assertEqual(
            keyed(found, instances),
            [key("0", "1", "0", "/o"), key("0", "2", "1", "/p")],
        )

    # relation.10: an optional object is one record or none.
    def test_optional_object(self):
        schema = (
            b'{"title":"t","type":"object","properties":{"o":{"type":"object",'
            b'"additionalProperties":false}},"additionalProperties":false}'
        )
        sheet_layout, found, _ = placed(schema, b"{}", b'{"o":{}}')
        self.assertEqual(
            keyed(found, below(sheet_layout)[0]), [key("1", "1", "0", "/o")]
        )

    # value.3: a value of any depth is placed, a kind that holds itself
    # one record per level, and a molten value one per instance.
    def test_any_depth(self):
        schema = (
            b'{"title":"t","type":"object","required":["n"],"properties":'
            b'{"n":{"$ref":"#/definitions/n"}},"additionalProperties":false,'
            b'"definitions":{"n":{"type":"object","properties":'
            b'{"n":{"$ref":"#/definitions/n"}},"additionalProperties":false}}}'
        )
        deep = b'{"n":' * DEEP + b"{}" + b"}" * DEEP
        sheet_layout, found, _ = placed(schema, deep)
        kind = below(sheet_layout)[0]
        self.assertEqual(len(keyed(found, kind)), DEEP)
        sheet_layout, found, _ = placed(b'{"title":"t"}', b"[" * DEEP + b"]" * DEEP)
        self.assertEqual(len(found), DEEP)


if __name__ == "__main__":
    unittest.main()
