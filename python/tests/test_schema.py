"""Tests of _schema: what a module specification supplies."""

import unittest

from living_memory import _schema as module
from living_memory._json import decode
from living_memory._json_pointer import PlacedError

# Deeper than any recursion Python allows by default.
DEEP = 3000


class TestRead(unittest.TestCase):
    # schema.1-8: a schema that does not conform is refused at its
    # place.
    def test_refused_at_its_place(self):
        cases = [
            (b'{"type":"object"', ""),
            (b"true", ""),
            (b'{"title":"t","type":5}', "/type"),
            (
                b'{"title":"t","properties":{"n":{"minLength":-1}}}',
                "/properties/n/minLength",
            ),
            (b'{"title":"t","items":{"$ref":"#A"}}', "/items/$ref"),
            (
                b'{"title":"t","definitions":{"L":{"items":[{}]}},'
                b'"items":{"$ref":"#/definitions/L/items/00"}}',
                "/items/$ref",
            ),
            (b'{"type":"object"}', ""),
            (b'{"title":"t","n":1,"n":2}', ""),
            (b'{"title":"a\\tb"}', "/title"),
            (b'{"title":"t","anyOf":[{"title":1}]}', "/anyOf/0/title"),
            (b'{"title":"t","properties":{"a\\nb":{}}}', "/properties/a\nb"),
            (
                b'{"title":"t","patternProperties":{"^a\\f":{}}}',
                "/patternProperties/^a\f",
            ),
            (b'{"title":"t","dependencies":{"a\\r":["b"]}}', "/dependencies/a\r"),
            (
                b'{"title":"t","properties":{"s":{"pattern":"\\\\d"}}}',
                "/properties/s/pattern",
            ),
            (b'{"title":"t","patternProperties":{".":{}}}', "/patternProperties/."),
            (b'{"title":"t","items":{"$ref":"other.json#/P"}}', "/items/$ref"),
            (
                b'{"title":"t","definitions":{"A":{"$ref":"#/none"}}}',
                "/definitions/A/$ref",
            ),
            (
                b'{"title":"t","properties":{"n":{"$ref":"#/definitions/a\\tb"}},'
                b'"definitions":{"a\\tb":{}}}',
                "/properties/n/$ref",
            ),
            (
                b'{"title":"t","items":{"$ref":"#/definitions/A"},'
                b'"definitions":{"A":{"anyOf":[{"$ref":"#/definitions/A"}]}}}',
                "/definitions/A/anyOf/0/$ref",
            ),
        ]
        for schema, expected in cases:
            with self.subTest(schema=schema):
                with self.assertRaises(PlacedError) as raised:
                    module.read(schema)
                self.assertEqual(raised.exception.pointer, expected)

    # schema.7, relation.1: a kind may hold itself at a child location,
    # the root schema among them.
    def test_conforming(self):
        schemas = [
            b'{"title":"t","definitions":{"A":{"properties":'
            b'{"a":{"$ref":"#/definitions/A"}}}},"anyOf":[{"$ref":"#/definitions/A"}]}',
            b'{"title":"t","properties":{"n":{"$ref":"#"}}}',
        ]
        for schema in schemas:
            with self.subTest(schema=schema):
                self.assertEqual(module.read(schema)["title"], "t")

    # schema.1: a schema of any depth of nesting is read and checked.
    def test_any_depth(self):
        schema = b'{"properties":{"n":' * DEEP + b"{}" + b"}}" * DEEP
        self.assertEqual(module.read(b'{"title":"t",' + schema[1:])["title"], "t")


class TestResolve(unittest.TestCase):
    # schema.10: a $ref is read as the schema it references, its other
    # members ignored; true is the empty schema.
    def test_references(self):
        root = decode(
            b'{"definitions":{"A":{"$ref":"#/definitions/B"},"B":{"type":"null"}}}'
        )
        found = module.resolve(
            {"$ref": "#/definitions/A", "type": "string"}, root, "/x"
        )
        self.assertEqual(found, ({"type": "null"}, "/definitions/B"))
        self.assertEqual(module.resolve(True, root, "/x"), ({}, "/x"))


class TestTaken(unittest.TestCase):
    # schema.11: subschemas are taken in the order JSON Schema
    # Validation presents the keywords, and within a keyword as written.
    def test_order(self):
        schema = decode(
            b'{"oneOf":[{}],"anyOf":[{}],"allOf":[{}],"else":{},"then":{},"if":{},'
            b'"dependencies":{"d":{},"e":["f"]},"additionalProperties":false,'
            b'"patternProperties":{"^x":{}},"properties":{"b":{},"a":{}},'
            b'"additionalItems":{},"items":[{}]}'
        )
        found = [(keyword, key) for keyword, key, _, _ in module.taken(schema, "")]
        expected = [("items", 0), ("additionalItems", None), ("properties", "b")]
        expected += [("properties", "a"), ("patternProperties", "^x")]
        expected += [("additionalProperties", None), ("dependencies", "d")]
        expected += [("if", None), ("then", None), ("else", None), ("allOf", 0)]
        expected += [("anyOf", 0), ("oneOf", 0)]
        self.assertEqual(found, expected)
        places = [at for _, _, _, at in module.taken(schema, "/s")][:3]
        self.assertEqual(
            places, ["/s/items/0", "/s/additionalItems", "/s/properties/b"]
        )

    # schema.11: an omitted items or additionalProperties is the empty
    # schema; additionalItems applies only beside a list of items, and
    # then and else only beside if.
    def test_omitted(self):
        found = module.taken({"then": {}}, "")
        expected = [("items", None, True, "/items")]
        expected += [("additionalProperties", None, True, "/additionalProperties")]
        self.assertEqual(found, expected)


if __name__ == "__main__":
    unittest.main()
