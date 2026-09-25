"""Tests of _json_schema: every assertion of draft-07."""

import unittest

from living_memory import _json_schema as module
from living_memory._json import decode
from living_memory._json_pointer import PlacedError


def valid(instance: str, schema: str) -> bool:
    root = decode(schema.encode())
    return module.validates(decode(instance.encode()), root, root)


class TestAnyType(unittest.TestCase):
    def test_booleans_and_empty(self):
        for schema, expected in [("true", True), ("false", False), ("{}", True)]:
            with self.subTest(schema=schema):
                self.assertIs(valid("null", schema), expected)

    def test_type(self):
        cases = [
            ("1.0", '"integer"', True),
            ("1.5", '"integer"', False),
            ("2", '"number"', True),
            ('"2"', '"number"', False),
            ("[]", '["object","array"]', True),
        ]
        for instance, name, expected in cases:
            with self.subTest(instance=instance, type=name):
                self.assertIs(valid(instance, f'{{"type":{name}}}'), expected)

    def test_enum_and_const_by_equality(self):
        self.assertTrue(valid("1.0", '{"enum":[1,"a"]}'))
        self.assertFalse(valid('"1"', '{"enum":[1]}'))
        self.assertTrue(valid('{"a":[1]}', '{"const":{"a":[1.0]}}'))
        self.assertFalse(valid("true", '{"const":1}'))


class TestNumbersAndStrings(unittest.TestCase):
    def test_numbers(self):
        cases = [
            ("7.5", '{"multipleOf":2.5}', True),
            ("7", '{"multipleOf":2}', False),
            ("5", '{"maximum":5}', True),
            ("5", '{"exclusiveMaximum":5}', False),
            ("0", '{"minimum":0}', True),
            ("0", '{"exclusiveMinimum":0}', False),
            ('"x"', '{"minimum":0}', True),
        ]
        for instance, schema, expected in cases:
            with self.subTest(instance=instance, schema=schema):
                self.assertIs(valid(instance, schema), expected)

    def test_strings(self):
        cases = [
            ('"ab"', '{"maxLength":1}', False),
            ('"é"', '{"maxLength":1}', True),
            ('""', '{"minLength":1}', False),
            ('"expression"', '{"pattern":"es"}', True),
            ('"x"', '{"pattern":"^y"}', False),
        ]
        for instance, schema, expected in cases:
            with self.subTest(instance=instance, schema=schema):
                self.assertIs(valid(instance, schema), expected)


class TestArrays(unittest.TestCase):
    def test_items(self):
        cases = [
            ('[1,"x"]', '{"items":{"type":"number"}}', False),
            ('[1,"x",true]', '{"items":[{"type":"number"},{"type":"string"}]}', True),
            ("[1,2]", '{"items":[{}],"additionalItems":{"type":"string"}}', False),
            ("[1,2]", '{"items":{},"additionalItems":false}', True),
            ("[1,2]", '{"maxItems":1}', False),
            ("[]", '{"minItems":1}', False),
            ("[1,1.0]", '{"uniqueItems":true}', False),
            ("[1,7]", '{"contains":{"minimum":5}}', True),
            ("[1]", '{"contains":{"minimum":5}}', False),
        ]
        for instance, schema, expected in cases:
            with self.subTest(instance=instance, schema=schema):
                self.assertIs(valid(instance, schema), expected)


class TestObjects(unittest.TestCase):
    def test_members(self):
        schema = (
            '{"properties":{"a":{"type":"string"}},'
            '"patternProperties":{"^x-":{"type":"boolean"}},'
            '"additionalProperties":{"type":"number"},"required":["a"]}'
        )
        cases = [
            ('{"a":"x","x-y":true,"b":1}', True),
            ('{"a":"x","x-y":1}', False),
            ('{"a":"x","b":"y"}', False),
            ('{"b":1}', False),
        ]
        for instance, expected in cases:
            with self.subTest(instance=instance):
                self.assertIs(valid(instance, schema), expected)

    def test_counts_names_and_dependencies(self):
        cases = [
            ('{"a":1,"b":2}', '{"maxProperties":1}', False),
            ("{}", '{"minProperties":1}', False),
            ('{"ab":1}', '{"propertyNames":{"maxLength":1}}', False),
            ('{"m":1}', '{"dependencies":{"m":["n"]}}', False),
            ('{"m":1,"n":2}', '{"dependencies":{"m":["n"]}}', True),
            ('{"m":1}', '{"dependencies":{"m":{"required":["n"]}}}', False),
        ]
        for instance, schema, expected in cases:
            with self.subTest(instance=instance, schema=schema):
                self.assertIs(valid(instance, schema), expected)


class TestSubschemas(unittest.TestCase):
    def test_boolean_logic_and_conditions(self):
        cases = [
            ('"x"', '{"allOf":[{"type":"string"},{"const":"y"}]}', False),
            ('"x"', '{"anyOf":[{"type":"null"},{"type":"string"}]}', True),
            ('"x"', '{"oneOf":[{"type":"string"},{"const":"x"}]}', False),
            ('"x"', '{"not":{"type":"string"}}', False),
            ("1", '{"if":{"const":1},"then":{"const":2}}', False),
            ("3", '{"if":{"const":1},"then":{"const":2},"else":{"const":3}}', True),
            ("1", '{"then":{"const":2}}', True),
        ]
        for instance, schema, expected in cases:
            with self.subTest(instance=instance, schema=schema):
                self.assertIs(valid(instance, schema), expected)

    def test_not_asserted(self):
        self.assertTrue(valid('"x"', '{"format":"date","contentEncoding":"base64"}'))
        self.assertTrue(valid('"x"', '{"definitions":{"a":false}}'))


class TestLocate(unittest.TestCase):
    # value.3: for an instance that does not validate, the place named
    # is the deepest instance location where an assertion fails.
    def test_deepest_place(self):
        cases = [
            ('{"n":"x"}', '{"properties":{"n":{"type":"number"}}}', "/n"),
            (
                '{"a":[1,"x"]}',
                '{"properties":{"a":{"items":{"type":"number"}}}}',
                "/a/1",
            ),
            ("{}", '{"required":["a"]}', ""),
            ('{"b":1}', '{"additionalProperties":false}', "/b"),
            ('{"n":"x"}', '{"allOf":[{"properties":{"n":{"type":"number"}}}]}', "/n"),
            ('{"n":"x"}', '{"anyOf":[{"properties":{"n":{"type":"number"}}}]}', ""),
            (
                '{"n":"x"}',
                '{"definitions":{"N":{"type":"number"}},'
                '"properties":{"n":{"$ref":"#/definitions/N"}}}',
                "/n",
            ),
        ]
        for instance, schema, expected in cases:
            with self.subTest(instance=instance, schema=schema):
                root = decode(schema.encode())
                found = module.locate(decode(instance.encode()), root, root)
                self.assertEqual(found, expected)


class TestReferences(unittest.TestCase):
    def test_within_the_schema(self):
        schema = (
            '{"definitions":{"A":{"type":"string"}},'
            '"items":{"$ref":"#/definitions/A","type":"number"}}'
        )
        self.assertTrue(valid('["x"]', schema))
        self.assertFalse(valid("[1]", schema))

    # module.1, module.3: a failing $ref or pattern is placed by its
    # pointer in the schema.
    def test_check(self):
        cases = [
            ('{"properties":{"p":{"$ref":"other.json#/P"}}}', "/properties/p/$ref"),
            ('{"items":{"$ref":"#/definitions/none"}}', "/items/$ref"),
            ('{"properties":{"s":{"pattern":"\\\\d"}}}', "/properties/s/pattern"),
            ('{"patternProperties":{".":{}}}', "/patternProperties/."),
            (
                '{"allOf":[{},{"items":[{},{"pattern":"("}]}]}',
                "/allOf/1/items/1/pattern",
            ),
        ]
        for schema, expected in cases:
            with self.subTest(schema=schema):
                root = decode(schema.encode())
                with self.assertRaises(PlacedError) as raised:
                    module.check(root, root)
                self.assertEqual(raised.exception.pointer, expected)
        root = decode(b'{"anyOf":[{"pattern":"^a"}],"definitions":{"A":{"$ref":"#"}}}')
        module.check(root, root)

    # module.3: a reference resolves to the schema and its pointer.
    def test_resolve(self):
        root = decode(b'{"definitions":{"A":{"type":"string"}}}')
        found = module.resolve("#/definitions/A", root)
        self.assertEqual(found, ({"type": "string"}, "/definitions/A"))

    def test_named(self):
        schema = decode(b'{"properties":{"a":{}},"patternProperties":{"^x-":{}}}')
        for name, expected in [("a", True), ("x-b", True), ("b", False)]:
            with self.subTest(name=name):
                self.assertIs(module.named(name, schema), expected)


if __name__ == "__main__":
    unittest.main()
