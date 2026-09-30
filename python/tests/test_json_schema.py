"""Tests of _json_schema: every assertion of draft-07."""

import unittest

from living_memory import _json_schema as module
from living_memory._json import decode


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


class TestEqual(unittest.TestCase):
    # Instance equality: numbers by mathematical value, objects
    # member by member whatever their order.
    def test_equal(self):
        one = decode(b'{"a":[1.0,"x"],"b":null}')
        self.assertTrue(module.equal(one, decode(b'{"b":null,"a":[1,"x"]}')))
        self.assertFalse(module.equal(one, decode(b'{"a":[1,"y"],"b":null}')))


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


class TestDepth(unittest.TestCase):
    # value.3: a value of any depth is validated against a schema that
    # holds itself, and compared, without a limit.
    def test_any_depth(self):
        schema = '{"type":"array","items":{"$ref":"#"}}'
        deep = "[" * 5000 + "]" * 5000
        self.assertTrue(valid(deep, schema))
        self.assertFalse(valid("[" * 5000 + "1" + "]" * 5000, schema))
        one = decode(deep.encode())
        self.assertTrue(module.equal(one, decode(deep.encode())))

    # schema.1, schema.4: a pattern of any depth of nesting is read and
    # matched.
    def test_pattern_any_depth(self):
        pattern = "(" * 5000 + "a" + ")" * 5000
        self.assertTrue(module.search(pattern, "a"))
        self.assertFalse(module.search(pattern, "b"))


class TestLocate(unittest.TestCase):
    # value.10: for an instance that does not validate, the place named
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


class TestPatterns(unittest.TestCase):
    # schema.4: a pattern of the subset matches, not anchored.
    def test_subset_matches_unanchored(self):
        cases = [
            ("es", "expression", True),
            ("^x-", "x-a", True),
            ("^x-", "a-x-", False),
            ("a$", "a\n", False),
            ("a$", "ba", True),
            ("[a-c]+", "zzb", True),
            ("[^a]", "a", False),
            ("[a-]", "-", True),
            ("(ab|cd){2}", "abcd", True),
            ("(ab|cd){3}", "abcd", False),
            ("a{1,}?b", "aab", True),
            ("^a{2,3}$", "aaaa", False),
            ("^a{2}$", "aa", True),
            ("^(a|)*$", "aaa", True),
            ("^b?c*$", "", True),
            ("é", "café", True),
        ]
        for pattern, text, expected in cases:
            with self.subTest(pattern=pattern, text=text):
                self.assertEqual(module.search(pattern, text), expected)

    # schema.4: any other token is refused.
    def test_tokens_outside_the_subset(self):
        patterns = ("\\d", ".", "(?:a)", "[a\\]]", "[]", "a)", "(a", "*", "{")
        patterns += ("a**", "^*", "a{2,1}", "[z-a]", "a{x}")
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    module.compile_pattern(pattern)


class TestReferences(unittest.TestCase):
    def test_within_the_schema(self):
        schema = (
            '{"definitions":{"A":{"type":"string"}},'
            '"items":{"$ref":"#/definitions/A","type":"number"}}'
        )
        self.assertTrue(valid('["x"]', schema))
        self.assertFalse(valid("[1]", schema))

    # schema.5, schema.6, schema.10: a reference resolves to the schema
    # and its pointer, its fragment percent-decoded as UTF-8; one
    # outside the schema, or whose fragment is not a JSON Pointer, is
    # refused.
    def test_resolve(self):
        root = decode(b'{"definitions":{"A":{"type":"string"},"a b\xc3\xa9":{}}}')
        found = module.resolve("#/definitions/A", root)
        self.assertEqual(found, ({"type": "string"}, "/definitions/A"))
        found = module.resolve("#/definitions/a%20b%C3%A9", root)
        self.assertEqual(found, ({}, "/definitions/a bé"))
        refused = ("other.json#/A", "#/definitions/none", "#A", "#/definitions/a b")
        refused += ("#/definitions/%2", "#/definitions/%FF")
        for reference in refused:
            with self.subTest(reference=reference):
                with self.assertRaises(ValueError):
                    module.resolve(reference, root)

    def test_covers(self):
        schema = decode(b'{"properties":{"a":{}},"patternProperties":{"^x-":{}}}')
        for name, expected in [("a", True), ("x-b", True), ("b", False)]:
            with self.subTest(name=name):
                self.assertIs(module.covers(name, schema), expected)

    def test_subschemas(self):
        schema = decode(
            b'{"items":[{},true],"anyOf":[{}],"dependencies":'
            b'{"a":["b"],"c":{}},"definitions":{"D":{}}}'
        )
        found = [at for at, _ in module.subschemas(schema, "")]
        expected = ["/anyOf/0", "/definitions/D", "/dependencies/c"]
        expected += ["/items/0", "/items/1"]
        self.assertEqual(found, expected)


if __name__ == "__main__":
    unittest.main()
