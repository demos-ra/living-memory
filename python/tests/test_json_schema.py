"""Tests of _json_schema: whether a value validates against a schema."""

import unittest

from living_memory import _json_schema as module
from living_memory._json import Number, decode


def valid(instance, schema):
    return module.validates(instance, schema, schema)


class TestKeywords(unittest.TestCase):
    def test_booleans_and_empty(self):
        self.assertTrue(valid(None, True))
        self.assertFalse(valid(None, False))
        self.assertTrue(valid(Number("1"), {}))

    def test_type(self):
        cases = [
            (Number("1.0"), "integer", True),
            (Number("1.5"), "integer", False),
            (Number("2"), "number", True),
            ("2", "number", False),
            (True, "boolean", True),
            (None, "null", True),
            ([], ["object", "array"], True),
        ]
        for instance, name, expected in cases:
            with self.subTest(instance=instance, type=name):
                self.assertIs(valid(instance, {"type": name}), expected)

    def test_enum_and_const_by_equality(self):
        self.assertTrue(valid(Number("1.0"), {"enum": [1, "a"]}))
        self.assertFalse(valid("1", {"enum": [1]}))
        self.assertTrue(valid("text", {"const": "text"}))
        self.assertFalse(valid(True, {"const": 1}))

    def test_numbers(self):
        self.assertTrue(valid(Number("0"), {"minimum": 0}))
        self.assertFalse(valid(Number("-1"), {"minimum": 0}))
        self.assertFalse(valid(Number("0"), {"exclusiveMinimum": 0}))
        self.assertTrue(valid("x", {"minimum": 0}))

    def test_arrays(self):
        self.assertFalse(valid([Number("1"), "x"], {"items": {"type": "number"}}))
        self.assertFalse(valid([], {"minItems": 1}))
        self.assertFalse(valid([Number("1"), Number("1.0")], {"uniqueItems": True}))
        self.assertTrue(valid([Number("1"), Number("2")], {"uniqueItems": True}))

    def test_objects(self):
        schema = {
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": {"type": "number"},
        }
        self.assertTrue(valid({"a": "x", "b": Number("1")}, schema))
        self.assertFalse(valid({"a": "x", "b": "y"}, schema))
        self.assertFalse(valid({"b": Number("1")}, schema))
        self.assertFalse(valid({"ab": 1}, {"propertyNames": {"const": "a"}}))

    def test_all_of_and_any_of(self):
        self.assertTrue(valid("x", {"anyOf": [{"type": "null"}, {"type": "string"}]}))
        self.assertFalse(valid("x", {"allOf": [{"type": "string"}, {"const": "y"}]}))

    def test_unsupported_keywords_ignored(self):
        self.assertTrue(valid("x", {"deprecated": True, "format": "binary"}))


class TestReferences(unittest.TestCase):
    def test_pointer(self):
        schema = {"$defs": {"A": {"type": "string"}}, "items": {"$ref": "#/$defs/A"}}
        self.assertTrue(valid(["x"], schema))
        self.assertFalse(valid([Number("1")], schema))

    def test_metaschema(self):
        schema = {"$ref": "http://json-schema.org/draft-07/schema#"}
        cases = [
            (decode('{"type": "object", "required": ["a"]}'), True),
            (True, True),
            (decode('{"type": "text"}'), False),
            (decode('{"required": "a"}'), False),
            (decode('{"properties": {"a": 1}}'), False),
            (decode('{"minLength": -1}'), False),
            (Number("1"), False),
        ]
        for instance, expected in cases:
            with self.subTest(instance):
                self.assertIs(valid(instance, schema), expected)

    def test_not_held(self):
        with self.assertRaises(ValueError):
            valid("x", {"$ref": "http://example.com/schema#"})


if __name__ == "__main__":
    unittest.main()
