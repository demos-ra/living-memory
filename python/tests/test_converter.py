"""Tests for _converter: how the input becomes files."""

import pickle
import unittest

from living_memory import _converter
from living_memory._converter import (
    NonConformingError,
    NonConformingInputError,
    convert,
)

SCHEMA = b'{"title": "t", "type": "object", "required": ["s"],'
SCHEMA += b' "properties": {"s": {"type": "string"}}}'


class TestNonConformingError(unittest.TestCase):
    # module.1: a module specification is rejected, naming the place in
    # the schema.
    def test_module_specification(self):
        error = NonConformingError("the schema: a pattern", "/pattern")
        self.assertIsInstance(error, ValueError)
        self.assertEqual(
            (error.msg, error.pointer), ("the schema: a pattern", "/pattern")
        )
        self.assertEqual(
            str(error), 'the schema: a pattern: the module specification, "/pattern"'
        )

    # value.3: an input value is rejected, naming its position and the
    # place within it.
    def test_input(self):
        error = NonConformingInputError("an assertion fails", 1, "/s")
        self.assertIsInstance(error, NonConformingError)
        self.assertEqual(error.msg, "an assertion fails")
        self.assertEqual(error.position, 1)
        self.assertEqual(error.pointer, "/s")
        self.assertEqual(str(error), 'an assertion fails: value 1, "/s"')

    def test_pickle(self):
        error = pickle.loads(pickle.dumps(NonConformingError("m", "/a")))
        self.assertEqual((error.msg, error.pointer), ("m", "/a"))
        error = pickle.loads(pickle.dumps(NonConformingInputError("m", 0, "/a")))
        self.assertEqual((error.msg, error.position, error.pointer), ("m", 0, "/a"))


class TestConvert(unittest.TestCase):
    # file.1: without a pointer there is one file, named "" for the
    # caller to name.
    def test_one_file(self):
        files = convert([b'{"s": "a"}', b'{"s": "b"}'], SCHEMA)
        self.assertEqual(list(files), [""])

    # file.1: without values, there is still one file.
    def test_no_values(self):
        self.assertEqual(list(convert([], SCHEMA)), [""])

    # file.1: there is one file per value of the member, named by its
    # text and .mtsv, in the order first met.
    def test_files_by_member(self):
        values = [b'{"s": "b"}', b'{"s": "a"}', b'{"s": "b"}']
        files = convert(values, SCHEMA, b'"/s"')
        self.assertEqual(list(files), ["b.mtsv", "a.mtsv"])

    # module.2: the pointer is a JSON string; one that is not fails
    # whole.
    def test_pointer_not_a_string(self):
        for pointer in (b"/s", b"1", b'"/s'):
            with self.subTest(pointer=pointer):
                with self.assertRaises(NonConformingError) as raised:
                    convert([], SCHEMA, pointer)
                self.assertNotIsInstance(raised.exception, NonConformingInputError)
                self.assertEqual(raised.exception.pointer, "")

    # module.1, module.3: a schema is rejected at its place.
    def test_schema_placed(self):
        cases = [
            (
                b'{"title": "t", "properties": {"a": {"pattern": "."}}}',
                "/properties/a/pattern",
            ),
            (b'{"title": "t", "items": {"$ref": "other.json"}}', "/items/$ref"),
            (
                b'{"title": "t", "type": "object", "properties": {"a\\tb": {}}}',
                "/properties/a\tb",
            ),
            (b'{"title": "a\\tb"}', "/title"),
            (b'{"type": "object"}', ""),
            (b'{"title": "t", "n": 1, "n": 2}', ""),
        ]
        for schema, expected in cases:
            with self.subTest(schema=schema):
                with self.assertRaises(NonConformingError) as raised:
                    convert([], schema)
                self.assertNotIsInstance(raised.exception, NonConformingInputError)
                self.assertEqual(raised.exception.pointer, expected)

    # field.4: what is not carried is a warning on this module's logger,
    # its pointer as a JSON string.
    def test_not_carried_logged(self):
        with self.assertLogs(_converter.__name__, "WARNING") as logged:
            convert([b'{"s": "a\\rb"}'], SCHEMA)
        self.assertEqual(
            logged.output, [f'WARNING:{_converter.__name__}:not carried: "/0/s"']
        )


if __name__ == "__main__":
    unittest.main()
