"""Tests of _converter: the input and its schema, as MTSV."""

import pickle
import unittest

from living_memory import _converter
from living_memory._converter import (
    NonConformingError,
    NonConformingInputError,
    convert,
    sheets,
)

SCHEMA = b'{"title": "t", "type": "object", "required": ["s"],'
SCHEMA += b' "properties": {"s": {"type": "string"}}, "additionalProperties": false}'


class TestNonConformingError(unittest.TestCase):
    # schema.8: a module specification is rejected, naming the place in
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

    # value.10: an input value is rejected, naming its position and the
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
    # file.1, file.2: the whole input is one MTSV file, of the sheets
    # that hold a record; without values, no sheet.
    def test_one_file(self):
        text = convert([b'{"s": "a"}', b'{"s": "b"}'], SCHEMA)
        self.assertEqual(text, "\ft\n_input value\ts\n0\ta\n1\tb\n")
        self.assertEqual(convert([], SCHEMA), "")

    # value.2: a part of an input keeps its values' positions in the
    # whole input; sheets names every sheet the schema gives, in order.
    def test_parts(self):
        self.assertEqual(
            convert([b'{"s": "b"}'], SCHEMA, 1), "\ft\n_input value\ts\n1\tb\n"
        )
        self.assertEqual(sheets(SCHEMA), ["t", "_runs"])

    # value.9, value.10: an input value is rejected by its position and
    # place.
    def test_input_rejected(self):
        with self.assertRaises(NonConformingInputError) as raised:
            convert([b'{"s": "a"}', b'{"s": 1}'], SCHEMA)
        self.assertEqual(
            (raised.exception.position, raised.exception.pointer), (1, "/s")
        )

    # schema.1-8: a schema is rejected at its place.
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

    # field.5: what is not carried is a warning on this module's logger,
    # its pointer as a JSON string.
    def test_not_carried_logged(self):
        with self.assertLogs(_converter.__name__, "WARNING") as logged:
            convert([b'{"s": "a\\rb"}'], SCHEMA)
        self.assertEqual(
            logged.output, [f'WARNING:{_converter.__name__}:not carried: "/0/s"']
        )


if __name__ == "__main__":
    unittest.main()
