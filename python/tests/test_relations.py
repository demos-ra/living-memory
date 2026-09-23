"""Tests of _relations: how a value becomes keyed sheets."""

import unittest

from living_memory import _relations
from living_memory._relations import Number, Shape

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


class TestDecode(unittest.TestCase):
    """decode keeps each number as the text it writes."""

    def test_numbers_as_written(self):
        value = _relations.decode('{"a": 0.950, "b": 5, "c": -1e3}')
        self.assertEqual(value, {"a": "0.950", "b": "5", "c": "-1e3"})
        for number in value.values():
            with self.subTest(number):
                self.assertIsInstance(number, Number)

    def test_not_json(self):
        with self.assertRaises(ValueError):
            _relations.decode("x")


class TestKind(unittest.TestCase):
    """kind names the seven JSON value types."""

    def test_kinds(self):
        cases = [
            ({}, "object"),
            ([], "array"),
            ("x", "string"),
            (Number("1"), "number"),
            (True, "true"),
            (False, "false"),
            (None, "null"),
        ]
        for value, expected in cases:
            with self.subTest(expected):
                self.assertEqual(_relations.kind(value), expected)


class TestCell(unittest.TestCase):
    """cell leaves empty what a field cannot hold."""

    def test_cells(self):
        cases = [
            ("x", "x"),
            ("", ""),
            (None, ""),
            ("a\tb", ""),
            ("a\nb", ""),
            ("a\fb", ""),
            ("a\rb", ""),
        ]
        for value, expected in cases:
            with self.subTest(repr(value)):
                self.assertEqual(_relations.cell(value), expected)


class TestLines(unittest.TestCase):
    """lines splits a text at LF and CRLF only."""

    def test_lines(self):
        cases = [
            ("first\nsecond", ["first", "second"]),
            ("first\r\nsecond", ["first", "second"]),
            ("a\n", ["a", ""]),
            ("a\rb", []),
            ("a\r", []),
            ("x", []),
            (None, []),
        ]
        for value, expected in cases:
            with self.subTest(repr(value)):
                self.assertEqual(_relations.lines(value), expected)

    def test_lines_join_back(self):
        value = "one\ntwo\r\nthree"
        self.assertEqual("\n".join(_relations.lines(value)), "one\ntwo\nthree")


class TestPointer(unittest.TestCase):
    """pointer escapes '~' and '/' in a token."""

    def test_pointer(self):
        self.assertEqual(_relations.pointer("", "a/b~c"), "/a~1b~0c")
        self.assertEqual(_relations.pointer("/0", 2), "/0/2")


class TestNodeEntries(unittest.TestCase):
    """node_entries writes a row per node, in the value's order."""

    def test_rows_and_lines(self):
        value = {"a": [Number("1"), "x\ny"], "b": None}
        self.assertEqual(
            _relations.node_entries("s", [A], value, ""),
            [
                ("s", [A, "", "object", ""]),
                ("s", [A, "/a", "array", ""]),
                ("s", [A, "/a/0", "number", "1"]),
                ("s", [A, "/a/1", "string", ""]),
                ("s.value", [A, "/a/1", "0", "x"]),
                ("s.value", [A, "/a/1", "1", "y"]),
                ("s", [A, "/b", "null", ""]),
            ],
        )


class TestItemEntries(unittest.TestCase):
    """item_entries writes a definition's row and its child sheets."""

    CHILD = Shape(
        columns=("type",), lines=frozenset({"type"}), known=frozenset({"type"})
    )
    SHAPE = Shape(
        columns=("id", "score"),
        lines=frozenset({"id"}),
        nodes=("content",),
        children=(("details", CHILD),),
        known=frozenset({"id", "score", "content", "details"}),
    )

    def test_rows(self):
        value = {
            "id": "a\nb",
            "score": Number("0.950"),
            "content": "x",
            "details": {"type": "t"},
            "extra": True,
        }
        self.assertEqual(
            _relations.item_entries("r", self.SHAPE, A, "/0", value),
            [
                ("r", [A, "/0", "", "0.950"]),
                ("r.id", [A, "/0/id", "0", "a"]),
                ("r.id", [A, "/0/id", "1", "b"]),
                ("r.content", [A, "/0/content", "string", "x"]),
                ("r.details", [A, "/0/details", "t"]),
                ("r.additionalProperties", [A, "/0/extra", "true", ""]),
            ],
        )

    def test_null_field_is_absent(self):
        value = {"id": None, "content": None}
        self.assertEqual(
            _relations.item_entries("r", self.SHAPE, A, "/0", value),
            [("r", [A, "/0", "", ""])],
        )

    def test_not_an_object(self):
        self.assertEqual(_relations.item_entries("r", self.SHAPE, A, "/0", "x"), [])


class TestVariantEntries(unittest.TestCase):
    """variant_entries names a part's sheet by its type value."""

    TEXT = Shape(columns=("content",), known=frozenset({"type", "content"}))
    GENERIC = Shape(columns=("type",), known=frozenset({"type"}))
    VARIANTS = {"text": TEXT, "generic": GENERIC}

    def test_dispatch(self):
        value = [{"type": "text", "content": "hi"}, {"type": "note"}, "x", {}]
        self.assertEqual(
            _relations.variant_entries("p", self.VARIANTS, A, "/0/parts", value),
            [
                ("p.text", [A, "/0/parts/0", "hi"]),
                ("p.generic", [A, "/0/parts/1", "note"]),
                ("p.generic", [A, "/0/parts/3", ""]),
            ],
        )


class TestSheets(unittest.TestCase):
    """shape_sheets lists a definition's sheets in the spec's order."""

    def test_order(self):
        child = Shape(columns=("type",), lines=frozenset({"type"}))
        shape = Shape(
            columns=("id", "score"),
            lines=frozenset({"id"}),
            nodes=("content",),
            children=(("details", child),),
            variants=(("parts", {"generic": Shape(columns=("type",))}),),
        )
        self.assertEqual(
            [name for name, _ in _relations.shape_sheets("r", shape)],
            [
                "r",
                "r.id",
                "r.content",
                "r.content.value",
                "r.details",
                "r.details.type",
                "r.details.additionalProperties",
                "r.details.additionalProperties.value",
                "r.additionalProperties",
                "r.additionalProperties.value",
                "r.parts.generic",
                "r.parts.generic.additionalProperties",
                "r.parts.generic.additionalProperties.value",
            ],
        )


class TestAssemble(unittest.TestCase):
    """assemble writes every sheet, with or without rows."""

    def test_every_sheet(self):
        headers = [("a", ["x"]), ("b", ["y"])]
        self.assertEqual(
            _relations.assemble(headers, [("b", ["1"]), ("b", ["2"])]),
            [
                {"sheet name": "a", "header": ["x"], "records": []},
                {"sheet name": "b", "header": ["y"], "records": [["1"], ["2"]]},
            ],
        )

    def test_unknown_sheet(self):
        with self.assertRaises(KeyError):
            _relations.assemble([("a", ["x"])], [("c", ["1"])])


if __name__ == "__main__":
    unittest.main()
