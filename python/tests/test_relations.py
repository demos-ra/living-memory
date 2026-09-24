"""Tests of _relations: how a value becomes keyed sheets."""

import unittest

from living_memory import _relations
from living_memory._json import Number
from living_memory._relations import Definition, Key, Variants

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


class TestLineRows(unittest.TestCase):
    def test_lines_split_at_lf_and_crlf_only(self):
        cases = [
            ("first\nsecond", ["first", "second"]),
            ("first\r\nsecond", ["first", "second"]),
            ("a\n", ["a", ""]),
            ("a\rb", []),
            ("x", []),
            (None, []),
        ]
        for value, expected in cases:
            with self.subTest(repr(value)):
                found = _relations.line_rows("s", Key((A,)), value)
                self.assertEqual([row[-1] for _, row in found], expected)

    def test_positions(self):
        self.assertEqual(
            _relations.line_rows("s", Key((A,)), "one\ntwo"),
            [("s", [A, "0", "one"]), ("s", [A, "1", "two"])],
        )


class TestKey(unittest.TestCase):
    def test_extend_extends_the_last_field(self):
        self.assertEqual(Key((A, "/0")).extend("a/b"), Key((A, "/0/a~1b")))
        self.assertEqual(Key((A,)).extend(2), Key((A + "/2",)))


class TestNodeRows(unittest.TestCase):
    def test_a_row_per_node_in_order(self):
        value = {"a": [Number("1"), "x\ny"], "b": None, "c": True}
        self.assertEqual(
            _relations.node_rows("s", Key((A, "")), value),
            [
                ("s", [A, "", "object", ""]),
                ("s", [A, "/a", "array", ""]),
                ("s", [A, "/a/0", "number", "1"]),
                ("s", [A, "/a/1", "string", "x\ny"]),
                ("s.value", [A, "/a/1", "0", "x"]),
                ("s.value", [A, "/a/1", "1", "y"]),
                ("s", [A, "/b", "null", ""]),
                ("s", [A, "/c", "boolean", "true"]),
            ],
        )


class TestArrayRows(unittest.TestCase):
    CHILD = Definition(
        columns=("type",), lines=frozenset({"type"}), properties=frozenset({"type"})
    )
    DEFINITION = Definition(
        columns=("id", "score"),
        lines=frozenset({"id"}),
        nodes=("content",),
        children=(("details", CHILD),),
        properties=frozenset({"id", "score", "content", "details"}),
    )

    def rows(self, value):
        return self.DEFINITION.array_rows("r", Key((A, "")), value)

    def test_rows(self):
        item = {
            "id": "a\nb",
            "score": Number("0.950"),
            "content": "x",
            "details": {"type": "t"},
            "extra": True,
        }
        self.assertEqual(
            self.rows([item]),
            [
                ("r", [A, "/0", "a\nb", "0.950"]),
                ("r.id", [A, "/0/id", "0", "a"]),
                ("r.id", [A, "/0/id", "1", "b"]),
                ("r.content", [A, "/0/content", "string", "x"]),
                ("r.details", [A, "/0/details", "t"]),
                ("r.additionalProperties", [A, "/0/extra", "boolean", "true"]),
            ],
        )

    def test_null_is_a_node_and_absent_is_empty(self):
        self.assertEqual(
            self.rows([{"content": None}, {}]),
            [
                ("r", [A, "/0", "", ""]),
                ("r.content", [A, "/0/content", "null", ""]),
                ("r", [A, "/1", "", ""]),
            ],
        )

    def test_not_an_array_or_object(self):
        self.assertEqual(self.rows("x"), [])
        self.assertEqual(self.rows(["x"]), [])


class TestVariantRows(unittest.TestCase):
    TEXT = Definition(columns=("content",), properties=frozenset({"type", "content"}))
    GENERIC = Definition(columns=("type",), properties=frozenset({"type"}))
    VARIANTS = Variants(
        {"text": TEXT, "generic": GENERIC},
        lambda name, item: isinstance(item.get("content"), str),
    )

    def test_placed_by_type_value_if_it_belongs(self):
        value = [
            {"type": "text", "content": "hi"},
            {"type": "text", "content": {}},
            {"type": "note"},
            "x",
            {},
        ]
        self.assertEqual(
            self.VARIANTS.array_rows("p", Key((A, "/0/parts")), value),
            [
                ("p.text", [A, "/0/parts/0", "hi"]),
                ("p.generic", [A, "/0/parts/1", "text"]),
                (
                    "p.generic.additionalProperties",
                    [A, "/0/parts/1/content", "object", ""],
                ),
                ("p.generic", [A, "/0/parts/2", "note"]),
                ("p.generic", [A, "/0/parts/4", ""]),
            ],
        )


class TestSheets(unittest.TestCase):
    def test_order(self):
        child = Definition(columns=("type",), lines=frozenset({"type"}))
        parts = Variants(
            {"generic": Definition(columns=("type",))}, lambda name, item: False
        )
        definition = Definition(
            columns=("id", "score"),
            lines=frozenset({"id"}),
            nodes=("content",),
            children=(("details", child),),
            variants=(("parts", parts),),
        )
        self.assertEqual(
            [name for name, _ in definition.sheets("r")],
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
    def test_every_sheet(self):
        headers = [("a", ["x"]), ("b", ["y"])]
        self.assertEqual(
            _relations.assemble(headers, [("b", ["1"]), ("b", [""])]),
            (
                [
                    {"sheet name": "a", "header": ["x"], "records": []},
                    {"sheet name": "b", "header": ["y"], "records": [["1"], [""]]},
                ],
                set(),
            ),
        )

    def test_fields_left_empty(self):
        headers = [("s", ["k"])]
        for value in ("a\tb", "a\nb", "a\fb", "a\rb"):
            with self.subTest(repr(value)):
                sheets, left = _relations.assemble(headers, [("s", [value])])
                self.assertEqual(sheets[0]["records"], [[""]])
                self.assertEqual(left, {"s.k"})

    def test_left_behind(self):
        headers = [("s", ["k", "v"]), ("s.v", ["line", "value"])]
        rows = [
            ("s", ["a\tb", "one\ntwo"]),
            ("s.v", ["0", "one"]),
            ("s.v", ["1", "t\fwo"]),
            ("s", ["k", "a\rb"]),
        ]
        sheets, left = _relations.assemble(headers, rows)
        self.assertEqual(sheets[0]["records"], [["", ""], ["k", ""]])
        self.assertEqual(left, {"s.k", "s.v", "s.v.value"})

    def test_unknown_sheet(self):
        with self.assertRaises(KeyError):
            _relations.assemble([("a", ["x"])], [("c", ["1"])])


if __name__ == "__main__":
    unittest.main()
