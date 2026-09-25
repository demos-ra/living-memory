"""Tests of _relations: a schema and its instances as related sheets."""

import unittest

from living_memory import _relations as module
from living_memory._json import decode
from living_memory._json_pointer import PlacedError


def convert(schema: str, *values: str) -> tuple[list[dict], list[str]]:
    layout = module.layout(decode(schema.encode()))
    records, reports = [], []
    for position, value in enumerate(values):
        found, missed = module.records(layout, decode(value.encode()), position)
        records += found
        reports += missed
    return module.sheets(layout, records), reports


def shape(sheets: list[dict]) -> list[tuple[str, list[str]]]:
    return [(sheet["sheet name"], sheet["header"]) for sheet in sheets]


class TestLayout(unittest.TestCase):
    # file.2, file.3: every sheet is written, in the order of the
    # schema's tree, a sheet with no records included.
    def test_sheets_in_order(self):
        schema = (
            '{"title":"t","type":"object","required":["a"],'
            '"properties":{"a":{"type":"string"},"o":{"type":"object",'
            '"additionalProperties":false},"n":{"type":"number"}}}'
        )
        sheets, _ = convert(schema, '{"a":"x"}')
        expected = [
            ("t", ["pointer", "a"]),
            ("t.a", ["t.pointer", "page", "line", "position", "value"]),
            ("t.o", ["t.pointer", "pointer"]),
            ("t.n", ["t.pointer", "pointer", "type", "value"]),
            ("t.n.value", ["t.n.pointer", "page", "line", "position", "value"]),
            ("t.additionalProperties", ["t.pointer", "pointer", "type", "value"]),
            (
                "t.additionalProperties.value",
                ["t.additionalProperties.pointer", "page", "line", "position", "value"],
            ),
        ]
        self.assertEqual(shape(sheets), expected)

    # module.1: the root schema holds a title, and names are text a
    # field can hold; a failing name is placed by its pointer in the
    # schema, through a $ref as well.
    def test_names_a_field_cannot_hold(self):
        cases = [
            ('{"type":"object"}', ""),
            ('{"title":"a\\tb","type":"object"}', "/title"),
            (
                '{"title":"t","type":"object","definitions":{"O":{"type":"object",'
                '"properties":{"a\\nb":{}}}},"properties":{"o":{"$ref":"#/definitions/O"}}}',
                "/definitions/O/properties/a\nb",
            ),
            (
                '{"title":"t","type":"object","anyOf":[{"type":"object","title":1}]}',
                "/anyOf/0/title",
            ),
        ]
        for schema, expected in cases:
            with self.subTest(schema=schema):
                with self.assertRaises(PlacedError) as raised:
                    module.layout(decode(schema.encode()))
                self.assertEqual(raised.exception.pointer, expected)


class TestRecords(unittest.TestCase):
    # record.3: the parent's key is copied down, then the record's own
    # pointer follows.
    def test_keys_copied_down(self):
        schema = (
            '{"title":"t","type":"object","additionalProperties":false,'
            '"properties":{"m":{"type":"array","items":{"type":"number"}}}}'
        )
        sheets, _ = convert(schema, '{"m":[5,6]}')
        self.assertEqual(
            sheets[1]["records"], [["/0", "/0/m/0", "5"], ["/0", "/0/m/1", "6"]]
        )

    # sheet.5: a member two collecting branches name is written by the
    # first; its field in the later one is empty.
    def test_each_value_once(self):
        schema = (
            '{"title":"t","type":"object","additionalProperties":false,'
            '"properties":{"p":{"anyOf":['
            '{"title":"a","type":"object","properties":{"k":{"type":"string"}},'
            '"required":["k"]},'
            '{"title":"b","type":"object","properties":{"k":{"type":"string"},'
            '"m":{"type":"number"}},"required":["k","m"],'
            '"additionalProperties":false}]}},"required":["p"]}'
        )
        sheets, _ = convert(schema, '{"p":{"k":"x","m":1}}')
        by_name = {sheet["sheet name"]: sheet["records"] for sheet in sheets}
        self.assertEqual(by_name["t.p.additionalProperties"], [])
        self.assertEqual(by_name["t.p.anyOf.a"], [["/0/p", "/0/p", "x"]])
        self.assertEqual(by_name["t.p.anyOf.a.additionalProperties"], [])
        self.assertEqual(by_name["t.p.anyOf.b"], [["/0/p", "/0/p", "", "1"]])

    # sheet.4, field.2-4: a value is written to a sheet of instances, a
    # string to its string's sheet, and what is not carried is reported
    # by its pointer.
    def test_instances_and_text(self):
        schema = '{"title":"t"}'
        sheets, reports = convert(schema, '{"a\\tb":"x\\ny\\rz"}')
        self.assertEqual(
            sheets[0]["records"],
            [["/0", "object", ""], ["/0/ab", "string", ""]],
        )
        self.assertEqual(
            sheets[1]["records"],
            [["/0/ab", "0", "0", "0", "x"], ["/0/ab", "0", "1", "0", "yz"]],
        )
        self.assertEqual(reports, ["/0/a\tb", "/0/ab"])


if __name__ == "__main__":
    unittest.main()
