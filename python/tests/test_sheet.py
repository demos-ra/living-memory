"""Tests of _sheet: each sheet's name and header."""

import unittest

from living_memory import _key
from living_memory import _sheet as module
from living_memory._relation import Domain, Relation, Segment

ROOT = Relation(Segment("root", (), "t"), _key.ROOT, (), ())


def below(segment: Segment, *domains: Domain) -> Relation:
    return Relation(segment, _key.SUBORDINATE, domains, ())


class TestName(unittest.TestCase):
    # sheet.1-4: the root by its title, a kind and the sheets of
    # instances and runs by their own names, every other by its
    # parent's name, the properties between them, and its own name.
    def test_names(self):
        cases = [
            (Segment("property", ("p",), "l"), "t.p.l"),
            (Segment("keyword", (), "patternProperties", "^x"), "t.patternProperties"),
            (Segment("keyword", (), "items", 1, several=True), "t.items.1"),
            (Segment("branch", (), "anyOf", 0, "a"), "t.anyOf.a"),
            (Segment("branch", (), "oneOf", 1, several=True), "t.oneOf.1"),
            (Segment("branch", (), "if"), "t.if"),
            (Segment("branch", (), "dependencies", "m"), "t.dependencies.m"),
            (Segment("kind", (), "pair"), "pair"),
            (Segment("instances", (), "_instances"), "_instances"),
            (Segment("runs", (), "_runs"), "_runs"),
        ]
        self.assertEqual(module.name((ROOT,)), "t")
        for segment, expected in cases:
            with self.subTest(segment=segment):
                self.assertEqual(module.name((ROOT, below(segment))), expected)

    # sheet.4: a branch of a location placed branch by branch is named
    # after that location.
    def test_branch_within(self):
        location = Segment("property", (), "p")
        branch = Segment("branch", (), "anyOf", 1, several=True, within=location)
        self.assertEqual(module.name((ROOT, below(branch))), "t.p.anyOf.1")

    # sheet.4: a kind's subordinate sheet is named after the kind.
    def test_below_a_kind(self):
        kind = below(Segment("kind", (), "pair"))
        path = (ROOT, kind, below(Segment("property", (), "l")))
        self.assertEqual(module.name(path), "pair.l")

    # sheet.3: a name composed of the schema's names that begins with
    # '_' is written with that '_' doubled, once.
    def test_doubled(self):
        root = Relation(Segment("root", (), "_t"), _key.ROOT, (), ())
        self.assertEqual(module.name((root,)), "__t")
        path = (root, below(Segment("property", (), "_l")))
        self.assertEqual(module.name(path), "__t._l")
        kind = below(Segment("kind", (), "_k"))
        self.assertEqual(module.name((root, kind)), "__k")


class TestHeader(unittest.TestCase):
    # sheet.3, sheet.5: the key columns by their names, then each domain
    # by its name, a name beginning with '_' doubled, one the
    # specification fixes as it is.
    def test_header(self):
        one = below(
            Segment("property", (), "o"),
            Domain("p.b", "value"),
            Domain("_c", "value"),
            Domain("_value", "value", True),
        )
        self.assertEqual(
            module.header(one),
            [
                "_input value",
                "_instance",
                "_parent",
                "_pointer",
                "p.b",
                "__c",
                "_value",
            ],
        )


if __name__ == "__main__":
    unittest.main()
