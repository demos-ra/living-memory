"""Shared helpers for the tests: the spec and the conformance files."""

from pathlib import Path

import mtsv

ROOT = Path(__file__).resolve().parents[2]
CONFORMANCE = ROOT / "conformance"
SPEC = ROOT / "spec" / "living-memory.mtsv"


def spec_sheets(prefix=""):
    """Return the spec's Sheets whose names start with a prefix.

    Each is a pair of its name and its header, in the spec's order.
    """
    with SPEC.open("rb") as file:
        spec = {sheet["sheet name"]: sheet for sheet in mtsv.load(file)}
    headers = {}
    for name, column in spec["Sheets"]["records"]:
        headers.setdefault(name, []).append(column)
    return [
        (name, header) for name, header in headers.items() if name.startswith(prefix)
    ]


def cases():
    """Return the input of every conformance case, sorted."""
    found = sorted(CONFORMANCE.glob("*/*.jsonl"))
    if not found:
        raise FileNotFoundError(CONFORMANCE)
    return found
