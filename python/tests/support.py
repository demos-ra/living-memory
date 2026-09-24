"""Shared helpers for the tests: the spec and the conformance files."""

from pathlib import Path

import mtsv

ROOT = Path(__file__).resolve().parents[2]
CONFORMANCE = ROOT / "conformance"
SPEC = ROOT / "spec" / "living-memory.mtsv"


def spec_sheets(prefix=""):
    with SPEC.open("rb") as file:
        spec = {sheet["sheet name"]: sheet for sheet in mtsv.load(file)}
    headers = {}
    for name, column in spec["Sheets"]["records"]:
        headers.setdefault(name, []).append(column)
    return [
        (name, header) for name, header in headers.items() if name.startswith(prefix)
    ]


def cases(folder):
    found = sorted((CONFORMANCE / folder).glob("*.jsonl"))
    if not found:
        raise FileNotFoundError(CONFORMANCE / folder)
    return found
