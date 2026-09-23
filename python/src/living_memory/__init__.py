"""The text around AI models, as MTSV sheets.

OTLP JSON Lines files of TracesData and LogsData, and the eight
structured text sets of the OpenTelemetry GenAI semantic conventions
among them, written as Multi-Sheet Tab-Separated Values (MTSV),
draft-demosra-mtsv-01. The sheets are written with mtsv.dump.

Functions:
load -- read MTSV sheets from a binary OTLP JSON Lines file
loads -- read MTSV sheets from an OTLP JSON Lines string

Subpackages:
integrations -- which reader reads which input
"""

__all__ = ["load", "loads"]

from typing import Any, BinaryIO

from living_memory import integrations


def load(fp: BinaryIO, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from a binary OTLP JSON Lines file.

    fp -- a binary file object open for reading, by position only

    Return every sheet of the output. Raise ValueError for a line that
    is not JSON.
    """
    return integrations.load(integrations.JSONL, fp)


def loads(s: str, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from an OTLP JSON Lines string.

    s -- the text of the file, by position only

    Return every sheet of the output. Raise ValueError for a line that
    is not JSON.
    """
    return integrations.lookup(integrations.JSONL).loads(s)
