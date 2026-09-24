"""The text around AI models, as MTSV sheets.

OTLP JSON Lines files of TracesData and LogsData, and the eight
structured text sets of the OpenTelemetry GenAI semantic conventions
among them, written as Multi-Sheet Tab-Separated Values (MTSV),
draft-demosra-mtsv-01. The sheets are written with mtsv.dump.
"""

__all__ = ["load", "loads", "OTLPDecodeError"]

from typing import Any, BinaryIO

from living_memory import integrations
from living_memory.integrations.otlp_json import OTLPDecodeError


def load(fp: BinaryIO, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from a binary OTLP JSON Lines file.

    Raise OTLPDecodeError for a file that is not OTLP JSON Lines.
    """
    return integrations.load(integrations.JSONL, fp)


def loads(s: str, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from an OTLP JSON Lines string.

    Raise OTLPDecodeError for a file that is not OTLP JSON Lines.
    """
    return integrations.lookup(integrations.JSONL).loads(s)
