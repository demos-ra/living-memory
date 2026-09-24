"""The text around AI models, as MTSV sheets.

OTLP JSON Lines files of TracesData and LogsData, and the eight
structured text sets of the OpenTelemetry GenAI semantic conventions
among them, written as Multi-Sheet Tab-Separated Values (MTSV),
draft-demosra-mtsv-01. The sheets are written with mtsv.dump.
"""

__all__ = ["load", "loads", "OTLPDecodeError"]

from living_memory._otlp_json import OTLPDecodeError, load, loads
