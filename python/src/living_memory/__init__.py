"""JSON described by a JSON Schema, converted to MTSV files.

The specification is spec/living-memory.mtsv; MTSV is
draft-demosra-mtsv-01.
"""

__all__ = ["convert", "NonConformingError", "NonConformingInputError"]

from living_memory._converter import (
    NonConformingError,
    NonConformingInputError,
    convert,
)
