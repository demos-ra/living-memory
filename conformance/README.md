# Conformance

Every implementation is tested against the same files. Each case is an
input, `name.jsonl`, an OpenTelemetry Protocol (OTLP) JSON Lines file,
and, where the input is converted, its expected output, `name.mtsv`. A
converter conforms when it writes the sheets of every expected file and
rejects every input it must reject (`spec/living-memory.mtsv` ›
Conformance). Every expected file holds every sheet of the Sheets, in
their order, each with its header (file.4).

| Folder                   | Cases                                                        |
|--------------------------|--------------------------------------------------------------|
| `conforming/`            | inputs a converter must convert, with their expected output  |
| `cannot-be-represented/` | inputs holding text MTSV cannot represent (character.1), with their expected output |
| `non-conforming/`        | inputs a converter must reject (file.5, set.3)               |

A case is named for what it checks: a sheet case by the sheet of the
Sheets it fills, as `resourceSpans.scopeSpans.spans.attributes`, and a
rule case by its rule and assertion, as `file.1.metrics-data`.

An expected file is derived from the spec's rules, never from an
implementation's output.
