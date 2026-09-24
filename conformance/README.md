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

A rule with no case of its own is asserted by the sheet cases it
governs:

| Rule         | Asserted by                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `resource.2` | `resourceSpans.resource.entityRefs.idKeys`                                  |
| `scope.1`    | `resourceSpans.scopeSpans`, `resourceSpans.scopeSpans.scope`                |
| `span.1`     | `resourceSpans.scopeSpans.spans.status`, `…spans.events`, `…spans.links`    |
| `set.2`      | `gen_ai.input.messages`, `gen_ai.output.messages`                           |
| `item.2`     | every case whose name begins `gen_ai.`                                      |
| `field.1`    | `gen_ai.input.messages` (role), `gen_ai.input.messages.parts.blob.modality` |
| `nested.1`   | `gen_ai.input.messages.parts.server_tool_call.server_tool_call`             |
| `nested.3`   | every sheet case, each named by the sheet it fills                          |
| `node.2`     | every case whose name ends `.additionalProperties`                          |
