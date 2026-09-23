# Conformance

Every implementation is tested against the same files. Each case is an
input, `name.jsonl`, an OpenTelemetry Protocol (OTLP) JSON Lines file, and
its expected output, `name.mtsv`. An implementation must convert the input
to MTSV sheets equal to those of the expected file. Every expected file
holds every sheet of the mapping, in the order of the working record's
Sheets, each with its header (M18).

| Folder                            | Cases                                          |
|-----------------------------------|------------------------------------------------|
| `otlp/`                           | `spans` and `logRecords`: the eight on each kind of record |
| `gen_ai.system_instructions/`     | one per sheet of the set, named as the sheet   |
| `gen_ai.tool.definitions/`        | one per sheet of the set, named as the sheet   |
| `gen_ai.input.messages/`          | one per sheet of the set, named as the sheet   |
| `gen_ai.output.messages/`         | one per sheet of the set, named as the sheet   |
| `gen_ai.tool.call.arguments/`     | one per sheet of the set, named as the sheet   |
| `gen_ai.tool.call.result/`        | one per sheet of the set, named as the sheet   |
| `gen_ai.memory.records/`          | one per sheet of the set, named as the sheet   |
| `gen_ai.retrieval.documents/`     | one per sheet of the set, named as the sheet   |
| `providers/<provider>/`           | a provider's own format, converting to the same MTSV |

`WORKING-RECORD.mtsv` lists every case, the sheet it checks and the rules
it exercises. An expected file is derived from the spec, never from an
implementation's output.
