# Conformance

Every implementation is tested against the same files. Each case is an
input, `name.jsonl`, an OpenTelemetry Protocol (OTLP) JSON Lines file, and
its expected output, `name.mtsv`. An implementation must convert the input
to MTSV sheets equal to those of the expected file.

| Folder                  | Cases                                                  |
|-------------------------|--------------------------------------------------------|
| `otlp/`                 | one per reading rule of the spec: `m1` … `m5`, `m17`   |
| `system-instructions/`  | one per sheet of the set                               |
| `tool-definitions/`     | one per sheet of the set                               |
| `input-messages/`       | one per sheet of the set                               |
| `output-messages/`      | one per sheet of the set                               |
| `tool-call-arguments/`  | one per sheet of the set                               |
| `tool-call-result/`     | one per sheet of the set                               |
| `memory-records/`       | one per sheet of the set                               |
| `retrieval-documents/`  | one per sheet of the set                               |
| `providers/<provider>/` | a provider's own format, converting to the same MTSV   |

A folder is the attribute key without `gen_ai.`, its dots and underscores
as hyphens; a case is named after the sheet it checks. `WORKING-RECORD.mtsv`
lists every case, the sheet it checks and the rules it exercises.

An expected file is derived from the spec, never from an implementation's
output. No cases are written yet.
