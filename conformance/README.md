# Conformance

Every implementation is tested against the same files. Each case is an
input, `name.jsonl`, an OpenTelemetry Protocol (OTLP) JSON Lines file, and
its expected output, `name.mtsv`. An implementation must convert the input
to MTSV sheets equal to those of the expected file.

| Folder                  | Cases for                                            |
|-------------------------|------------------------------------------------------|
| `m1/` … `m17/`          | the mapping rule of the same number in the spec      |
| `providers/<provider>/` | a provider's own format, converting to the same MTSV |

No cases are written yet: an expected file needs the names of the key and
position columns, which the spec leaves open (M16).
