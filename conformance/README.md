# Conformance

Every implementation is tested against the same files. Each case checks
one rule of `spec/living-memory.mtsv` and is named for it: the rule's Id,
then what it asserts, as `field.3.text`. Every expected file is derived
from the rules by hand, never from an implementation's output.

## Files of a case

| File                   | Holds                                                        |
|------------------------|--------------------------------------------------------------|
| `name.jsonl`           | the input values, in order, one JSON text per line           |
| `name.schema.json`     | the schema the module specification supplies (module.1)      |
| `name.pointer.json`    | the JSON Pointer, as a JSON string, of the member that selects the file, where the case names one (module.2) |
| `name.mtsv`            | the expected file, the caller naming it `name`; with a pointer, `name.VALUE.mtsv` for each value of the member (file.1) |

## Folders: what a converter must do

| Folder                   | A converter must                                             |
|--------------------------|--------------------------------------------------------------|
| `conforming/`            | write exactly the expected files                             |
| `cannot-be-represented/` | write exactly the expected files, and report as not carried the pointers listed below (field.4) |
| `non-conforming/`        | reject the module specification or the input, naming what is listed below (module.1, value.3) |

## Rules and their cases, in the spec's order

| Rule     | Cases                                                              |
|----------|--------------------------------------------------------------------|
| module.1 | `module.1.title`, `sheet.2.pattern-properties` (a pattern in the subset); non-conforming `module.1.schema-name`, `module.1.pattern-token`, `module.3.ref-outside` |
| module.2 | `module.2.file-member`                                             |
| module.3 | `module.3.ignored-keyword`, `module.3.ref`, `module.3.no-sheet` (not, propertyNames, definitions), `module.3.contains`, `module.3.format-not-asserted` |
| value.1  | `value.1.order`                                                    |
| value.2  | `value.2.number-as-written`, `value.2.member-order`, `value.2.integer` |
| value.3  | non-conforming `value.3.not-json`, `value.3.invalid-utf-8`, `value.3.duplicate-names`, `value.3.schema-mismatch` (type), `value.3.maximum` (a numeric assertion), `value.3.missing-member` |
| file.1   | `module.2.file-member` (two files, named by the value); every other case (one file, named by the caller) |
| file.2   | `file.2.empty-sheet`                                               |
| file.3   | `file.3.order`, and the order of every case with more than one keyword |
| file.4   | every expected file                                                |
| sheet.1  | `module.1.title`                                                   |
| sheet.2  | `file.3.order` (object, arrays), `sheet.2.array-of-arrays`, `sheet.2.root-array`, `sheet.2.optional-object`, `sheet.2.items-tuple` (items, additionalItems), `sheet.2.pattern-properties`, `sheet.2.additional-properties-omitted` |
| sheet.3  | `sheet.3.any-of`, `sheet.3.any-of-in-items`, `sheet.3.one-of`, `sheet.3.all-of`, `sheet.3.if-then-else`, `sheet.3.dependencies` |
| sheet.4  | `sheet.4.optional`, `value.2.member-order` (additionalProperties), `sheet.3.dependencies` (an optional member), `sheet.4.root-any` |
| sheet.5  | `sheet.5.once` (a member two branches name), `sheet.3.any-of` and `sheet.3.any-of-in-items` (additionalProperties takes nothing named), `sheet.3.all-of`, `sheet.3.if-then-else`, `sheet.3.dependencies` (no column for what the location names), `module.3.contains` |
| record.1 | every expected file                                                |
| record.2 | `value.1.order`, `record.2.nested-order`                           |
| record.3 | `file.3.order`, `sheet.2.array-of-arrays`, `sheet.3.any-of`         |
| field.1  | `field.1.columns`, `value.2.integer`                               |
| field.2  | `field.2.types`, `sheet.4.optional`                                |
| field.3  | `field.3.text`, `field.3.empty-runs`                               |
| field.4  | cannot-be-represented `field.4.lone-cr`, `field.4.surrogate`, `field.4.member-name` |

## Expected reports and rejections

Values are counted from 0; pointers are written as JSON strings (field.4, value.3).

| Case                            | Expected                                   |
|---------------------------------|--------------------------------------------|
| `field.4.lone-cr`               | not carried: `"/0/t"`                      |
| `field.4.surrogate`             | not carried: `"/0/t"`                      |
| `field.4.member-name`           | not carried: `"/0/a\tb"`                   |
| `module.1.schema-name`          | rejected: the module specification         |
| `module.1.pattern-token`        | rejected: the module specification         |
| `module.3.ref-outside`          | rejected: the module specification         |
| `value.3.not-json`              | rejected: value 1, pointer `""`            |
| `value.3.invalid-utf-8`         | rejected: value 1, pointer `""`            |
| `value.3.duplicate-names`       | rejected: value 0, pointer `""`            |
| `value.3.schema-mismatch`       | rejected: value 1, pointer `"/n"`          |
| `value.3.maximum`               | rejected: value 0, pointer `"/n"`          |
| `value.3.missing-member`        | rejected: value 1, pointer `"/s"`          |
