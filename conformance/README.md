# Conformance

Every implementation is tested against the same files. Each case checks
one rule of `spec/living-memory.mtsv` and is named for it: the rule's Id,
then what it asserts, as `field.2.text`. Every expected file is derived
from the rules by hand, never from an implementation's output.

## Files of a case

| File               | Holds                                                    |
|--------------------|----------------------------------------------------------|
| `name.jsonl`       | the input values, in order, one JSON text per line       |
| `name.schema.json` | the schema the module specification supplies (module.1)  |
| `name.mtsv`        | the expected file, the caller naming it `name` (file.1)  |

## Folders: what a converter must do

| Folder                   | A converter must                                                  |
|--------------------------|-------------------------------------------------------------------|
| `conforming/`            | write exactly the expected file                                   |
| `cannot-be-represented/` | write exactly the expected file, and report as not carried the pointers listed below (field.3) |
| `non-conforming/`        | reject the module specification or the input, naming what is listed below (module.1, module.2, value.3) |

## Rules and their cases, in the spec's order

A case named for another rule also checks the rule it is listed under.

| Rule       | Cases                                                                  |
|------------|------------------------------------------------------------------------|
| module.1   | `module.1.title`, `module.1.pattern-subset` (every token of the subset); non-conforming `module.1.schema-not-json`, `module.1.no-title`, `module.1.property-name` (HT), `module.1.title-name` (LF), `module.1.pattern-name` (FF), `module.1.dependency-name` (CR), `module.1.pattern-token` |
| module.2   | `module.2.ref`, `module.2.ref-members-ignored`, `module.2.no-sheet` (not, propertyNames, definitions), `module.2.contains`, `module.2.format-not-asserted`, `module.2.content-not-asserted`, `module.2.ignored-keyword`; non-conforming `module.2.ref-outside` |
| value.1    | `value.1.order`                                                        |
| value.2    | `value.2.number-as-written`, `value.2.member-order`, `value.2.integer` |
| value.3    | non-conforming `value.3.not-json`, `value.3.invalid-utf-8`, `value.3.duplicate-names`, `value.3.schema-mismatch` (type), `value.3.maximum` (a numeric assertion), `value.3.deepest-place` |
| relation.1 | `relation.1.all-of-narrows`; `relation.4.optional` (types listed), `relation.4.root-any` (none listed), `relation.3.any-of`, `relation.3.one-of` (the union of branches) |
| relation.2 | `relation.2.object` (a required object, nested), `relation.2.all-of`; `order.3.columns` (simple domains), `relation.3.pattern-properties` (one column, value) |
| relation.3 | `relation.3.array-of-arrays`, `relation.3.root-array`, `relation.3.items-tuple` (items, additionalItems), `relation.3.pattern-properties`, `relation.3.additional-properties`, `relation.3.optional-object`, `relation.3.any-of`, `relation.3.any-of-in-items`, `relation.3.one-of`, `relation.3.if-then-else`, `relation.3.dependencies` |
| relation.4 | `relation.4.optional`, `relation.4.additional-properties-omitted`, `relation.4.root-any`; `field.1.types` (additionalProperties), `relation.3.dependencies` (an optional member) |
| relation.5 | `relation.5.once` (a member two branches name); `relation.2.all-of` (no allOf column for a member named), `relation.3.any-of`, `relation.3.any-of-in-items` (additionalProperties takes nothing named), `relation.3.if-then-else`, `relation.3.dependencies` (no column for what the location names), `module.2.contains` |
| key.1      | `key.1.escaped-name` ('~0', '~1'); `order.1.keywords`, `relation.3.array-of-arrays`, `relation.3.any-of` (the parent's key copied down) |
| order.1    | `order.1.keywords`, and the order of every case with more than one keyword |
| order.2    | `order.2.nested-order`, `value.1.order`                                |
| order.3    | `order.3.columns`; `relation.2.object` (a member in its property's place), `relation.2.all-of` (allOf's after the location's own) |
| file.1     | every case: one file, named by the caller                             |
| file.2     | `file.2.empty-sheet`                                                   |
| file.3     | every expected file                                                    |
| sheet.1    | `module.1.title`; `relation.3.items-tuple` (positions), `relation.3.pattern-properties` (one pattern), `relation.3.array-of-arrays` (items), `relation.3.any-of` (titles), `relation.3.one-of` (positions), `relation.3.dependencies` (a key), `relation.2.object` (a property that relation.2 writes, in the name), `field.2.text` (a column's name) |
| sheet.2    | every expected file; `relation.2.object` (property.member)             |
| record.1   | every expected file                                                    |
| field.1    | `field.1.types`, `relation.4.optional`                                 |
| field.2    | `field.2.text`, `field.2.empty-runs`                                   |
| field.3    | cannot-be-represented `field.3.lone-cr`, `field.3.surrogate`, `field.3.member-name` (HT, LF, FF, CR) |

## Expected reports and rejections

Values are counted from 0; pointers are written as JSON strings (field.3, value.3).

| Case                        | Expected                                                   |
|-----------------------------|------------------------------------------------------------|
| `field.3.lone-cr`           | not carried: `"/0/t"`                                      |
| `field.3.surrogate`         | not carried: `"/0/t"`                                      |
| `field.3.member-name`       | not carried: `"/0/a\tb"`, `"/0/c\nd"`, `"/0/e\ff"`, `"/0/g\rh"` |
| `module.1.schema-not-json`  | rejected: the module specification                        |
| `module.1.no-title`         | rejected: the module specification                        |
| `module.1.property-name`    | rejected: the module specification                        |
| `module.1.title-name`       | rejected: the module specification                        |
| `module.1.pattern-name`     | rejected: the module specification                        |
| `module.1.dependency-name`  | rejected: the module specification                        |
| `module.1.pattern-token`    | rejected: the module specification                        |
| `module.2.ref-outside`      | rejected: the module specification                        |
| `value.3.not-json`          | rejected: value 1, pointer `""`                            |
| `value.3.invalid-utf-8`     | rejected: value 1, pointer `""`                            |
| `value.3.duplicate-names`   | rejected: value 0, pointer `""`                            |
| `value.3.schema-mismatch`   | rejected: value 1, pointer `"/n"`                          |
| `value.3.maximum`           | rejected: value 0, pointer `"/n"`                          |
| `value.3.deepest-place`     | rejected: value 0, pointer `"/p/k"`                        |
