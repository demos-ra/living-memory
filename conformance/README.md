# Conformance

Every implementation is tested against the same files. Each case checks
one rule of `spec/living-memory.mtsv` and is named for it: the rule's Id,
then what it asserts, as `relation.3.text`. Every expected file is derived
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
| module.1   | `module.1.title`, `module.1.pattern-subset` (every token of the subset); non-conforming `module.1.schema-not-json`, `module.1.no-title`, `module.1.property-name` (HT), `module.1.title-name` (LF), `module.1.pattern-name` (FF), `module.1.dependency-name` (CR), `module.1.ref-name` (a $ref's last token, HT), `module.1.pattern-token`, `module.1.ref-root`, `module.1.ref-loop` (two allOf's that refer to each other) |
| module.2   | `module.2.ref`, `module.2.ref-members-ignored`, `module.2.no-sheet` (not, propertyNames, definitions), `module.2.contains`, `module.2.format-not-asserted`, `module.2.content-not-asserted`, `module.2.ignored-keyword`; `order.1.keywords` (the order subschemas are taken); non-conforming `module.2.ref-outside` |
| value.1    | `value.1.order`; every conforming case of more than one value, converted in two parts: its values before the middle, then the rest, each part's sheets appended in the file's order |
| value.2    | `value.2.number-as-written`, `value.2.member-order`, `value.2.integer` |
| value.3    | non-conforming `value.3.not-json`, `value.3.invalid-utf-8`, `value.3.duplicate-names`, `value.3.schema-mismatch` (type), `value.3.maximum` (a numeric assertion), `value.3.deepest-place` |
| relation.1 | `relation.1.kind-one-sheet` (one kind, referred to twice, one sheet), `relation.1.kind-holds-itself` (a kind whose rows key into its own sheet), `relation.1.all-of-narrows`; `module.2.ref`, `module.2.ref-members-ignored` (a required $ref property is a kind, not columns), `relation.4.optional` (types listed), `relation.4.root-any` (none listed), `relation.3.any-of`, `relation.3.one-of` (the union of branches), `relation.3.any-of-types` (a $ref a branch applies, its kind's sheet) |
| relation.2 | `relation.2.object` (a required object, nested), `relation.2.all-of`; `order.3.columns` (simple domains), `relation.3.pattern-properties` (one column, value), `relation.1.all-of-narrows`, `relation.3.any-of`, `relation.3.one-of`, `relation.5.once` (a required object that allows one type, object, by allOf or the union of branches) |
| relation.3 | `relation.3.array-of-arrays`, `relation.3.root-array`, `relation.3.items-tuple` (items, additionalItems), `relation.3.pattern-properties`, `relation.3.additional-properties`, `relation.3.optional-object`, `relation.3.any-of`, `relation.3.any-of-in-items`, `relation.3.one-of`, `relation.3.if-then-else`, `relation.3.dependencies`, `relation.3.text` (a string's runs), `relation.3.empty-runs`, `relation.3.one-runs-sheet` (two strings' runs, one sheet of runs), `relation.3.any-of-types` (a location of several types placed branch by branch: a string molten, an array by its elements, an object by its kind) |
| relation.4 | `relation.4.optional`, `relation.4.additional-properties-omitted`, `relation.4.root-any` (the sheet of the input values is the sheet of instances), `relation.4.one-sheet` (two molten places, one sheet of instances); `field.1.types` (additionalProperties), `relation.3.dependencies` (an optional member), `relation.1.all-of-narrows` (additionalProperties omitted in an allOf subschema), `relation.3.any-of-types` (a branch of one simple type, a property taken as not required) |
| relation.5 | `relation.5.once` (a member two branches name), `relation.5.covers` (a member a branch's pattern matches); `relation.2.all-of` (no allOf column for a member named), `relation.3.any-of`, `relation.3.any-of-in-items` (additionalProperties takes nothing covered), `relation.3.if-then-else`, `relation.3.dependencies`, `sheet.1.untitled-branch` (a branch's column for a member the location writes, empty), `module.2.contains` |
| key.1      | `key.1.escaped-name` ('~0', '~1'); `relation.3.array-of-arrays`, `relation.3.any-of` (parent, then pointer), `order.2.nested-order` (an instance's parent the instance that holds it), `relation.1.kind-holds-itself` (parent a row of the same sheet), `relation.3.text` (a run keyed by its string's pointer) |
| order.1    | `order.1.keywords`, `relation.1.kind-one-sheet` (a kind at its first reference), `order.1.shared-last` (the sheet of instances, then the sheet of runs, last), and the order of every case with more than one keyword |
| order.2    | `order.2.nested-order`, `value.1.order`                                |
| order.3    | `order.3.columns`; `relation.2.object` (a member in its property's place), `relation.2.all-of` (allOf's after the location's own) |
| file.1     | every case: one file, named by the caller                             |
| file.2     | `file.2.empty-sheet` (a sheet with no records left out), and every expected file |
| file.3     | every expected file                                                    |
| sheet.1    | `module.1.title`; `module.2.ref` (a kind by its $ref's last token), `relation.3.items-tuple` (positions), `relation.3.pattern-properties` (one pattern), `relation.3.array-of-arrays` (items), `relation.3.any-of` (titles), `relation.3.one-of` (positions), `relation.3.dependencies` (a key), `relation.2.object` (a property that relation.2 writes, in the name), `sheet.1.runs-name` (the sheet of runs, runs), `field.1.types` (the sheet of instances, instances), `sheet.1.untitled-branch` (no position for a keyword of one subschema) |
| sheet.2    | every expected file (parent, pointer, page, line, position); `relation.2.object` (property.member) |
| record.1   | every expected file                                                    |
| field.1    | `field.1.types`, `relation.4.optional`                                 |
| field.2    | `relation.3.text`, `relation.3.empty-runs` (an empty field), `sheet.1.runs-name` (a property's member) |
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
| `module.1.ref-name`         | rejected: the module specification                        |
| `module.1.pattern-token`    | rejected: the module specification                        |
| `module.1.ref-root`         | rejected: the module specification                        |
| `module.1.ref-loop`         | rejected: the module specification                        |
| `module.2.ref-outside`      | rejected: the module specification                        |
| `value.3.not-json`          | rejected: value 1, pointer `""`                            |
| `value.3.invalid-utf-8`     | rejected: value 1, pointer `""`                            |
| `value.3.duplicate-names`   | rejected: value 0, pointer `""`                            |
| `value.3.schema-mismatch`   | rejected: value 1, pointer `"/n"`                          |
| `value.3.maximum`           | rejected: value 0, pointer `"/n"`                          |
| `value.3.deepest-place`     | rejected: value 0, pointer `"/p/k"`                        |
