# Living Memory

Persistent semantic memory for AI agents. SQLite-backed. Zero dependencies beyond Python stdlib.

```bash
pip install living-memory
```

```python
from living_memory import LivingMemory

with LivingMemory(db_path="agent.db") as lm:
    lm.remember("goals", subject="user", predicate="targets", object="ship LM v1")
    context = lm.recall("current goals")
```

---

## How it works

The core problem: an agent's context window resets on every call. Memory that isn't injected is gone. Naive injection — dump everything — hits the budget. Smart injection requires knowing what matters *right now*, and forgetting what doesn't.

Living Memory models this as a physical system with three dimensions.

**D1 — Permanence.** The backbone is permanent: identity facts, resolved decisions, completed work. Everything else is active memory subject to decay and compression.

**D2 — Domain.** Branches are conceptual manifolds — goals, preferences, threads, beliefs. A node lives in one branch. Retrieval scores across all branches simultaneously.

**D3 — Relevance.** Every node has a relevance score — the product of semantic similarity to the current query and its decay buoyancy. Only nodes above the retrieval threshold are injected.

These three dimensions are not separate systems. They interact: backbone anchors the relevance distribution, which sets the retrieval threshold, which determines what gets injected and what goes cold. The system is one thing, not three.

**Memory as a triple.** Every write is `subject + predicate + object_text`. Predicates are not labels — they carry physics. Each predicate has a temporality axis that sets its decay rate, a certainty axis that governs compression eligibility, and a polarity axis that governs conflict sensitivity. `completed` nodes never decay and never compress. `prefers` nodes decay fast and compress freely. The physics is in the predicate table, not in code.

**Self-organization.** Two processes keep the tree healthy without manual intervention. Branch-level OLS fires when a branch grows past φ × system mean — it merges eligible nodes (by subject+predicate group) into a centroid. Root-level OLS detects cold branches — those whose mean node relevance has fallen below the retrieval threshold — and archives them. Both thresholds emerge from the live distribution. Nothing is hardcoded.

**Decay is activity-relative, not clock-relative.** A node's inactivity is measured as a fraction of total system accesses since it was last accessed — not seconds. A node written yesterday in a dormant system stays fresh. A node written last week in a heavily-used system may be stale. This matches how memory works in practice.

**Injection renders summaries.** Retrieved nodes are grouped by `(subject, predicate)` before injection. Three `user targets` nodes become one line: `[goals] user targets: ship LM v1 | publish to npm | open source release`. One context slot, three facts.

**Multi-agent consensus.** Multiple agents writing to the backbone go through signal detection before committing. Signal = mean pairwise cosine similarity across agent versions. Threshold = SDT criterion derived from live tree distribution. Predicate divergence (agents using different predicates for the same concept) is a structural conflict flagged before signal evaluation. On commit, the backbone node inherits subject and predicate from the agreeing agents.

---

## Predicate table

Predicates define the semantic physics of memory. The 12 seed predicates cover most use cases. Unknown predicates are classified automatically on first use.

| predicate | temporality | decay_mult | compression | conflict |
|-----------|-------------|------------|-------------|---------|
| `knows` | permanent | 0.0 | conservative | low |
| `completed` | permanent | 0.0 | never | none |
| `owns` | permanent | 0.0 | conservative | low |
| `member-of` | permanent | 0.0 | conservative | low |
| `requires` | semi-permanent | 0.3 | conservative | high |
| `targets` | transient | 1.0 | eligible | medium |
| `blocked-by` | transient | 1.0 | conservative | high |
| `conflicts-with` | transient | 1.0 | never | high |
| `believes` | transient | 1.8 | eligible | medium |
| `prefers` | transient | 1.8 | eligible | low |
| `avoids` | transient | 1.8 | eligible | low |
| `tolerates` | transient | 1.8 | eligible | low |

New predicates are inserted by providing five axis values on first write. If the axis signature matches an existing predicate exactly, the write is mapped as a synonym. If it occupies unoccupied axis space, a new predicate row is inserted with computed physics.

```python
lm.remember("relationships", subject="user", predicate="trusted-by", object="Demos",
            axes={
                "polarity": "positive", "temporality": "semi-permanent",
                "directionality": "relational", "certainty": "belief", "agency": "passive"
            })
# trusted-by inserted into predicate table — used directly on subsequent writes
```

---

## API

### Write

```python
# Structured triple
lm.remember(branch, subject=str, predicate=str, object=str)

# Free-text fallback (v0.3/v0.4 compat)
lm.remember(branch, "content string")

# With options
lm.remember(branch,
    subject      = "user",
    predicate    = "targets",
    object       = "ship LM v1",
    tags         = ["sprint-1"],
    embedding_fn = my_embed_fn,   # callable(str) -> dict[str, float]
    normalize_fn = my_norm_fn,    # callable(str) -> str
    axes         = {...},         # for unknown predicates
)
```

`branch="backbone"` — permanent, no decay, no compression.

### Read

```python
# Formatted summary string — ready for context injection
context = lm.recall("current goals")
# → "=== MEMORY CONTEXT ===
#    [goals] user targets: ship LM v1 | publish to npm
#    [prefs] user prefers: dark mode
#    === END MEMORY ==="

# Raw node dicts
nodes = lm.recall_nodes("current goals")
```

### Multi-agent

```python
# Write through consensus pipeline
result = lm.contribute(branch,
    content     = "...",
    agent_id    = "grok",
    concept_key = "subject:predicate",  # groups agents writing on same concept
    stakes      = "standard",           # or "high" — scales consensus threshold
    subject     = "...",
    predicate   = "...",
    object      = "...",
)
# result["status"]: "committed" | "pending" | "conflict" | "written"
# result["predicate"]: predicate preserved from agents on commit

# Inspect and resolve conflicts
conflicts = lm.pending_conflicts()
lm.resolve_conflict(
    conflict_id = conflicts[0]["conflict_id"],
    resolved_content = "...",
    concept_key = conflicts[0]["concept_key"],
    subject     = "...",   # preserve triple structure
    predicate   = "...",
    object      = "...",
)
```

### Maintenance

```python
lm.compress()       # root-level OLS on cold branches — call at session end
lm.decay_scores()   # {node_id: relevance} for all active non-backbone nodes
lm.status()         # snapshot: node count, branch sizes, predicate count, bootstrap prior
lm.export("out.json")
```

### Predicate management

```python
lm.predicates()         # full predicate table
lm.predicate_synonyms() # logged synonym mappings

# Explicit classification
result = lm.classify_predicate("depends-on", axes={
    "polarity": "positive", "temporality": "transient",
    "directionality": "relational", "certainty": "fact", "agency": "passive"
})
# {"action": "inserted", "predicate": "depends-on", "props": {...}}
# {"action": "synonym",  "mapped_to": "requires",   "original": "depends-on"}
# {"action": "quarantine","predicate": "...",        "reason": "..."}
```

---

## Architecture

```
R0  src/config/
        memory.json          physics constants, predicate seed, injection config

R1  src/domain/
        nodes.py             Node dataclass, decay, similarity, predicate props
        retrieval.py         TF-IDF, SDT thresholds, injection, summary grouping
        compression.py       OLS — branch-level and root-level

R2  src/app/
        memory_tree.py       orchestration — wires R1 and R4, holds predicate cache

R3  src/contract/
        api.py               public surface — LivingMemory class

R4  src/exec/
        storage.py           SQLite — schema, migrations, atomic writes
```

Living Architecture R0–R4 layers. Each layer depends only on layers below it. No circular imports. R4 is pure I/O. R1 is pure logic. R3 exposes nothing from R4 directly.

---

## Physics constants

All empirically grounded. All in `memory.json` under the `physics` block with source citations.

| constant | value | source |
|----------|-------|--------|
| φ (phi) | 1.618 | Golden ratio — structural packing ratio for OLS trigger |
| λ (lambda) | 0.5 | Ebbinghaus forgetting curve — base decay rate |
| β dedup | 3.0 | SDT — false alarm (losing distinct info) 3× worse than miss |
| β retrieval | 0.3 | SDT — miss (lost context) 3× worse than false alarm |
| β consensus | 5.0 | SDT — false alarm (bad backbone fact) 5× worse than miss |
| quorum | signal | Quorum sensing — convergence measured, not voted |

---

## Migration

**v0.3 → v0.5:** Automatic. `_migrate()` adds missing columns on open. Existing data preserved. `tfidf_vector` column copied to `content_vector`.

**v0.4 → v0.5:** Automatic. Same migration. Predicate table seeded on first open. Existing nodes defaulting to `predicate='knows'` are compatible — `knows` is in the seed table.

---

## Tests

```bash
python3 test_lm.py
# 21/21 passed
```

| range | covers |
|-------|--------|
| T1–T9 | backbone permanence, branch creation, dedup, OLS, decay, injection, persistence |
| T10–T14 | export, relational activation, multi-agent convergence/conflict, emergent threshold |
| T15–T17 | predicate-aware dedup, neutral buoyancy, obj_vector dedup |
| T18–T21 | cold branch physics gate, summary groups, consensus predicate preservation, auto-classification |
