# Living Memory (LM)

Persistent, context-aware agent memory. Built on [Living Architecture](https://github.com/your-org/living-architecture).

---

## What it does

Stores agent memory as a self-organizing tree. Memory compresses automatically, retrieves by semantic relevance, and persists across sessions indefinitely.

No time-based expiry. No manual management. No context window bloat.

---

## Core mechanics

**Three-dimensional memory space**

- D1 — Permanence axis: backbone (permanent) → active branches → cold → archive
- D2 — Domain manifolds: dynamic branches, created at runtime by the agent
- D3 — Relevance space: cosine similarity × decay buoyancy per query

**Backbone**
Permanent nodes. Never decay. Never compress. Store identity, long-term facts, stable context.

**Dynamic branches**
Created on first write. Any name. Subject to OLS compression and Ebbinghaus decay.

**OLS compression (branch-level)**
Fires when a branch exceeds φ (1.618) × mean branch size across all active branches.
Merges N nodes into 1 centroid node. Source nodes archived, never deleted.
Self-organizing — no hardcoded node counts.

**OLS compression (root-level)**
Compresses cold branches (access ratio below 10% of system mean) into single archive nodes.
Call at session boundaries.

**Ebbinghaus decay**
`relevance = base_score × e^(−λ × inactivity_ratio)`
Inactivity is activity-relative, not clock-relative. High system activity with no node access = fast decay. Low system activity = slow decay. λ = 0.5 (configurable).

**Deduplication**
On every write, raw TF cosine similarity checked against existing branch nodes.
Above 0.85 threshold: update existing node, do not insert.

**Conditional injection**
On every query, all active nodes scored by cosine similarity × decay buoyancy.
Top-N returned within character budget cap. Never loads full tree into context.

**Storage**
SQLite with WAL mode. Every write operation is atomic. Compression is a single transaction: insert compressed node + archive source nodes + write log. Crash-safe.

---

## File structure

```
living-memory/
├── src/
│   ├── config/          R0 — r-layers.json, f-tags.json, memory.json
│   ├── domain/          R1 — nodes.py, retrieval.py, compression.py
│   ├── app/             R2 — memory_tree.py
│   ├── contract/        R3 — api.py
│   └── exec/            R4 — storage.py
├── main.py
└── test_lm.py
```

Built under Living Architecture v2.0 R0–R4 layer constraints and F-tag feature manifolds.

---

## Usage

```python
from src.contract.api import LivingMemory

with LivingMemory(db_path="agent.db") as lm:

    # Permanent — never decays
    lm.remember("backbone", "User is Demos. Building LIV.", tags=["identity"])

    # Dynamic branches — created on first use
    lm.remember("goals",   "Ship LM v1.0 as standalone library.")
    lm.remember("beliefs", "Memory must be context-aware, not time-indexed.")
    lm.remember("threads", "Open: benchmark TF-IDF vs sentence-transformers.")

    # Retrieve — scores all nodes against query, returns top-N as string
    context = lm.recall("what are the current goals")

    # Raw nodes if needed
    nodes = lm.recall_nodes("open threads")

    # Current state
    status = lm.status()

    # Root-level OLS — compress cold branches (call at session boundaries)
    lm.compress()

    # JSON snapshot for inspection
    lm.export("memory_export.json")
```

---

## Configuration

`src/config/memory.json`

| Key | Default | Description |
|-----|---------|-------------|
| `compression.ols_multiplier` | 1.618 | φ — branch compression trigger ratio |
| `compression.min_nodes_to_compress` | 3 | Minimum nodes required for OLS to fire |
| `decay.lambda` | 0.5 | Ebbinghaus decay rate |
| `decay.backbone_decays` | false | Backbone nodes exempt from decay |
| `similarity.dedup_threshold` | 0.85 | Cosine sim above which incoming node is a duplicate |
| `similarity.retrieval_threshold` | 0.70 | Minimum score for injection |
| `injection.top_n` | 10 | Maximum nodes injected per query |
| `injection.budget_chars` | 4000 | Hard character cap on total injection |
| `archival.cold_branch_access_ratio` | 0.1 | Branch access ratio below which branch is cold |

---

## Tests

```bash
python test_lm.py
```

Covers: backbone permanence, dynamic branch creation, deduplication, OLS trigger (φ), OLS content preservation, Ebbinghaus gradient decay, injection budget cap, retrieval relevance ordering, SQLite session persistence, JSON export integrity.

---

## Run demo

```bash
python main.py
```

---

## Dependencies

None. Python 3.10+ standard library only.
SQLite is built into Python. No external packages required.

---

## Living Architecture layers

| Layer | Directory | Role |
|-------|-----------|------|
| R0 | `src/config` | Configuration — φ, λ, thresholds, budgets |
| R1 | `src/domain` | Pure logic — node structure, TF-IDF, OLS, decay |
| R2 | `src/app` | Orchestration — tree management, pipeline wiring |
| R3 | `src/contract` | Public API surface |
| R4 | `src/exec` | I/O — SQLite, JSON export |

F-tags: `F-memory`, `F-compression`, `F-retrieval`, `F-storage`
