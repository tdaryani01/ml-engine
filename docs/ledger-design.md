# Training Ledger Design (Phase E)

Design doc for the append-only training ledger: replay, rewind, branching, and
the SQL-style single-thread training loop. Captures decisions from the Phase E
planning discussion (2026-09-02).

**Related:** [phase-e-reference.md](./phase-e-reference.md) (read first) ·
[engine-roadmap.md](./engine-roadmap.md) Phase E · `src/training_session.py`
(`TrainStepResult`, grad/apply split)

---

## 1. Product goals

The engine supports a training manager that understands **overfit**, **underfit**,
and **memorization** — not just loss curves. Target scenarios include:

- Data arrives slowly or grows over time (incremental collection).
- Training may go down a path that starts healthy then diverges; the system should
  **keep the good prefix**, **step back** to a last-healthy point, and **try a
  different route** (new hyperparameters, regularization, schedule).
- Failed paths stay **frozen on the tape** for audit and product explanation
  (“this lr memorized after step 420”).
- Longer term: **MLP + CNN + attention** (and other blocks) complement each other;
  one model’s “bad path” metadata can inform another model’s next attempt (policy
  handoff, not necessarily weight transfer).

The ledger is the **storage engine**. The training manager is the **planner**
that reads metrics, chooses rewind targets, forks branches, and assigns work.

### Core abstraction: the ledger keeps documents

At the highest level, the ledger is an **append-only log of documents**. Each
point on the tape is one document: a typed, serializable blob with metadata
(`lsn`, `branch_id`, `doc_type`, …). Training steps, metrics, checkpoints,
forks — all are documents.

```text
LSN 1   →  document { type: "step.command", ... }
LSN 2   →  document { type: "step.result",  ... }
LSN 3   →  document { type: "step.metrics", ... }
LSN 50  →  document { type: "checkpoint",   ... }   # larger payload, same idea
LSN 51  →  document { type: "branch.fork",  ... }
```

Implementation detail (numpy bytes, `.npz`, msgpack) lives **inside** the
document body. The engine reasoned about **documents**; swap file / SQLite /
Mongo / S3 by implementing the same push/get interface. See
[ledger-record-model.md](./ledger-record-model.md) for document schemas.

---

## 2. Terminology

| Term | Meaning |
|------|---------|
| **Step** | One minibatch: one `train_step(X, y)` (forward + backward). Same as `step_id` in `TrainStepResult`. |
| **Epoch** | One full pass over the data provider; many steps. Used for schedulers/logging, not the primary rewind unit. |
| **LSN** | Log sequence number: monotonic append index. **Every document** gets the next LSN. |
| **Document** | One append unit on the tape: typed payload + metadata (`doc_type`, `branch_id`, …). |
| **Version** | Materialized model state after optimizer apply (weights + optimizer state at a point in time). |
| **Checkpoint** | A `checkpoint` document (or linked blob): full snapshot at a version. Rewind anchor. |
| **Branch** | A lineage of training from a fork LSN/version. Immutable prefix shared with parent; own head. |
| **Head** | Current LSN/version on the active branch where new steps append. |
| **PathRecord** | Model-agnostic summary of a trajectory (metrics, settings, verdict) for cross-model handoff. |

### Version granularity (decision: hybrid)

- **Tape:** always fine-grained (one WAL entry per batch step).
- **Version:** bumps when the consolidator applies optimizer update(s).
- **Config `consolidate_every`:** default `1` (one version per batch). Set to `N` to
  bump version every N batches; replay still works at LSN granularity.

---

## 3. Architecture

```
TrainingRun (one dataset / customer session)
  ├── LedgerStore          ← pluggable persistence (file, DB, cache)
  ├── TrainingLedger       ← LSN, branches, replay/rewind API
  ├── PathRegistry         ← branch metadata, verdicts, cross-model handoff
  └── TrainingManager      ← single-thread main loop + policies
        └── slot(s)        ← (model_instance_id, branch_id, state)
```

### Database analogy

| Database | Training equivalent |
|----------|---------------------|
| WAL redo log | Append-only tape: commands, results, metrics |
| LSN | Monotonic record index |
| Checkpoint | Full weight + optimizer snapshot |
| Roll-forward | Replay tape from checkpoint → target version |
| Roll-back / rewind | Restore checkpoint; optionally fork new branch |
| Savepoint | Local-best or last-healthy checkpoint |
| Transaction | One batch step (command → result → consolidate) |
| Query planner | Training manager (overfit detect, rewind, fork, assign) |

### Threading model (decision: single main loop)

No extra worker threads required for v1. One main loop (like a SQL executor)
cycles states: train batch → evaluate → consolidate → checkpoint → decide
(rewind/fork/handoff). Multiple model instances later = **round-robin slots** in
the same loop, not shared mutable model state.

---

## 4. `LedgerStore` interface

The store appends and reads **documents**. Backend is pluggable (file, SQLite,
Mongo, Redis, S3, …) — same as swapping a database driver.

```python
class LedgerStore(Protocol):
    def push(self, doc: LedgerDocument) -> int: ...       # → LSN
    def get(self, lsn: int) -> LedgerDocument: ...
    def scan(self, from_lsn: int, to_lsn: int | None = None) -> Iterator[LedgerDocument]: ...
    def head_lsn(self) -> int: ...
    def put_checkpoint(self, doc: LedgerDocument) -> None: ...   # type=checkpoint
    def get_checkpoint(self, branch_id: str, version: int) -> LedgerDocument | None: ...
```

`LedgerDocument` = envelope (`lsn`, `branch_id`, `doc_type`, `model_instance_id`,
`architecture_id`, `body`). The **body** schema depends on `doc_type`; see
[ledger-record-model.md](./ledger-record-model.md).

**v1 implementation:** `FileLedgerStore` — one JSON/msgpack file per document
or append-only journal with document boundaries.

---

## 5. Document types

Each `doc_type` is a schema for the document **body**. Checkpoints are documents
too — just larger bodies, often stored in a sidecar for efficiency.

### Every step (small documents on the tape)

| Type | Payload | Purpose |
|------|---------|---------|
| `step.command` | `step_id`, `base_version`, `batch_ref`, `lr`, … | Intent |
| `step.result` | grads, loss | Apply via consolidator |
| `step.metrics` | train/val loss, verdict | Manager policy |

`verdict`: `HEALTHY` | `SUSPECT` | `OVERFIT`

### Consolidation

| Type | Payload |
|------|---------|
| `step.consolidated` | `version`, `step_id`, `optimizer_t` |

### Branching / control

| Type | Payload |
|------|---------|
| `branch.fork` | parent version/lsn, new branch, settings_delta |
| `rewind` | from/to version, reason |
| `path.record` | verdict, metrics summary, handoff hint |

### Checkpoints (heavy — separate from log stream or checkpoint index)

`CheckpointSnapshot`:

- `version`, `lsn`, `branch_id`
- weights, biases, gammas, betas (as applicable)
- optimizer state (Adam `m`, `v`, `t`)
- optional: val_loss at checkpoint time

---

## 6. Rewind, branch, and “keep the good prefix”

Example: steps 1–420 healthy, 421–500 overfit detected at head.

```
Branch "main":  [1 … 420 … 500]  ← frozen, tail preserved for audit
Branch "retry": [1 … 420] → 421′ …  ← active, new settings from 420
```

**Actions:**

1. Manager sets `last_healthy_version` (e.g. 420) from metrics + local-best tracking.
2. **Freeze** branch `main` at head 500 (tape immutable; nothing deleted).
3. **Fork** at version 420 → new branch with tweaked hyperparameters.
4. Restore checkpoint at 420 (weights **and** optimizer state at that point).

**Read-only replay:** load checkpoint N into a working model without changing head
(inspection, inference at a point in time).

**Truncate (optional):** active head on a branch moves back; entries after rewind
point remain on tape but are not on the active lineage (archive semantics).

You cannot know which step will become a branch point until a path fails — hence
checkpoint policy below.

---

## 7. Checkpoint policy (decision)

| Trigger | Full snapshot (weights + optimizer) |
|---------|-------------------------------------|
| Every **N** steps | Yes (config `checkpoint_every_steps`) |
| **Local best val** | Yes (`checkpoint_on_local_best: true`) |
| **Fork / rewind** | Yes (`checkpoint_on_fork: true`) |

**Metrics** every step on tape (cheap). **Full checkpoints** only on triggers above.

Suggested config (in `config/runtime.yaml` or dedicated `ledger` section):

```yaml
ledger:
  consolidate_every: 1
  checkpoint_every_steps: 50
  checkpoint_on_local_best: true
  checkpoint_on_fork: true
  keep_last_k_checkpoints: 20
```

Rationale:

- **Every N** caps worst-case rewind distance if overfit onset is detected late.
- **Local best** captures rewind targets that do not fall on N boundaries (e.g.
  best val at step 437 when N=100).

Early stopping today restores best weights from RAM; with this policy, that
checkpoint already exists on disk and can become a branch point.

**Record-level detail:** see **[ledger-record-model.md](./ledger-record-model.md)**.

---

## 8. Tape vs checkpoints (Q4 — decided)

**Grad WAL on the tape every step; full weights + Adam state only at checkpoint
events.** See [ledger-record-model.md](./ledger-record-model.md) for per-record
schemas and timeline examples.

Recompute-from-batch-indices remains a future optimization; Q5 decided (UUID).

---

## 9. Training manager and multi-model handoff

### Same architecture, multiple instances

Portable: checkpoint, hyperparams, metrics, branch_id. Another instance can:

- Take over a branch from a checkpoint.
- Retry a path another instance abandoned.
- Leave the failed branch frozen.

### Different architectures (MLP, CNN, attention)

Weights are **not** portable. Portable via `PathRecord`:

- Where training went wrong (version/LSN range).
- Settings tried and verdict (`OVERFIT`, etc.).
- Data version / sample count at fork.

Manager assigns the next attempt to the appropriate model; complementary models
share **training intelligence**, not tensors.

Record `model_instance_id` and `architecture_id` on every ledger entry from day
one.

---

## 10. Batch identity (Q5 — decided)

`(epoch, batch_idx)` is **not unique** — it repeats every epoch and across reruns,
branches, and growing datasets. Each materialized minibatch gets a **`batch_id`**
(UUID) when fetched.

```yaml
batch_ref:
  batch_id:       "f47ac10b-58cc-4372-a567-0e02b2c3d479"   # canonical identity
  data_version:   1                                         # bumps when dataset grows
  epoch:          3                                         # hint / UI only
  batch_idx:      17                                        # hint / UI only
```

- **`batch_id`**: assigned at `next_batch()` time; never reused on the tape.
- **`data_version`**: increment when new samples are added.
- **epoch / batch_idx**: context for humans and logs; not used for replay equality.

---

## 11. Integration entry point (Q6 — decided)

**Build `TrainingEngine` + tests first.** Do not wire `ModelController` /
`run_pipeline.py` until ledger replay passes in isolation.

| Milestone | Delivers |
|-----------|----------|
| E1–E3 | `src/ledger.py`, `src/training_engine.py`, `testing/test_ledger.py` |
| Later | Optional: `ModelController.fit` → `TrainingEngine.run()` |

`TrainingSession.fit()` + `train_and_apply()` stay the production path until E
gates green.

---

## 12. Main loop (single thread)

```text
while running:
    for slot in manager.active_slots():   # one slot in v1
        match slot.state:
            TRAIN_BATCH:
                X, y, batch_ref = provider.next_batch()   # batch_ref.batch_id = new UUID
                cmd = step.command doc(..., batch_ref)
                result = session.train_step(X, y, lr)
                ledger.push(cmd)
                ledger.push(step.result doc)

            CONSOLIDATE:
                apply_step(result) → version++
                ledger.push(step.consolidated doc)

            EVALUATE:
                ledger.push(step.metrics doc); update last_healthy / local_best

            CHECKPOINT:
                if triggers met: ledger.put_checkpoint(...)

            DECIDE:
                if overfit policy fires:
                    fork from last_healthy_version

            VALIDATE / FIT_EPOCH:
                (existing epoch val pass — feeds step.metrics)
```

Eventually `TrainingEngine.run()` owns this loop; `TrainingSession.fit()` delegates
after integration.

---

## 13. Implementation phases

| ID | Scope | Notes |
|----|--------|-------|
| **E1** | `LedgerStore` ABC + record schemas + `TrainStepCommand` | `src/ledger.py` |
| **E2** | `TrainStepResult.to_bytes()` / `from_bytes()` | Round-trip tests |
| **E3** | `TrainingLedger` + single-thread consolidator / main loop | Replaces threaded E3 in roadmap |
| **E4** | `FileLedgerStore` + checkpoint files | Crash recovery, replay |
| **E5** | Rewind + fork + read-only replay | Branch tests |
| **E6** | `TrainingManager` policy hooks (local best, last healthy) | Wire to early-stop semantics |
| **E7** | Multi-slot / PathRecord / cross-model handoff | After v1 proof |
| **E8** | VM transport doc | Redis/SQS/Kafka TBD |

### Exit criteria (revised)

1. **E v1:** Replay from checkpoint matches sequential `train_and_apply` for N
   steps (same seed, one branch).
2. **E v1.5:** Rewind to local-best checkpoint + fork produces expected branch
   isolation on tape.
3. **E v2:** Manager detects synthetic overfit curve and forks; frozen branch
   retained.

---

## 14. Open questions

All Phase E planning questions resolved. See decisions log.

---

## 15. Decisions log

| Date | Decision |
|------|----------|
| 2026-09-02 | Step = batch; LSN fine-grained; version configurable via `consolidate_every` |
| 2026-09-02 | Immutable tape + rewind to last healthy + fork; preserve bad tail on frozen branch |
| 2026-09-02 | Checkpoints: every N steps + local best val + fork events |
| 2026-09-02 | Single-thread SQL-style main loop; multi-model via slots later |
| 2026-09-02 | Ledger = append-only **documents**; `LedgerStore.push` / `get` / `scan` |
| 2026-09-02 | Q4: grads + metrics on tape; full snapshots at checkpoint events — [ledger-record-model.md](./ledger-record-model.md) |
| 2026-09-02 | Q5: `batch_id` UUID per fetch; epoch/batch_idx hints only |
| 2026-09-02 | Q6: `TrainingEngine` + tests first; no `ModelController` wire until E gates pass |
