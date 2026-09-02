# Phase E — Ledger Reference (read this first)

**Purpose:** Memory refresh for humans and agents. Single-page summary of all Phase E
planning decisions. Details in [ledger-design.md](./ledger-design.md) and
[ledger-record-model.md](./ledger-record-model.md).

**Status:** E1–E5 implemented · **Not wired to ModelController** · Production path still
`TrainingSession.train_and_apply()` / `ModelController.fit()`.

**Last updated:** 2026-09-02

---

## TL;DR

The ledger is an **append-only log of documents**. Each batch step pushes small
documents (command, grads, metrics); checkpoints push large documents (full
weights + Adam state). Rewind = load a checkpoint document. Branch = fork at a
checkpoint, freeze the old path, try new settings. One **single-thread main loop**
(SQL-style) — no worker threads for v1. Swap file/SQLite/Mongo by implementing
`LedgerStore.push` / `get` / `scan`.

---

## What problem this solves

Product goal: understand **overfit / underfit / memorization**, especially when
data grows slowly. When training starts to overfit halfway through a path:

1. **Keep** the good prefix on the tape (immutable).
2. **Step back** to `last_healthy_version` (checkpoint document).
3. **Fork** a new branch with tweaked settings.
4. **Freeze** the bad tail on the old branch for audit.

Later: MLP + CNN + attention share **path metadata** (`path.record` documents),
not weights.

---

## Terminology

| Term | Meaning |
|------|---------|
| **Step** | One batch = one `train_step()` |
| **LSN** | Monotonic document index on the tape |
| **Version** | Model state after optimizer apply |
| **Document** | One append unit: envelope + typed `body` |
| **Checkpoint** | `doc_type: checkpoint` — full weights + Adam |
| **Branch** | Lineage from a fork; shared prefix, own head |
| **batch_id** | UUID per fetch — **canonical** batch identity |

---

## All decisions (Q1–Q6)

| # | Question | Decision |
|---|----------|----------|
| Q1 | Version granularity | **Hybrid:** LSN every document; version per consolidate (`consolidate_every`, default 1) |
| Q2 | Rewind / branch | Immutable tape; rewind to **last healthy**; fork; freeze bad tail; read-only replay OK |
| Q3 | Checkpoint policy | Full snapshot every **N** steps + **local best val** + **fork** |
| Q4 | Tape vs checkpoint bodies | **Grads + metrics** every step; **full weights + Adam** only at checkpoint events |
| Q5 | Batch identity | **`batch_id` UUID** at `next_batch()`; epoch/batch_idx are hints only |
| Q6 | Integration | **`TrainingEngine` + tests first**; no `ModelController` wire until E gates pass |

**Core abstraction:** ledger = documents. Tensors live **inside** document bodies.

---

## Architecture

```text
TrainingRun
  LedgerStore          ← interface: push / get / scan / put_checkpoint
  TrainingLedger       ← LSN, branches, replay/rewind
  TrainingManager      ← overfit policy, fork, handoff
  TrainingEngine       ← single-thread main loop (NEW, E3)
    └── TrainingSession.train_step / apply_step  (EXISTS, Phase C)
```

### Files to create (implementation)

| File | Phase | Role | Status |
|------|-------|------|--------|
| `src/ledger.py` | E1–E2 | `LedgerDocument`, `LedgerStore`, `FileLedgerStore`, `TrainingLedger` | **done** |
| `src/training_engine.py` | E3 | Main loop: `run_step` / `run_steps` | **done** |
| `src/training_manager.py` | E6 | last_healthy, local_best, fork policy | planned |
| `testing/test_ledger.py` | E1+ | Round-trip, replay, fork tests | **done** |

### Files that exist today (do not break)

| File | Role |
|------|------|
| `src/training_session.py` | `TrainStepResult`, `train_step`, `apply_step`, `fit` |
| `src/controller.py` | Delegates to `TrainingSession.fit()` |
| `testing/test_training_session.py` | C gates + E-bridge sequential parity |

---

## LedgerStore interface

```python
class LedgerStore(Protocol):
    def push(self, doc: LedgerDocument) -> int: ...              # → LSN
    def get(self, lsn: int) -> LedgerDocument: ...
    def scan(self, from_lsn: int, to_lsn: int | None = None) -> Iterator[LedgerDocument]: ...
    def head_lsn(self) -> int: ...
    def put_checkpoint(self, doc: LedgerDocument) -> None: ...
    def get_checkpoint(self, branch_id: str, version: int) -> LedgerDocument | None: ...
```

v1 backend: `FileLedgerStore`.

---

## Document types (quick ref)

### Envelope (every document)

`lsn`, `branch_id`, `model_instance_id`, `architecture_id`, `doc_type`, `version?`, `step_id?`, `body`

### Per batch (4 documents)

| doc_type | When | Body highlights |
|----------|------|-----------------|
| `step.command` | Before compute | `batch_ref.batch_id` (UUID), lr, base_version |
| `step.result` | After train_step | grads, loss → `TrainStepResult` |
| `step.consolidated` | After apply_step | version, optimizer_t |
| `step.metrics` | After apply | train/val loss, verdict HEALTHY/SUSPECT/OVERFIT |

### Episodic

| doc_type | When |
|----------|------|
| `checkpoint` | Every N, local best, fork |
| `branch.fork` | Manager forks |
| `rewind` | Explicit rewind marker |
| `path.record` | Cross-model handoff (no tensors) |

### batch_ref schema

```yaml
batch_id:       "<uuid>"      # canonical — assign at next_batch()
data_version:   1             # bump when dataset grows
epoch:          3             # hint only
batch_idx:      17            # hint only
```

---

## Config (planned)

```yaml
ledger:
  consolidate_every: 1
  checkpoint_every_steps: 50
  checkpoint_on_local_best: true
  checkpoint_on_fork: true
  keep_last_k_checkpoints: 20
```

---

## Main loop (single thread)

```text
TRAIN_BATCH   → next_batch() assigns batch_id UUID → train_step → push command + result
CONSOLIDATE   → apply_step → push step.consolidated → version++
EVALUATE      → push step.metrics → update last_healthy / local_best
CHECKPOINT    → if N or local_best or fork → put_checkpoint
DECIDE        → if overfit → fork from last_healthy_version
```

No threads. Multiple models later = round-robin **slots** in this loop.

---

## Overfit → fork flow (example)

```text
main:   steps 1 … 420 (healthy) … 500 (overfit detected)
        └── frozen; tail 421–500 kept for product narrative

retry:  fork at checkpoint 420 → new lr → steps 421′ …
```

Checkpoint at 420 must include **weights + Adam m/v/t** at 420 (not head at 500).

---

## Replay recipes

**Roll forward A→B:** load checkpoint ≤ A → scan tape → apply step.result bodies via `apply_step`.

**Rewind + branch:** load checkpoint at `last_healthy_version` → push `branch.fork` → continue on new branch.

**Read-only:** load checkpoint N; do not push; do not move head.

---

## Training manager — this phase

**Early stopping is the manager for now.** Existing `TrainingSession.fit()` already:

- Tracks best validation epoch
- Stops when patience expires
- **Rolls back** weights to the best checkpoint (in RAM today)

Phase E does **not** add a separate `TrainingManager` with auto-fork / overfit
policies yet. The ledger must **support** that same story:

- `checkpoint` documents at **local best val** = the rewind target early stopping uses
- Tape keeps the full path; rollback = load that checkpoint document

**Deferred (post–function-proper):** E6 auto-fork, SUSPECT/OVERFIT verdict policies,
multi-model handoff. Get ledger + early-stop rollback aligned first.

---

## Implementation phases & exit criteria

| ID | Deliverable | Status |
|----|-------------|--------|
| E1 | `LedgerStore` + `LedgerDocument` + doc_type schemas | done |
| E2 | `TrainStepResult` serialize round-trip | done |
| E3 | `TrainingEngine` main loop | done |
| E4 | `FileLedgerStore` + checkpoint sidecars | done |
| E5 | Rewind + fork + read-only replay tests | done |
| E6′ | Ledger ↔ early-stop rollback (local-best checkpoint) | **next** |
| E7+ | Multi-slot, path.record, auto-fork policy | deferred |

**Exit this phase:** Ledger replay works; local-best checkpoint matches early-stop
restore; no new policy layer until that is solid.

~~**Exit v2:** Synthetic overfit triggers fork.~~ Deferred.

---

## Never on the ledger

Scratch buffers, activations, dataset blobs, DLL handles, global backend singleton.

---

## Related docs

| Doc | Contents |
|-----|----------|
| [ledger-design.md](./ledger-design.md) | Full architecture, product goals, multi-model |
| [ledger-record-model.md](./ledger-record-model.md) | Per-doc body schemas, timeline examples |
| [engine-roadmap.md](./engine-roadmap.md) | Phase map A–E |

---

## Decisions log (canonical)

| Date | Decision |
|------|----------|
| 2026-09-02 | Ledger = append-only documents; pluggable `LedgerStore` |
| 2026-09-02 | Step = batch; LSN fine-grained; version via `consolidate_every` |
| 2026-09-02 | Rewind to last healthy; fork; freeze bad tail |
| 2026-09-02 | Checkpoints: every N + local best + fork |
| 2026-09-02 | Single-thread main loop (SQL-style) |
| 2026-09-02 | Q4: grads on tape; full weights at checkpoints only |
| 2026-09-02 | Q5: batch_id UUID; epoch/batch_idx hints |
| 2026-09-02 | Q6: TrainingEngine + tests before ModelController |
| 2026-09-02 | **This phase:** early stopping = manager (stop + rollback to best); defer E6 auto-fork |
