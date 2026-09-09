# Ledger Document Model — What Lives at Each Point

Companion to [ledger-design.md](./ledger-design.md). **Memory refresh:**
[phase-e-reference.md](./phase-e-reference.md).

**The ledger is a document log.** Each point on the tape is one document: a typed
envelope plus a body. Replay, rewind, and branching are operations over that
sequence of documents — not over raw tensors exposed to the rest of the engine.

This doc defines **document types** and **body schemas** (Phase E Q4).

---

## 1. Document envelope (every point)

Every append, regardless of type:

```yaml
lsn:                1042              # assigned by store.push()
branch_id:          "main"
model_instance_id:  "cnn-0"
architecture_id:    "cnn_v1"
doc_type:           step.result       # schema selector for body
version:            420               # when applicable
step_id:            420               # when applicable
body:               { ... }           # type-specific payload
```

The training loop and manager only talk in documents: `push(doc)`,
`scan(from_lsn)`, `get_checkpoint(branch, version)`.

---

## 2. Three document classes

| Class | `doc_type` examples | When | Purpose |
|-------|---------------------|------|---------|
| **Step documents** | `step.command`, `step.result`, `step.consolidated`, `step.metrics` | Every batch | The training story, grad apply, health |
| **Checkpoint documents** | `checkpoint` | Every N, local best, fork | Rewind anchors (large body) |
| **Control documents** | `branch.fork`, `rewind`, `path.record` | Manager acts | Branching, handoff, verdicts |

Nothing is deleted when you rewind or fork — you append new documents. Older
documents stay on the tape for audit and product narrative.

---

## 3. Q4 decision — what goes in the body

**Step documents every batch; checkpoint documents only on checkpoint triggers.**

| Document | Body holds |
|----------|------------|
| `step.command`, `step.result`, `step.metrics` | Every batch |
| `checkpoint` | Every N steps, local best val, fork |
| Full weights every batch | **No** — too heavy; use checkpoint docs |

Grad arrays live **inside** the `step.result` document body (serialized). That is
still a document — not a separate storage concept.

---

## 4. One batch step — document sequence

Each batch appends **up to four documents** (in order):

### 4.1 `step.command`

Written **before** compute.

```yaml
doc_type: step.command
body:
  step_id:          420
  base_version:     419
  batch_ref:
    batch_id:       "f47ac10b-58cc-4372-a567-0e02b2c3d479"   # UUID, canonical
    data_version:   1
    epoch:          3                                         # hint only
    batch_idx:      17                                        # hint only
  m_samples:        32
  lr:               0.001
  scheduler_epoch:  3
```

### 4.2 `step.result`

Written **after** `train_step`, **before** optimizer apply. Body maps to
`TrainStepResult` in `src/training_session.py`.

```yaml
doc_type: step.result
body:
  step_id:          420
  loss:             0.0842
  m_samples:        32
  grad_weights:     [...]                 # serialized in body
  grad_biases:      [...]
  grad_gammas:      [...] | null
  grad_betas:       [...] | null
```

### 4.3 `step.consolidated`

Written **after** `apply_step()`.

```yaml
doc_type: step.consolidated
body:
  step_id:          420
  version:          420
  optimizer_t:      420
```

### 4.4 `step.metrics`

```yaml
doc_type: step.metrics
body:
  step_id:          420
  version:          420
  train_loss:       0.0842
  val_loss:         0.0911
  train_val_gap:    0.0069
  verdict:          HEALTHY
  is_local_best_val: false
```

---

## 5. Checkpoint document (`doc_type: checkpoint`)

Same document model; larger **body**. Written on checkpoint triggers (every N,
local best val, fork). May live in a sidecar file keyed by `(branch_id, version)`
but is still one logical document.

```yaml
doc_type: checkpoint
body:
  version:          420
  weights:          [...]
  biases:           [...]
  gammas / betas:   ... | null
  optimizer:        { type: Adam, t: 420, ms_w: [...], vs_w: [...], ... }
  val_loss:         0.0911
  is_local_best:    true
```

Rewind = load this document and hydrate the model.

---

## 6. Control documents (episodic)

### 6.1 `branch.fork`

```yaml
doc_type: branch.fork
body:
  parent_branch_id: "main"
  parent_version:   420
  parent_lsn:       1045
  new_branch_id:    "retry_lr_half"
  reason:           "overfit_onset"
  settings_delta:
    lr:             0.0005
    weight_decay:   0.001
  checkpoint_version: 420
```

### 6.2 `rewind`

```yaml
doc_type: rewind
body:
  branch_id:        "main"
  from_version:     500
  to_version:       420
  reason:           "last_healthy"
```

### 6.3 `path.record` (cross-model, no tensors)

```yaml
doc_type: path.record
body:
  source_branch_id: "main"
  source_version_range: [421, 500]
  fork_version:     420
  verdict:          OVERFIT
  recommender:      "cnn-0"
  architecture_id:  "cnn_v1"
  metrics_summary:
    best_val_version: 420
    best_val_loss:  0.0911
    onset_version:  480
  settings_tried:
    lr:             0.001
  handoff_hint:     "try lower lr or more data before step 400"
```

Portable to MLP/attention instances; no weight blobs in the body.

---

## 7. Timeline example

One batch at step 420 (checkpoint trigger: local best val):

```text
LSN 1041  step.command       batch_id=f47ac10b-..., epoch=3, batch_idx=17
LSN 1042  step.result        grads + loss
LSN 1043  step.consolidated  version=420
LSN 1044  step.metrics       verdict=HEALTHY, is_local_best_val=true
          ── checkpoint sidecar: version 420 (weights + Adam) ──
```

Step 421 (normal, no checkpoint):

```text
LSN 1045  step.command
LSN 1046  step.result
LSN 1047  step.consolidated  version=421
LSN 1048  step.metrics       verdict=SUSPECT
```

Step 500 (overfit detected → fork back to 420):

```text
LSN 1200  step.metrics       verdict=OVERFIT
LSN 1201  rewind             to_version=420
LSN 1202  branch.fork        new_branch=retry_lr_half
          ── checkpoint 420 already on disk ──
```

Branch `main` tape: steps 1–500 intact. Branch `retry`: continues from checkpoint 420.

---

## 8. What never goes on the ledger

| Excluded | Reason |
|----------|--------|
| ScratchArena buffers (`col`, gemm buf) | Ephemeral; reallocated per step |
| ForwardCache / activations | Recomputed on replay if needed; too large |
| Full training dataset | Referenced by `batch_ref` + `data_version` |
| ctypes / DLL handles | Process-local |
| Global backend singleton state | Session uses `EngineContext` |

---

## 9. Size and retention (implementation notes)

| Artifact | Rough scale (CNN tiny) | Retention |
|----------|----------------------|-----------|
| `TrainStepResult` | ~same order as grad tensors | Keep on tape (branch history) |
| `MetricRecord` | tens of bytes | Keep on tape |
| `CheckpointSnapshot` | 2× model params (weights + Adam) | `keep_last_k_checkpoints`; always keep local-best per branch |

Future compaction: shared checkpoint blobs keyed by `hash(weights)` when branches
fork from the same version (copy-on-write at storage layer).

---

## 10. Replay and rewind recipes

### Roll forward (version A → B)

1. Load nearest checkpoint ≤ A.
2. Scan tape: apply `TrainStepResult` via `apply_step` for versions (A, B].
3. Verify `ConsolidatedVersion` chain.

### Rewind and branch

1. Find `last_healthy_version` from `MetricRecord` / manager state.
2. Load `CheckpointSnapshot` at that version.
3. Append `BranchFork`; set active head to new branch.
4. Continue appending new steps on new branch.

### Read-only inspect version N

1. Load checkpoint at N (or replay if no checkpoint — slow path).
2. Do not append; do not mutate head.

---

## 11. Serialization format (E2)

| Record | Format (v1) |
|--------|-------------|
| `TrainStepResult` | msgpack or custom binary: header + numpy `.tobytes()` per tensor + dtype/shape |
| Other tape records | msgpack / JSON for debuggability |
| Checkpoints | `.npz` sidecar per `(branch_id, version)` or single structured blob |

Exact codec is E2 implementation detail; schema fields above are the contract.

---

## 12. Decisions log

| Date | Decision |
|------|----------|
| 2026-09-02 | Q4: grads + metrics on tape every step; full weights + Adam only at checkpoints |
| 2026-09-02 | Command/result/consolidate/metric = four-record step pattern |
| 2026-09-02 | PathRecord for cross-model handoff; no tensors |
| 2026-09-02 | Q5: `batch_id` UUID per fetch; epoch/batch_idx are hints only |
