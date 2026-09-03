# Contract-List Architecture (Phase F)

**Milestone (F8):** one compiled contract, one native run — full training step
(forward + backward + apply) with zero Python between ops. Grad parity + benchmark.

**Status:** F0 done · working toward F8  
**Last updated:** 2026-09-02  
**Related:** [phase-e-reference.md](./phase-e-reference.md), [engine-roadmap.md](./engine-roadmap.md)

---

## Milestone definition (F8)

| Criterion | Target |
|-----------|--------|
| Entry | Manager → CNN `add_training_step` → one contract submit |
| Execution | Native runs entire op list; OMP stays hot inside one call |
| Python | No per-layer `conv_dispatch` on hot path |
| Correctness | Grad check parity vs sync path |
| Perf | Less idle OMP gap vs current layer-by-layer Python |
| Flag | `contract_list_enabled: true` (CNN only); sync remains default |

Steps F1–F7 below are **build order toward F8**, not separate deliverables.

---

## Problem

Main thread does real work between native calls (graph walk, cache, dense head, ledger).
OMP workers spin idle in those gaps (~ms per layer). Adding Python `ThreadPoolExecutor`
threads fights the fixed 4-thread OMP model (1 master + 3 workers).

**Goal:** keep OMP workers fed; main/manager stays thin; roll-forward stays honest via
`version` on successful steps only.

---

## Layer responsibilities

| Layer | Role | Knows about |
|-------|------|-------------|
| **TrainingManager** (main loop) | Push work, ES/policy, ledger, val mode, yield/retry | version, ledger, batch schedule — **not** conv |
| **CNN orchestrator** | Training queue, capacity/pushback, compile & submit contract lists, react to completion | graph, queue depth — **not** ledger |
| **Native executor** | Run compiled contract list (opcodes + handles), report OK/FAIL | ops, buffers, OMP — **not** policy, **not** ledger |

### Manager ↔ CNN handshake

```
Manager → CNN: "add training step" (batch, step_id hint)
CNN     → OK (accepted) | BUSY (queue full — drain first)
Manager → if BUSY: do other work (ledger tick, val, yield), retry later
```

Bounded queue so manager gets timely ES feedback — don't schedule far ahead of ground truth.

### CNN ↔ Native handshake

```
CNN     → submit ContractList (compiled at init, batch handles bound per step)
Native  → runs full list without interpreting English; one completion event
CNN     → acts on completion (queue state, notify manager)
```

No blocking wait on native. Completion via **shared-memory ring** (preferred) or callback.

### Version semantics

- **Success:** apply → `version++` (weight timeline)
- **Failure:** record on tape (`step.failed`), **no version**, no apply — rerunnable later
- Batch order doesn't matter (shuffled epochs); **version order** is the contract

### Val / inference

Another **manager loop state** — forward-only contract list, same machinery.

---

## Contract list (SQL analogy)

- **Writer:** CNN at init — compiles spatial_pipeline + dense head into opcodes + buffer handles
- **Executor:** Native — pointer-chasing kernel dispatch, tile queue inside OMP
- **Default:** one list per full training step (forward + backward + adam) — avoid re-queue gaps
- **Option (later):** chained lists (fwd → bwd) if native gets same-step priority — not v1

Adam may live inside the step contract or as a separate list CNN chains — CNN decides;
native doesn't care.

---

## What changes vs today

| Today | Target |
|-------|--------|
| Spatial layers call `conv_dispatch` sync per layer | Emit nothing at runtime; contract compiled at init |
| Python between every conv | Zero Python between ops inside native |
| `ThreadPoolExecutor` worker | **Remove** — OMP pool is the worker pool |
| Ledger in training hot path | Manager only; CNN reports OK/FAIL up |
| Version every step | Version only on successful apply |

---

## Sync default (Phase F bridge)

- **Sync `train_step` / `train_and_apply`** remains the only training path until F8.
- **`contract_list_enabled: false`** (CNN only) — opt-in when contract path is proven.
- **No Python `ThreadPoolExecutor`** — removed F0 (wrong OMP model).
- Remove sync CNN path only after F8 grad + benchmark parity; no permanent dual-path flag.

---

## Implementation steps (granular)

Confirm each group before coding. One PR per letter-group where possible.

### F0 — Doc & cleanup (this file + remove wrong path)

- [x] **F0.1** Land this design doc; link from `phase-e-reference.md`
- [x] **F0.2** Remove `ThreadPoolExecutor` from `FifoTrainQueue` / async path (wrong OMP model)
- [x] **F0.3** Sync default; `contract_list_enabled` flag; remove `async_compute` / `manager_tick_every`

### F1 — Contract IR (init-time, compiled once)

- [ ] **F1.1** Define `ContractOp` enum + fixed struct layout (opcode, in/out handles, offsets, flags)
- [ ] **F1.2** Define `ContractList` header (graph_id, op_count, buffer table offset)
- [ ] **F1.3** Define handle/buffer table (weight ptr, scratch pool id, batch X/y ptr)
- [ ] **F1.4** Python: `compile_graph(spatial_pipeline, dense_head) → ContractList` at CNN init
- [ ] **F1.5** Test: compile existing 2-conv synthetic CNN; assert op count/order matches manual trace

### F2 — Native dumb executor

- [ ] **F2.1** C++: `run_contract_list(ContractList*, batch_handles*) → status` — single entry, no return until list done (OMP stays hot **inside** native)
- [ ] **F2.2** C++: dispatch table — conv_fwd, conv_bwd, relu, pool, flatten, dense_fwd, dense_bwd, adam_apply
- [ ] **F2.3** C++: non-blocking submit API — `submit_contract(...)` returns immediately; runs on OMP master thread or internal queue (TBD perf)
- [ ] **F2.4** Test: run single-op list vs existing sync ctypes — numerical parity

### F3 — Buffer pool (native-owned, init once)

- [ ] **F3.1** C++: size-bucket pool (linked lists per size class)
- [ ] **F3.2** Pull / return API; background or on-return zeroing
- [ ] **F3.3** Wire pool handles into contract buffer table at compile time
- [ ] **F3.4** Test: no leak across N submit/complete cycles

### F4 — Completion ring

- [ ] **F4.1** Shared-memory ring: `{ step_id, status, loss, error_code, grad_handles? }`
- [ ] **F4.2** Native writes slot on complete; CNN/manager drains on tick
- [ ] **F4.3** Test: submit → drain → exactly one completion event

### F5 — CNN queue + pushback

- [ ] **F5.1** `CnnTrainingQueue`: accept / busy based on depth cap (configurable)
- [ ] **F5.2** On accept: bind batch handles to contract template, submit to native
- [ ] **F5.3** On completion drain: OK → notify manager; FAIL → notify manager (no version)
- [ ] **F5.4** Test: pushback when queue full; manager retry accepts after drain

### F6 — CNN orchestrator refactor (defer compute)

- [ ] **F6.1** `spatial_layers` / `ConvBlock`: stop calling `conv_dispatch` on training path
- [ ] **F6.2** Keep layer objects as **spec sources** for compile only (geometry, param indices)
- [ ] **F6.3** `cnn_network`: `add_training_step(batch) → OK|BUSY`; completion handler updates queue state
- [ ] **F6.4** Test: full step numerical parity vs current sync path (gradient check)

### F7 — TrainingManager loop

- [ ] **F7.1** States: `TRAIN`, `VAL`, `TICK_LEDGER`, `YIELD`
- [ ] **F7.2** Train: ask CNN add step; on BUSY → tick ledger / val / yield / retry
- [ ] **F7.3** On step OK from CNN: apply outer policy — version++, ledger docs, ES check
- [ ] **F7.4** On step FAIL: ledger `step.failed`, no version, optional retry batch later
- [ ] **F7.5** Val: forward-only contract, same queue handshake
- [ ] **F7.6** Test: ES still works with bounded queue; failed step doesn't bump version

### F8 — Full step contract (milestone)

- [ ] **F8.1** One compiled list: full forward + backward + adam for 2-conv CNN
- [ ] **F8.2** End-to-end: manager → CNN → native → completion → ledger
- [ ] **F8.3** Benchmark: compare wall time vs current sync path — workers should show less idle gap
- [ ] **F8.4** Exit: grad parity + benchmark within acceptable regression band

### F9 — Later (optional, not v1)

- [ ] Chained contract lists (fwd list → bwd list) with same-step priority
- [ ] Native → ledger direct write
- [ ] Re-run failed batch_id from manager

---

## Open perf decisions (benchmark to decide)

1. **Buffer pool:** native-owned vs shared region (default: native-owned)
2. **Submit API:** blocking master runs list inline vs native internal work queue
3. **Dense head:** always in contract (default: yes)

Escalate to discussion only if correctness is affected.

---

## Exit criteria (Phase F v1)

1. No Python `ThreadPoolExecutor` on training path
2. No per-layer sync ctypes on hot path — one contract submit per step
3. Manager bounded queue + pushback + ES still functional
4. Version increments only on successful steps; failures on tape without version
5. Grad check + benchmark parity with pre-F baseline

---

## Not in scope (v1)

- Auto-fork / overfit policy (`TrainingManager` advanced policy)
- Hogwild / multi-model parallel training
- Chained multi-contract steps (option only)
