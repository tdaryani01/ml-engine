# ML Engine Architecture Roadmap

Design doc for refactoring the Python training stack before expanding im2col/GEMM
and moving more ops to native. Goal: **separation**, **testable increments**, and a
path toward **parallel training workers** and **multi-request inference** without
shared mutable globals.

Last updated: 2026-09-01 (Phase D in progress: `conv_dispatch.py` rename, native `im2col.cpp`).

---

## 1. Current state

### What works

- Native conv: k=1..7, stride 1/2, pad 1/2 (PyTorch parity at 28×28 verified).
- Training loop: `ModelController.fit()` → `CNNNetwork.backward()` (one forward inside backward per batch).
- Benchmarks: head-to-head vs PyTorch, Docker matrix sweep (`benchmark_diagnostics/`).

### Technical debt (do not build im2col→native on top of this)

| Issue | Location | Risk |
|-------|----------|------|
| Global backend singleton | `utils/conv_dispatch.py` (`_active_backend`, `_native_lib`) | Races if two sessions use different backends |
| Layer-owned step scratch | `ConvBlock`, `Conv2D` (`col`, `x_cached`, gemm buffers) | Not serializable; blocks ledger/worker model |
| `init_engine_backend()` at layer init | `spatial_layers.py` | Side effect on import/construct |
| Dual weight storage | `CNNNetwork.weights` + `layer.W` | ES restore bugs (fixed once; structurally fragile) |
| Shared ctypes BLAS scalars | `utils/im2col_fast.py` | **Fixed D1** — stack-local per GEMM call; delete file after D-native |
| Implicit NATIVE→im2col fallback | `conv_dispatch.py` dispatch | **Fixed D2** — strict unless `ML_ENGINE_NATIVE_FALLBACK=1` |

### Non-goals for early phases

- Merge `CNNNetwork` into `BaseNeuralNetwork` (cosmetic).
- True Hogwild / lock-free shared-model training (defer until ledger exists).
- Multi-VM infrastructure (queues, object store) before local ledger proof.

---

## 2. Target architecture

```
EngineContext              # model-agnostic session: backend + ops registry
  ├── conv: ConvOps        # CNN / spatial models
  ├── ops("attention")     # future: transformer / attention models
  └── native_lib           # shared DLL handle when backend=NATIVE

TrainingSession            # model + optimizer + provider + ctx + optional ledger client
  └── train_step(X, y)     # → TrainStepResult (grads, loss, metrics)

Consolidator (later)       # single writer: aggregate results → optimizer → version++

PublishedModel (later)     # versioned read-only weights for inference

Layer (ConvBlock, …)       # params + config only; no globals; receives ctx + scratch at call time
ForwardCache               # activations, masks, argmax (per batch step)
ScratchArena               # col, gemm, padded buffers (per step or thread-local)
```

### Database analogy (distributed / serving vision)

| Database | Training equivalent |
|----------|---------------------|
| Command / transaction | `TrainStepCommand` (batch, lr, `base_version`) |
| WAL / ledger | Append-only `TrainStepResult` (grads or deltas) |
| Roll-forward | Consolidator applies optimizer → new weight version |
| Read replica | Inference serves immutable checkpoint at version N |
| Single writer | Consolidator owns Adam `m/v/t` |

Workers do **not** share live `CNNNetwork` state. They emit immutable step results;
one consolidator materializes the latest weights.

---

## 3. Phase map (granular)

Each sub-phase is one reviewable PR. Run grad check + one benchmark case before merge.
Do not combine more than **one letter-group** (e.g. all of A) in a single PR unless trivial.

---

### Phase A — Engine context & ops backends (no behavior change)

**Outcome:** Dispatch reads `EngineContext`, not module globals. External API unchanged.

| ID | Scope | Files (typical) | Done when |
|----|--------|-------------------|-----------|
| **A1** | Define `ConvOps` protocol (forward, backward_fused, conv_block_*) | `utils/engine_ops.py` (new) | Protocol + docstrings; no callers yet |
| **A2** | `NativeConvOps` — delegate to existing ctypes paths in `conv_dispatch.py` | `engine_ops.py`, `conv_dispatch.py` | Native grad matrix still passes |
| **A3** | `NumpyConvOps` — delegate to reference im2col path | `engine_ops.py` | Used only in tests |
| **A4** | `Im2colGemmConvOps` — explicit class (same code as today’s IM2COL_GEMM path) | `engine_ops.py`, `im2col_fast.py` | No implicit fallback from NATIVE |
| **A5** | `EngineContext(backend)` factory loads correct `ConvOps` + native handle | `engine_ops.py` | Unit test: ctx.native vs ctx.numpy |
| **A6** | Add optional `ctx=` param to top-level conv entrypoints; default = legacy global | `conv_dispatch.py` | All existing callers pass without change |
| **A7** | `ConvBlock` stores `self._ctx`, removes `init_engine_backend()` from `__init__` | `spatial_layers.py` | Construct block with injected ctx |
| **A8** | `ModelFactory` / `CNNNetwork` create one `EngineContext`, pass to layers | `model_factory.py`, `cnn_network.py` | Pipeline + benchmark unchanged numerically |
| **A9** | Deprecate direct `init_engine_backend()` in app code; shim for tests only | `conv_dispatch.py`, `testing/*` | Grep shows no production init calls |

**Exit criteria (Phase A):** 28-case native grad check; one `benchmark_cnn.py` run; no new globals.

---

### Phase B — Step state separation

**Outcome:** Layers hold parameters only; per-batch state lives in cache/arena.

| ID | Scope | Files | Done when |
|----|--------|-------|-----------|
| **B1** | Define `ForwardCache` dataclass (spatial_inputs, masks, activations refs) | `src/training_cache.py` (new) | Populated in one forward pass |
| **B2** | Define `ScratchArena` (col, gemm, dout_trans, pad buffers) sized by batch cap | `src/scratch_arena.py` (new) | Alloc once per session or per step |
| **B3** | `ConvBlock.forward(..., cache, arena)` — stop writing `self.x_cached` | `spatial_layers.py` | ConvBlock tests or grad check k=3 |
| **B4** | `ConvBlock.backward(..., cache, arena)` — read from cache | `spatial_layers.py` | Same |
| **B5** | `CNNNetwork._forward` returns/ fills `ForwardCache` | `cnn_network.py` | backward uses cache, no re-forward |
| **B6** | Wire `set_train_batch_cap` to arena policy, not layer fields | `cnn_network.py`, `spatial_layers.py` | Variable batch sizes work |
| **B7** | Remove dead cache fields from layer instances | `spatial_layers.py` | Layers only have W, b, hyperparams |

**Exit criteria (Phase B):** Grad check + benchmark; backward must not call `_forward` (explicit split).

---

### Phase C — Training session boundary

**Outcome:** Training is a session object; controller becomes thin orchestration.

| ID | Scope | Files | Done when |
|----|--------|-------|-----------|
| **C1** | Define `TrainStepResult` (loss, grad_weights, grad_biases, step_id) | `src/training_session.py` (new) | Serializable numpy arrays |
| **C2** | `TrainingSession.train_step(X, y, lr)` — forward+cache → backward → return result | `training_session.py` | Does not call optimizer yet |
| **C3** | `TrainingSession.apply_step(result)` or inline optimizer in `train_step` | `training_session.py` | Matches current Adam behavior |
| **C4** | `TrainingSession.fit(epochs)` — epoch loop extracted from controller | `training_session.py` | ES + val predict optional flags |
| **C5** | `ModelController` delegates to `TrainingSession` (keep public API) | `controller.py` | `run_pipeline.py` unchanged |
| **C6** | Benchmark constructs session explicitly (optional cleanup) | `benchmarks/benchmark_cnn.py` | Benchmark numbers unchanged |

**Exit criteria (Phase C):** Two independent `TrainingSession` instances in one process (sequential OK); no shared globals mutated between them.

---

### Phase D — im2col/GEMM first-class & native expansion

**Outcome:** Three backends are peers; Python path is the grad-check oracle; native im2col+GEMM
replaces Numba/ctypes for performance.

#### D-py — Python peer backend (done)

| ID | Scope | Files | Status |
|----|--------|-------|--------|
| **D1** | Thread-safe GEMM — stack-local ctypes scalars per BLAS call | `im2col_fast.py` | Done |
| **D2** | Strict backend separation — `Im2colGemmConvOps` only; no silent native fallback | `engine_ops.py`, `conv_dispatch.py` | Done |
| **D3** | Grad check: native vs im2col_gemm vs numpy conv matrix (28 cases) | `testing/test_gradient_check.py` | Done |
| **D4** | ReLU/maxpool behind `ConvOps` protocol | `engine_ops.py`, `cnn_network.py` | Done |
| **D-rename** | Rename `im2col.py` → `conv_dispatch.py` (dispatch layer, not the algorithm) | `utils/conv_dispatch.py` | Done |
| **D-thread** | im2col+gemm: OpenBLAS=`num_threads`, OMP/Numba=1 (no dual pools) | `utils/runtime.py`, `config/runtime.yaml` | Done |

#### D-native — C++ im2col+GEMM (in progress)

| ID | Scope | Files | Done when |
|----|--------|-------|-----------|
| **D6** | Native `im2col_avx2` / `col2im_avx2` primitives | `src/native/im2col.cpp` | Done |
| **D7** | Wire native im2col/col2im into `IM2COL_GEMM` dispatch (fallback to Numba if DLL missing) | `conv_dispatch.py` | Done |
| **D8** | Native fused conv fwd/bwd via OpenBLAS `sgemm` | `src/native/conv_im2col_gemm.cpp`, `blas_dynamic.cpp` | Grad-check parity vs Python path |
| **D9** | BLAS via scipy capsule (`init_blas_sgemm_ptr`); ctypes exports in `conv_dispatch.py` | `build_native.ps1`, `conv_dispatch.py` | Benchmark win vs Python on k≥3 (pending AC rerun) |
| **D10** | Port fuse-dout, maxpool, ReLU to native (or keep thin Python for cold ops) | `src/native/*` | `im2col_fast.py` deletable |
| **D11** | Extract NumPy reference loops → `utils/conv_reference.py`; delete `im2col_fast.py` | `conv_reference.py` | Grep: zero `im2col_fast` imports |
| **D5** | Port dense head hot paths to native (if/when needed) | `src/native/*` | Per-op decision |

**Threading policy (im2col+gemm backend):** OpenBLAS owns parallelism at `optimization.num_threads`;
Numba/OpenMP pinned to 1 during `training_threadpool` — never both pools at full width.

**Exit criteria (Phase D):** IM2COL_GEMM selectable and correct; native im2col+GEMM beats Python path
on representative geometries; `im2col_fast.py` removed.

**Not in scope yet:** Replacing `GENERIC_FALLBACK` tiled loops in `conv_fallback.cpp` with native
im2col+GEMM routing — that is a separate dispatcher decision after D8–D9 are proven.

---

### Phase E — Ledger, training manager, serving (local proof first)

**Outcome:** Append-only training tape with replay/rewind/branch; SQL-style
single-thread main loop; pluggable storage.

**Start here:** [phase-e-reference.md](./phase-e-reference.md) · Details:
[ledger-design.md](./ledger-design.md) · [ledger-record-model.md](./ledger-record-model.md)

| ID | Scope | Files | Done when |
|----|--------|-------|-----------|
| **E1** | `LedgerStore` ABC + `TrainStepCommand` + record schemas | `src/ledger.py` (new) | Schema + interface tests |
| **E2** | `TrainStepResult.to_bytes()` / `from_bytes()` | `ledger.py` | Round-trip test |
| **E3** | `TrainingLedger` + single-thread consolidator / main loop | `src/training_engine.py` (new) | N-step replay matches `train_and_apply` |
| **E4** | `FileLedgerStore` + checkpoint sidecars | `ledger.py` | Rewind to checkpoint |
| **E5** | Branch fork/freeze + read-only replay | `ledger.py` | Frozen branch + active fork test |
| **E6** | Manager hooks: local best, last healthy, fork policy | `src/training_manager.py` (new) | Synthetic overfit → fork |
| **E7** | Document VM protocol (queue transport TBD) | `docs/distributed-training.md` | No impl required yet |

**Exit criteria (Phase E):** Replay/rewind from checkpoint matches single-thread
training for N steps; fork preserves bad-path tail on frozen branch.

---

### Phase F — Contract-list orchestration (OMP utilization)

**Outcome:** Manager pushes work; CNN owns queue + compiled contract lists; native
executes full steps without Python between ops. Version on success only.

**Design:** [contract-list-architecture.md](./contract-list-architecture.md)

| ID | Scope | Done when |
|----|--------|-----------|
| **F0** | Doc + remove Python worker thread path | Design landed |
| **F1** | Contract IR + init compile | 2-conv graph compiles to op list |
| **F2** | Native dumb executor | Op list parity vs sync ctypes |
| **F3** | Buffer pool | No leak across N steps |
| **F4** | Completion ring | Submit/drain event |
| **F5** | CNN queue + pushback | BUSY when full |
| **F6** | CNN defers compute | No per-layer conv_dispatch on hot path |
| **F7** | TrainingManager loop | ES + version-on-success |
| **F8** | Full step milestone | Grad check + benchmark |

---

## 4. Parallelism models (what we support when)

| Model | Phase | Notes |
|-------|-------|-------|
| Sequential two experiments | A | Two sessions, two models, different configs |
| Thread-safe IM2COL GEMM | D1 | Required before any multi-thread compute |
| Two training workers → consolidator | E3 (legacy) | **Replaced:** single-thread loop + optional multi-slot round-robin |
| Two inference requests, same version | E5 | Read-only weights + thread-local scratch |
| Shared-model multi-thread training (Hogwild) | — | **Not planned** unless ledger path insufficient |

---

## 5. Testing gates (every PR)

### Universal gates (all phases)

1. `python testing/test_gradient_check.py` (native conv matrix).
2. At least one `benchmarks/benchmark_cnn.py` or sweep case (quick config).
3. No new `init_engine_backend()` in `src/` outside factory/shim.

### Phase-specific gates (`testing/test_engine_ops.py`)

| Phase | Test function | Pass criteria |
|-------|---------------|---------------|
| **A1** | `test_a1_conv_ops_protocol` | `ConvOps` exposes forward/backward/block/pool methods |
| **A2** | `test_a2_native_conv_ops_matches_im2col` | `NativeConvOps` output matches `im2col` with same ctx |
| **A3** | `test_a3_numpy_conv_ops_runs_reference_path` | NumPy reference path runs without error |
| **A4** | `test_a4_im2col_gemm_conv_ops` | `Im2colGemmConvOps` is explicit, not implicit fallback |
| **A5** | `test_a5_engine_context_factory` | Factory sets backend; native lib only for NATIVE |
| **A6** | `test_a6_ctx_param_overrides_global` | `ctx=` dispatch ignores stale module global |
| **A7** | `test_a7_conv_block_uses_injected_ctx` | Layer stores injected ctx, no layer-level init |
| **A8** | `test_a8_model_factory_shares_engine_ctx` | One `EngineContext` shared across all spatial layers |
| **A9** | `test_a9_no_init_engine_backend_in_src` | Grep: zero `init_engine_backend` in `src/` |
| **A\*** | `test_model_agnostic_ops_registry` | `ctx.register("attention", …)` extensibility |
| **A\*** | `test_sequential_two_contexts` | Two contexts (NATIVE + NUMPY) coexist sequentially |

Run Phase A suite: `python testing/test_engine_ops.py`

### Future phase test stubs (add when implementing)

| Phase | Planned test | Pass criteria |
|-------|--------------|---------------|
| **B\*** | `test_b5_explicit_forward_backward_split` | `_backward_from_cache` does not call `_forward` |
| **B\*** | `test_b7_convblock_no_layer_cache_fields` | Layers hold params only, not step buffers |

Run Phase B suite: `python testing/test_training_cache.py`

### Future phase test stubs (add when implementing)
| **C** | `test_c_two_training_sessions` | two `TrainingSession` instances, no shared globals |
| **D** | `test_d_gemm_thread_safe` | concurrent GEMM smoke test passes |
| **D** | `test_d_three_backend_grad_parity` | native vs im2col_gemm vs numpy matrix |
| **D** | `test_native_im2col_matches_numpy_reference` | C++ im2col/col2im vs NumPy in `test_im2col.py` |
| **E** | `test_e_two_worker_consolidator` | two workers → consolidator matches single-thread |

Optional before Phase E: extend matrix sweep for regression timing.

---

## 6. File layout (target)

```
utils/
  engine_ops.py       # ConvOps, EngineContext, backend factories
  conv_dispatch.py    # backend routing, ctypes, conv2d_*, pool/ReLU (renamed from im2col.py)
  conv_reference.py   # NumPy-only im2col/col2im loops (grad-check oracle) — extract from dispatch
  im2col_fast.py      # TEMP: Numba+ctypes GEMM; delete after D-native (D10–D11)

src/native/
  conv_fallback.cpp   # tiled AVX2 direct conv (existing native path)
  conv_dispatcher.cpp # ctypes exports for conv blocks
  im2col.cpp          # im2col_avx2 / col2im_avx2 primitives (D6)
  conv_im2col_gemm.cpp # OpenBLAS sgemm fwd/bwd (D8)
  blas_dynamic.cpp     # scipy capsule / optional DLL load (D9)

src/
  training_session.py
  training_cache.py
  scratch_arena.py
  ...
```

---

## 7. Suggested order of work

```
A1 → … → A9
  → B1 → … → B7
  → C1 → … → C6
  → D1 → D2 → D3 → D4 → D-rename → D-thread   # D-py (done)
  → D6 → D7 → D8 → D9 → D10 → D11             # D-native (im2col.cpp → conv_im2col_gemm.cpp → delete im2col_fast.py)
  → D5 (optional dense head)
  → E1 → …
```

**Stop and ship** after each sub-phase. If a sub-phase grows beyond ~300 lines changed, split it.

---

## 8. Open decisions (record answers here)

| Question | Decision | Date |
|----------|----------|------|
| Ledger stores grads or weight deltas? | TBD (recommend **grads**; consolidator owns Adam) | |
| Queue technology for VMs? | TBD | |
| Keep `ModelController` name vs rename to `TrainingSession` publicly? | TBD (recommend keep controller as facade) | |
| im2col/GEMM parity required before native pool/ReLU port? | **Yes** — Python path is oracle; native ports after D3 passes | 2026-09-01 |
| Rename `im2col.py` → `conv_dispatch.py`? | **Done** — file is dispatch, not the algorithm | 2026-09-01 |
| Kill `im2col_fast.py` when? | After D8–D11 (native GEMM + optional fuse/pool/ReLU) | |

---

## 9. References

- Grad matrix: `testing/test_gradient_check.py` (k=1..7 × s=1,2 × p=1,2).
- Docker matrix log: `benchmark_diagnostics/conv_matrix_k1-7_s1-2_p1-2_*.log`.
- Runtime threading: `config/runtime.yaml`, `utils/runtime.py`.
