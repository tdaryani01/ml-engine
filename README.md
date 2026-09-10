ML Engine
A neural network training engine built from scratch in Python/NumPy, with performance-critical ops (im2col+GEMM convolution, causal multi-head self-attention) implemented as hand-written, SIMD-accelerated native C++ kernels. No PyTorch/TensorFlow dependency in the training path — this repo is the framework.
Includes dense networks, CNNs, and a stacked causal Transformer block, all verified against numerical gradient checks, plus an append-only training ledger (checkpoint / fork / replay) for crash-recoverable training.
Status: active development. main has dense networks, CNNs, and the native/im2col+GEMM backends. The Transformer (MHSA) block lives on the feature/mhsa branch pending merge. See Roadmap below.

What's actually here
Component
Status
Notes
Dense (MLP) network
 stable
Adam/SGD, batch norm, dropout, L1/L2, LR scheduling, early stopping
CNN (Conv2D, MaxPool)
 stable
im2col-based, gradient-checked
Native backend (C++/AVX2/OpenMP)
 stable
Hand-written SIMD kernels; benchmarked against PyTorch
Causal MHSA / Transformer block
 feature/mhsa branch
Multi-layer, learned positional encoding, native Adam
Training ledger (checkpoint/fork/replay)
 stable (main)
Event-sourced training state; see docs/ledger-design.md
Streaming ingestion (AMQP)
 stable
data/stream_provider.py

Every numerical component (dense backprop, batch norm, CNN backward pass, Transformer backward pass — including cross-layer gradients in stacked blocks) is checked against finite-difference gradients before being trusted. See Testing & correctness.

Requirements
numpy>=1.26.0,<2.0.0
scipy>=1.14.0
numba>=0.59.0
pandas>=2.1.0
joblib>=1.3.0
pyyaml>=6.0
threadpoolctl>=3.4.0

Install:
pip install -r requirements.txt

A C++ compiler with AVX2/OpenMP support (g++ or clang++ on Linux/macOS, MSVC on Windows) is required to build the native backend. Without it, the engine falls back to a pure-NumPy/Numba path automatically.
Build the native backend (optional, recommended)
# Linux / macOS
./build_native.sh release

# Windows
.\build_native.ps1

This compiles src/native/*.cpp into bin/conv_kernels.so (or .dll). The engine detects and uses it automatically if present; if it's missing, or if ML_ENGINE_NATIVE_FALLBACK=1 is set, it falls back to the Numba-JIT path.
Run training
python run_pipeline.py

Configuration lives in config/config.yaml (architecture, optimizer, scheduler, data source, backend selection).
Run the tests
python run_tests.py

This runs the full suite (config, optimizers, schedulers, gradient checks, native conv, ledger, session isolation, etc.) and blocks with a non-zero status if anything fails.
Run a benchmark against PyTorch
python benchmarks/benchmark_cnn.py

Reports throughput, epoch time, and inference latency side-by-side against an equivalent PyTorch model, with explicit thread-count reporting (both sides use the same thread budget by default — see the script header for BENCHMARK_THREADS / ENGINE_NUMBA_PARALLEL). See benchmarks/sweep_kernel_pad.py for a fuller sweep across kernel size, stride, and pooling configuration — performance relative to PyTorch varies by configuration; the sweep script exists specifically to characterize where the crossover happens rather than report a single number.

Testing & correctness
This project's guiding principle: a component isn't trusted until its gradients are numerically verified. testing/test_gradient_check.py implements a central-difference gradient checker (with RNG seeding to handle stochastic ops like dropout correctly) and checks every learnable parameter — weights, biases, batch norm gamma/beta, and (on the MHSA branch) per-layer attention/FFN/LayerNorm parameters, including gradients flowing across stacked layers.
Run the full suite with python run_tests.py, or an individual module directly, e.g. python testing/test_gradient_check.py.

Roadmap & docs
docs/engine-roadmap.md — architecture roadmap, phased refactor plan, known technical debt
docs/ledger-design.md — training ledger design
docs/ledger-record-model.md — ledger record schema
docs/contract-list-architecture.md — async native contract/dispatch architecture
docs/phase-e-reference.md — Phase E (ledger) reference
License
See LICENSE.

