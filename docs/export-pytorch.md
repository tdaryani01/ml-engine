# Export a trained model to PyTorch

A model saved by `ModelSerializer.save_model` can be exported as a standalone PyTorch module: the dense families (`binary_classification`, `regression`, `multi_class`), `cnn` and `mhsa`.

```
pip install torch                       # to write weights.pt, and to use the exported module
python -m src.export_pytorch model.npz out_dir
```

`out_dir` gets three files:

| File | What it is |
|---|---|
| `model.py` | The network as plain torch code. It does not import anything from this project. `load()` returns it in eval mode. |
| `weights.pt` | A torch `state_dict` (float32): the weights and, for the dense families, the input/output scaling. |
| `preprocess.json` | What the module takes and returns, and any warning (below). |

```python
import torch
from model import load            # run from out_dir, or put it on the path
net = load()
with torch.no_grad():
    y = net(torch.tensor(x))
```

`x` may be float32 or float64 (numpy and pandas give float64). What it holds, and what comes back:

| Family | `x` | Returns |
|---|---|---|
| `binary_classification` | `(N, n_features)` raw features, **columns in the training file's order** (a 1-D row also works) | the probability of the positive class, `(N, 1)` |
| `multi_class` | the same | class probabilities, `(N, n_classes)`; column k is the label k the engine trained with |
| `regression` | the same | values in the original units of the target (the module undoes the target standardisation) |
| `cnn` | `(N, C, H, W)` images, **scaled the way you trained them** (the image loader divides by 255, the module does not) and **without the width padding** (a single `(C, H, W)` image also works; images of a different size than the model was trained on raise a `ValueError` that names the shape and the expected feature count) | class probabilities (`multiclass`), a probability (`binary`) or raw values (`regression`), by the model's `task_type` |
| `mhsa` | `(N, T, input_dim)` sequences with `T <= max_seq_len` (a single `(T, input_dim)` sequence also works) | the action of the **last** position: `tanh`-bounded for `continuous`, raw logits for `discrete` |

The dense modules standardise the features themselves with the training mean/std. **A cnn or mhsa module uses its input as given: no mean/std is stored for them**, so repeat any scaling you did before training (the export says so in `preprocess.json`).

## What is carried over

- **Dense:** Linear, batch norm (if used; saved running statistics, eps 1e-5), LeakyReLU (slope 0.01), then the head. The engine stores weights as (in, out); torch `Linear` wants (out, in), so they are transposed. Dropout is a no-op at inference and is not exported.
- **CNN:** each conv layer (conv weights are (out, in, kh, kw), which is what torch uses), ReLU, max-pool and flatten in the engine's order, then a dense head with ReLU between layers and a softmax / sigmoid / identity output by `task_type`. A fused conv + ReLU + pool block is exported as those three operations.
- **MHSA:** the optional input projection, the learned positional embedding, then per layer Pre-LN attention (causal, 1/sqrt(d_head), `qkv` packed as [Q|K|V] with heads side by side) and a GELU (tanh form) feed-forward block with residuals (layer-norm eps 1e-5), then the head on the last position.

The tests (`testing/test_export_pytorch.py`) compare each module with the engine's own `predict` on random inputs, to 1e-9 for the dense math in float64 and about 1e-4 to 1e-3 relative for cnn and mhsa (the engine computes those in float32). The MHSA comparisons and the overlapping-pool CNN case run against the native kernels, so they are skipped where those are not built (`build_native.sh`; OpenBLAS must be found, see `ML_ENGINE_OPENBLAS_DLL`).

## Saving a cnn or mhsa model for export

`ModelSerializer.save_model` now also writes cnn and mhsa models (the parameters plus the layer list, heads, sizes and flags that are not in the weights' shapes). `ModelSerializer.load_model` refuses such a file with a clear message: reloading a cnn or mhsa model into the engine is not supported yet. The dense archives are unchanged, plus `y_mean`/`y_std` for regression models.

## Warnings

- **Dense, no input mean/std in the archive** (saved without a data provider): the module expects features that are already standardised.
- **Regression, no target mean/std in the archive** (saved before `y_mean`/`y_std` were stored): the module returns standardised predictions. Multiply by the training std of the target and add its mean.
- **cnn / mhsa:** inputs are used as given (above). If the data provider had a non-trivial mean/std when the model was saved, the save prints a warning, the archive records it, and `preprocess.json` repeats it: the module expects inputs scaled the same way.
- **cnn `task_type`:** the engine compares it verbatim (`"multiclass"` softmax, `"binary"` sigmoid, anything else raw). The exporter accepts `multiclass`, `binary` and `regression` and refuses any other value, or a missing one, instead of guessing the output.
- **Archive format:** an archive written by a newer engine than this exporter reads is refused.

## A known engine limit the export does not paper over

- The pure-NumPy max-pool fallback (`conv_dispatch.maxpool_forward`) reshapes by `stride` and ignores `pool_size`, so it is right only when the two are equal (the default 2/2). With an overlapping pool (stride below pool size) the engine's NumPy path gives different numbers from the native kernels and from PyTorch; the export follows the correct max-pool and the native kernels. This is the only NumPy max-pool, so an overlapping pool is wrong for every user without the native library: it deserves its own issue.

## Not covered

Training or fine-tuning in PyTorch from the exported weights (the batch-norm momentum is converted, but the optimiser state is not exported); the other families (closed-loop, imitation); a Desktop button.
