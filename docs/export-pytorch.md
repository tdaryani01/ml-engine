# Export a trained model to PyTorch

Dense models (`binary_classification`, `regression`, `multi_class`) saved by `ModelSerializer` can be exported as a standalone PyTorch module. CNN and MHSA are not supported yet and are refused by name.

```
pip install torch                       # to write weights.pt, and to use the exported module
python -m src.export_pytorch model.npz out_dir
```

`out_dir` gets three files:

| File | What it is |
|---|---|
| `model.py` | The network as plain torch code. It does not import anything from this project. `load()` returns it in eval mode. |
| `weights.pt` | A torch `state_dict` (float32): the weights, the batch-norm statistics and the input/output scaling. |
| `preprocess.json` | What the module takes and returns, and any warning (below). |

```python
import torch
from model import load            # run from out_dir, or put it on the path
net = load()
with torch.no_grad():
    y = net(torch.tensor(x_raw))  # x_raw: (N, n_features), RAW features
```

`x_raw` may be float32 or float64 (numpy and pandas give float64), or a single 1-D row. **The columns must be in the same order as the training file's features**: the module has no column names. The module standardises the features itself with the training mean/std, and returns:

- `binary_classification`: the probability of the positive class, `(N, 1)`
- `multi_class`: class probabilities, `(N, n_classes)`; column k is the label k the engine trained with
- `regression`: values in the original units of the target (the module undoes the target standardisation)

## What is carried over

Layer order is Linear, batch norm (if the model used it), LeakyReLU (slope 0.01), then the head. Batch norm uses the saved running statistics (eps 1e-5). Dropout is a no-op at inference and is not exported. The engine stores weights as (in, out); torch `Linear` wants (out, in), so they are transposed.

The tests (`testing/test_export_pytorch.py`) compare the module with `model.predict` on random inputs: to 1e-9 in float64 (the math) and to about 2e-4 relative in float32 (the files).

## Warnings

- **No input mean/std in the archive** (saved without a data provider): the module expects features that are already standardised.
- **No target mean/std in a regression archive** (saved before `y_mean`/`y_std` were stored): the module returns standardised predictions. Multiply by the training std of the target and add its mean.

`ModelSerializer.save_model` now stores `y_mean` and `y_std` for regression models, and `load_model` restores them.

## Not covered

CNN and MHSA models; training or fine-tuning in PyTorch from the exported weights (the batch-norm momentum is converted, but the optimiser state is not exported); a Desktop button.
