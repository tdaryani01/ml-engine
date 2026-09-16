# Kitchen data generators

Supervised synth diets for Training Manager Start (recipe → `materialize` → diet path).

## Layout

| Path | Role |
|------|------|
| `csv/` | Tabular CSV generators (binary / multi-class / regression) |
| `cnn/` | Procedural RGB shape NPZ for the CNN path |
| `mhsa/` | Sequence NPZ for MHSA |
| `archive/security/` | **Not** kitchen-registered — keep archived |
| `_cli.py` | Shared `--out / --n / --seed` CLI helper |

## Contract

Every live kitchen module exposes:

```python
def materialize(out_path: str, *, n_samples: int = ..., seed: int = 42, **kwargs) -> str:
    ...
    return str(resolved_path)
```

CLI entrypoints call `_cli.run_materialize_cli(...)`.

TM allowlists recipes in `training-manager/backend/app/feeds/synth_generators.py` (`SYNTH_RECIPES`).
