"""A9: the zigzag of the per-step training loss rides the step document; reading it never changes a loss."""
import math

import numpy as np

from src import step_loss_stats as S


def test_the_accumulator_reports_the_population_std_and_resets():
    a = S.StepLossStats()
    for x in (1.0, 2.0, 3.0, 4.0):
        a.observe(x)
    assert math.isclose(a.drain()[S.KEY], float(np.std([1.0, 2.0, 3.0, 4.0])))
    assert a.drain() == {}                                                            # reset; nothing observed = nothing reported
    a.observe(float("nan"))
    assert a.drain() == {}


def test_off_switch(monkeypatch):
    monkeypatch.setenv("ML_STEP_LOSS_STATS", "0")
    S.observe_epoch([1.0, 5.0])
    assert S.drain() == {}
    monkeypatch.delenv("ML_STEP_LOSS_STATS")
    S.STATS.drain()


def test_a_small_batch_zigzags_more_than_a_large_one_through_the_real_fit_and_losses_are_bit_equal(monkeypatch, tmp_path):
    """A real tiny MLP fit through the ledger's step documents: the reported field is larger at batch 8 than at batch 64, and the epoch losses are the same with the statistic on or off."""
    from src.launcher.payload import SupervisedFitSpec
    from src.launcher.runner import run_me_fit

    rng = np.random.default_rng(0)
    x = rng.normal(size=(512, 4))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    data = tmp_path / "t.csv"
    data.write_text("a,b,c,d,y\n" + "\n".join(",".join(f"{v:.6f}" for v in r) + f",{k}" for r, k in zip(x, y)) + "\n")

    def fit(batch, seed, sub):
        w = tmp_path / sub
        w.mkdir()
        spec = SupervisedFitSpec(model_type="binary_classification", data_path=str(data), num_classes=1, hidden_layers=(8,), feature_names=("a", "b", "c", "d"), learning_rate=0.01, batch_size=batch, epochs=3, patience=None, seed=seed, checkpoint_every=40, num_threads=1)
        import sys

        snaps = list(run_me_fit(spec, work_dir=w, should_stop=lambda: False, checkpoint_path=None, base_traj=0, python=sys.executable, max_wall_s=300))
        return [(s["train_loss"], s["loss"]) for s in snaps], [s.get("extra_metrics", {}).get(S.KEY) for s in snaps]

    small = [fit(8, s, f"s{s}") for s in (1, 2, 3)]
    large = [fit(64, s, f"l{s}") for s in (1, 2, 3)]
    ms = [np.mean([v for v in e if v is not None]) for _, e in small]
    ml = [np.mean([v for v in e if v is not None]) for _, e in large]
    assert all(any(v is not None for v in e) and e[-1] is not None for _, e in small + large)      # a checkpoint carries the last step document's value: those before the first epoch ends have none (as the gradient statistics)
    assert min(ms) > max(ml)                                                              # higher for the small batch, by more than the spread across seeds
    monkeypatch.setenv("ML_STEP_LOSS_STATS", "0")
    off = fit(8, 1, "off")
    assert off[0] == small[0][0] and all(v is None for v in off[1])                       # default losses bit-equal; nothing reported when off
