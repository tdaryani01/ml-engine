"""The fit reports ``n_train``: the training rows it draws from, counted by the engine where the data is known (ml-engine; contracts n_train on LiveMetrics / DoneMetrics).
It must equal the engine's OWN count of its training set, for a supervised fit and a closed-loop one, and reach the runner's snapshots; and it is absent (never a guess) where a family cannot say."""
from __future__ import annotations

import numpy as np

from src.closed_loop.assembler import assemble_closed_loop
from src.closed_loop.fit import fit_closed_loop
from testing.test_closed_loop_demonstrations import _Ledger as _BaseLedger
from testing.test_closed_loop_demonstrations import _cfg as _demo_cfg
from testing.test_closed_loop_demonstrations import _demos
from testing.test_closed_loop_navigate import _cfg as _nav_cfg
from testing.test_closed_loop_navigate import _grid


class _Ledger(_BaseLedger):
    def push_step_metrics(self, step, version, train, val, **kw):
        super().push_step_metrics(step, version, train, val, **kw)
        self.n_train = getattr(self, "n_train", [])
        self.n_train.append(kw.get("n_train"))


def test_navigate_reports_the_training_origin_destination_pairs(tmp_path) -> None:
    run = assemble_closed_loop(_nav_cfg(_grid(tmp_path / "g.npz")), seed=0)
    pairs = sum(len(c[1]) for c in run.data.train_cases)          # the engine's own training pool, counted from its cases
    assert pairs > len(run.data.train_cases) > 0 and run.data.n_train == pairs
    led = _Ledger()
    fit_closed_loop(run, led, lr=0.003, steps=3, patience=0, checkpoint_every=3)
    assert led.n_train == [pairs, pairs, pairs]                   # the same on every step: the pool, not the batch


def test_demonstrations_report_the_training_episodes_left_after_the_held_out_ones(tmp_path) -> None:
    run = assemble_closed_loop(_demo_cfg(_demos(tmp_path / "d.npz", episodes=40)), seed=0)
    assert run.data.n_train == len(run.data.train_idx) == 32       # 40 episodes, 20% held out
    led = _Ledger()
    fit_closed_loop(run, led, lr=0.02, steps=2, patience=0, checkpoint_every=2)
    assert led.n_train == [32, 32]


def test_a_supervised_fit_reports_the_rows_of_its_train_split(tmp_path) -> None:
    from src.data.tabular_loader import TabularCSVLoader
    from src.launcher import run_me_fit
    from testing.test_launcher import _spec

    spec = _spec(tmp_path, epochs=2, checkpoint_every=10)
    X_train = TabularCSVLoader(spec.data_path, ["Feature_1", "Feature_2"], train_split=0.7, val_split=0.15).load_splits()[0]
    snaps = list(run_me_fit(spec, work_dir=tmp_path / "w", should_stop=lambda: False))
    reported = {s.get("n_train") for s in snaps if s.get("n_train")}      # a snapshot before the first epoch has ended has no count yet
    assert reported == {len(X_train)} and len(X_train) > 100 and snaps[-1]["n_train"] == len(X_train)      # the loader's own training rows (a fraction of the 300-row plate)


def test_a_closed_loop_run_carries_n_train_out_with_each_snapshot(tmp_path) -> None:
    from src.closed_loop.demonstration_options import Demonstrations   # noqa: F401  (the family under test)
    from src.launcher import FamilyFitSpec, run_me_fit

    cfg = _demo_cfg(_demos(tmp_path / "d.npz", episodes=40), max_steps=4)
    cfg["assembly"]["family_id"] = "closed_loop"
    spec = FamilyFitSpec(model_id="m", config=cfg, fit={"run_budget": 6, "patience": 0, "checkpoint_every": 3, "lr": 0.02}, num_threads=1)
    snaps = list(run_me_fit(spec, work_dir=tmp_path / "w", should_stop=lambda: False))
    assert snaps and {s.get("n_train") for s in snaps} == {32}


def test_the_ledger_writes_n_train_into_the_step_metrics_document_and_leaves_it_out_when_unknown() -> None:
    import tempfile

    from src.ledger import FileLedgerStore, TrainingLedger

    tmp = tempfile.mkdtemp()
    store = FileLedgerStore(tmp)
    led = TrainingLedger(store=store, branch_id="main", architecture_id="x")
    led.push_step_metrics(1, 1, 0.5, 0.4, n_train=7)
    led.push_step_metrics(2, 2, 0.5, 0.4)                           # a family that cannot say
    led.push_step_metrics(3, 3, 0.5, 0.4, n_train=0)                # zero is not a count
    store.flush()
    got = [d.body.get("n_train") for d in FileLedgerStore(tmp).scan(1) if d.doc_type == "step.metrics"]
    assert got == [7, None, None]
