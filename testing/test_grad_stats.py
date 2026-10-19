"""Per-boundary gradient statistics (src/grad_stats.py): exact values from the Python optimizers, bit-identical weight updates, the drain into every step document, and the off switch."""
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import grad_stats
from src.ledger import FileLedgerStore, TrainingLedger
from src.optimizers import AdamOptimizer, SGDOptimizer


def _problem():
    w = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]
    b = [np.array([[0.5, -0.5]], dtype=np.float32)]
    gw = [np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)]
    gb = [np.array([[0.05, -0.05]], dtype=np.float32)]
    return w, b, gw, gb


def test_the_accumulator_reports_mean_variance_and_ratio_over_the_steps_and_resets():
    s = grad_stats.GradStats()
    for g in (1.0, 2.0, 3.0):
        s.observe(g * g, 0.01 * g * g, 100.0)
    out = s.drain()
    assert math.isclose(out["grad_norm_mean"], 2.0) and math.isclose(out["grad_norm_var"], 2.0 / 3.0)
    assert math.isclose(out["update_weight_ratio"], 0.02)                          # per step 0.01 * g, mean of 0.01, 0.02, 0.03
    assert s.drain() == {}                                                         # drained: nothing until the next steps


def test_adam_reports_the_exact_gradient_norm_and_update_ratio_and_the_weights_are_unchanged_by_it(monkeypatch):
    grad_stats.STATS.drain()
    w, b, gw, gb = _problem()
    opt = AdamOptimizer(lr=0.1)
    w0, b0 = [x.copy() for x in w], [x.copy() for x in b]
    opt.update(weights=w, biases=b, grad_weights=gw, grad_biases=gb, m_samples=1, lam_l2=0.0, active_lr=0.1)
    out = grad_stats.drain()
    g_norm = math.sqrt(sum(float((x.astype(np.float64) ** 2).sum()) for x in (*gw, *gb)))
    assert math.isclose(out["grad_norm_mean"], g_norm, rel_tol=1e-6) and out["grad_norm_var"] == 0.0
    step = 0.1 * 1.0                                                                   # Adam's first step is lr per element (m/sqrt(v) = sign)
    w_norm = math.sqrt(sum(float((x.astype(np.float64) ** 2).sum()) for x in (*w0, *b0)))
    u_norm = math.sqrt(sum(x.size for x in (*w0, *b0))) * step
    assert math.isclose(out["update_weight_ratio"], u_norm / w_norm, rel_tol=1e-3)
    monkeypatch.setenv("ML_GRAD_STATS", "0")                                           # the same update with the switch off gives the same weights
    w2, b2, gw2, gb2 = _problem()
    AdamOptimizer(lr=0.1).update(weights=w2, biases=b2, grad_weights=gw2, grad_biases=gb2, m_samples=1, lam_l2=0.0, active_lr=0.1)
    assert np.array_equal(w[0], w2[0]) and np.array_equal(b[0], b2[0]) and grad_stats.drain() == {}


def test_sgd_reports_the_gradient_norm_and_the_applied_velocity_ratio():
    grad_stats.STATS.drain()
    w, b, gw, gb = _problem()
    SGDOptimizer(lr=0.1, momentum=0.0).update(weights=w, biases=b, grad_weights=gw, grad_biases=gb, m_samples=1, lam_l2=0.0, active_lr=0.1)
    out = grad_stats.drain()
    g_norm = math.sqrt(sum(float((x.astype(np.float64) ** 2).sum()) for x in (*gw, *gb)))
    assert math.isclose(out["grad_norm_mean"], g_norm, rel_tol=1e-6)
    assert math.isclose(out["update_weight_ratio"], 0.1 * g_norm / math.sqrt(1 + 4 + 9 + 16 + 0.25 + 0.25), rel_tol=1e-5)


def test_native_applied_steps_report_only_the_gradient_norm():
    grad_stats.STATS.drain()
    grad_stats.observe_grads([np.ones((2, 2), dtype=np.float32), np.ones(3, dtype=np.float32)])
    out = grad_stats.drain()
    assert math.isclose(out["grad_norm_mean"], math.sqrt(7.0)) and "update_weight_ratio" not in out


def test_every_step_document_carries_the_statistics_beside_a_familys_own_extras():
    grad_stats.STATS.drain()
    w, b, gw, gb = _problem()
    AdamOptimizer(lr=0.1).update(weights=w, biases=b, grad_weights=gw, grad_biases=gb, m_samples=1, lam_l2=0.0, active_lr=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main")
        ledger.push_step_metrics(1, 1, 0.5, 0.4, extra_metrics={"val_accuracy": 0.9})
        store.flush()
        docs = [d for d in store.scan() if d.doc_type == "step.metrics"]
        body = docs[-1].body
    assert {"val_accuracy", "grad_norm_mean", "grad_norm_var", "update_weight_ratio"} <= set(body["extra_metrics"])
    assert grad_stats.drain() == {}                                                    # the document drained it


def test_an_imitation_fit_reports_gradient_norms_and_the_update_ratio_through_the_contract_step(tmp_path, monkeypatch) -> None:
    """Imitation trains in the contracts' native trainer, which steps ml-engine's TrainableModel: that step is where its gradients are observed."""
    from testing.test_imitation_run import _corpus, _payload, _run

    monkeypatch.setenv("ML_ENGINE_ROOT", str(Path(__file__).resolve().parents[1]))
    corpus = _corpus(tmp_path / "c.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    docs = _run(tmp_path, _payload(out, corpus, steps=6))
    extras = [d.body.get("extra_metrics") or {} for d in docs if d.doc_type == "step.metrics"]
    assert extras and all({"grad_norm_mean", "grad_norm_var", "update_weight_ratio"} <= set(e) and 0.0 < e["update_weight_ratio"] < 1.0 for e in extras)
