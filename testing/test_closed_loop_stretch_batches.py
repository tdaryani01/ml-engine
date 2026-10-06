"""Each fit of a chain draws its own training batches (a chain that replayed the first fit's batches would train on 60 batches in 36 fits)."""
from __future__ import annotations

import numpy as np

from src.closed_loop.assembler import assemble_closed_loop
from testing.test_closed_loop_demonstrations import MODS as DEMO_MODS, _cfg as demo_cfg, _demos
from testing.test_closed_loop_navigate import MODS as NAV_MODS, _cfg as nav_cfg, _grid
from testing.test_closed_loop_reach import MODS as REACH_MODS, _cfg as reach_cfg


def _with_stretch(cfg: dict, stretch: int) -> dict:
    return {**cfg, "fit": {"stretch_index": stretch}}


def _signature(goal) -> bytes:
    arrays = [getattr(goal, k) for k in ("observations", "origin", "targets") if hasattr(goal, k)]
    return b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)


def _check(cfg: dict) -> None:
    first = [assemble_closed_loop(_with_stretch(cfg, 1), seed=0) for _ in range(2)]
    later = assemble_closed_loop(_with_stretch(cfg, 7), seed=0)
    g1 = [_signature(r.data.train_goal(i)) for r in first for i in (1, 2, 3)]
    assert g1[:3] == g1[3:]  # the same stretch is reproducible
    g7 = [_signature(later.data.train_goal(i)) for i in (1, 2, 3)]
    assert all(a != b for a, b in zip(g1[:3], g7))  # a later stretch draws different batches at the same step numbers
    assert _signature(first[0].data.val_goal(1)) == _signature(later.data.val_goal(1))  # while validation stays the same set, so fits stay comparable
    for r in (*first, later):
        r.close()


def test_demonstrations_draw_new_batches_each_stretch(tmp_path) -> None:
    _check(demo_cfg(_demos(tmp_path / "d.npz")))


def test_reach_targets_draw_new_batches_each_stretch() -> None:
    _check(reach_cfg())


def test_road_routes_draw_new_batches_each_stretch(tmp_path) -> None:
    _check(nav_cfg(_grid(tmp_path / "g.npz")))
