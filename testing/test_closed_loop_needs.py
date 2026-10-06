"""Options declare what they need from the machine; the Desktop asks for it at Add time."""
from __future__ import annotations

import pytest

from src.closed_loop import needs as nd

MODS = {"encoder": "cnn_upstream", "policy": "mhsa", "env": "soft_canvas", "loss": "canvas_reconstruction"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(nd, "_NEEDS", {})


def test_an_option_that_declares_nothing_needs_nothing() -> None:
    assert nd.assembly_needs(MODS) == {"needs": [], "missing": [], "conflicts": []}


def test_needs_of_the_chosen_options_are_merged_by_name_and_required_if_any_requires() -> None:
    nd.declare_needs("env", "soft_canvas", [{"name": "targets", "kind": "file", "label": "Targets", "required": False}])
    nd.declare_needs("loss", "canvas_reconstruction", [{"name": "targets", "kind": "file", "required": True}, {"name": "threads", "kind": "number"}])
    out = nd.assembly_needs(MODS)
    by = {n["name"]: n for n in out["needs"]}
    assert set(by) == {"targets", "threads"} and by["targets"]["required"] is True and by["targets"]["label"] == "Targets"
    assert out["conflicts"] == [] and out["missing"] == []


def test_the_same_need_with_two_kinds_is_a_conflict() -> None:
    nd.declare_needs("env", "soft_canvas", [{"name": "x", "kind": "file"}])
    nd.declare_needs("loss", "canvas_reconstruction", [{"name": "x", "kind": "number"}])
    assert "need 'x' is a file" in nd.assembly_needs(MODS)["conflicts"][0]


def test_an_option_this_ml_engine_lacks_is_reported_with_what_it_has() -> None:
    out = nd.assembly_needs({**MODS, "policy": "future_policy"})
    assert len(out["missing"]) == 1 and "no policy option named 'future_policy'" in out["missing"][0] and "mhsa" in out["missing"][0]


def test_a_bad_declaration_is_refused() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        nd.declare_needs("env", "soft_canvas", [{"name": "x", "kind": "blob"}])
    with pytest.raises(ValueError, match="unknown seat"):
        nd.declare_needs("nope", "x", [])


def test_answers_are_checked_by_kind(tmp_path) -> None:
    f = tmp_path / "t.csv"
    f.write_text("a,b\n")
    needs = [{"name": "f", "kind": "file", "label": "File", "required": True}, {"name": "d", "kind": "folder", "label": "Dir", "required": False},
             {"name": "n", "kind": "number", "label": "N", "required": True}, {"name": "s", "kind": "secret", "label": "Key", "required": True}]
    assert nd.check_answers(needs, {"f": str(f), "d": str(tmp_path), "n": "4", "s": "x"}) == []
    bad = nd.check_answers(needs, {"f": "rel/path", "d": "rel", "n": "four"})
    assert "File: give the full path" in bad and "Dir: give the full path" in bad and "N: not a number" in bad and "Key: required" in bad
    assert "File: no such file" in nd.check_answers(needs, {"f": str(tmp_path / "missing.csv"), "n": "1", "s": "x"})[0]
    assert nd.check_answers(needs, {"n": "1", "s": "x"}) == ["File: required"]
