"""Options declare what they need from the machine; the Desktop asks for it at Add time."""
from __future__ import annotations

import pytest

from src.closed_loop import needs as nd

MODS = {"encoder": "cnn_upstream", "policy": "mhsa", "env": "soft_canvas", "loss": "canvas_reconstruction"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    from src.closed_loop import registry

    registry._load_builtin_options()  # the built-ins declare their needs on first import: let them, before the dict is swapped
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


def test_answers_land_at_the_needs_config_key_over_what_tm_sent() -> None:
    needs = [{"name": "f", "kind": "file", "config_key": "closed_loop.path"}, {"name": "n", "kind": "number", "config_key": "closed_loop.threads"},
             {"name": "x", "kind": "folder"}]
    cfg = {"closed_loop": {"path": "/tm/sent/this", "keep": 1}}
    out = nd.apply_answers(cfg, [nd._clean(n) for n in needs], {"f": "/mine/d.npz", "n": "3"})
    assert out["closed_loop"] == {"path": "/mine/d.npz", "keep": 1, "threads": 3.0} and cfg["closed_loop"]["path"] == "/tm/sent/this"  # a copy
    gone = nd.apply_answers(cfg, [nd._clean(needs[0])], {})
    assert "path" not in gone["closed_loop"]  # no answer: a path TM sent is not used
    made = nd.apply_answers({}, [nd._clean(needs[0])], {"f": "/a"})
    assert made == {"closed_loop": {"path": "/a"}}


def test_a_home_shorthand_in_a_file_or_folder_answer_becomes_the_real_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    needs = [nd._clean({"name": "f", "kind": "file", "config_key": "closed_loop.path"}), nd._clean({"name": "d", "kind": "folder", "config_key": "closed_loop.dir"}),
             nd._clean({"name": "s", "kind": "secret", "config_key": "closed_loop.key"})]
    out = nd.apply_answers({}, needs, {"f": "~/data/a.npz", "d": "~/out", "s": "~keep-as-is"})
    assert out["closed_loop"] == {"path": f"{tmp_path}/data/a.npz", "dir": f"{tmp_path}/out", "key": "~keep-as-is"}  # a secret is not a path
