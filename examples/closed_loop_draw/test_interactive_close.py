# Window closed mid-round: the widgets are gone (tk.TclError on every call), the round in flight must go on and end without raising.
# Run: PYTHONPATH=. .venv/bin/python -m pytest examples/closed_loop_draw/test_interactive_close.py
from __future__ import annotations

import tkinter as tk

from examples.closed_loop_draw.interactive import DrawInteractiveApp


class _Widget:
    """A widget that works while the window is open and raises what Tk raised at 05:27 on 2026-11-27 once it is closed."""

    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        def call(*a, **k):
            if self.app._closing:
                raise tk.TclError('invalid command name ".!frame.!button2"')
        return call

    def __setitem__(self, k, v):
        if self.app._closing:
            raise tk.TclError("dead")


def _app():
    app = object.__new__(DrawInteractiveApp)
    app.busy = False
    app._closing = False
    app._stop_requested = False
    for n in ("btn_draw", "btn_train", "btn_warm", "btn_brain", "btn_sweep", "btn_cont", "btn_stop", "entry", "log_box", "progress", "status_var", "root"):
        setattr(app, n, _Widget(app))
    app.sketch = type("S", (), {"use_btn": _Widget(app)})()
    app._pump = lambda: None
    return app


def test_round_in_flight_finishes_after_the_window_is_closed():
    app = _app()
    done = []

    def round_():
        app._closing = True                     # what _on_close does while the round runs
        app._brain_log("saves of the round")
        app._set_progress(0.5)
        done.append("saved")

    DrawInteractiveApp._run_action(app, round_)   # the finally block runs on a closed window: must not raise
    assert done == ["saved"] and app.busy is False


def test_error_after_close_is_printed_not_raised(capsys):
    app = _app()

    def boom():
        app._closing = True
        raise RuntimeError("round broke")

    DrawInteractiveApp._run_action(app, boom)
    assert "round broke" in capsys.readouterr().out and app.busy is False
