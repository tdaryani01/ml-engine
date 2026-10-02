# testing/test_optional_dependencies.py
"""A black-box fit must not need pika (AMQP) or matplotlib (plots) just to import ME's entry modules."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_entry_modules_import_without_pika_or_matplotlib() -> None:
    with tempfile.TemporaryDirectory() as blockers:
        for pkg in ("pika", "matplotlib", "mpl_toolkits"):
            os.makedirs(os.path.join(blockers, pkg))
            with open(os.path.join(blockers, pkg, "__init__.py"), "w") as fh:
                fh.write(f"raise ImportError('{pkg} is not installed (test blocker)')\n")
        code = (
            "import src.data.stream_provider, utils.diagnostics;"
            "from utils.diagnostics import NeuralNetworkDiagnostics; print('ok')"
        )
        env = {**os.environ, "PYTHONPATH": blockers + os.pathsep + ROOT}
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-800:]
    assert "ok" in out.stdout
