"""Run with the INSTALLED wheel's Python, from outside the checkout: the fit launcher (src/launcher/runner.py) starts `python run_pipeline.py` with the engine's root as the working directory,
so for an installed engine that file must be in the root the launcher computes (site-packages), and it must compile."""
import py_compile

import src.launcher.runner as runner

root = runner.ML_ENGINE_ROOT
assert (root / "run_pipeline.py").is_file(), f"run_pipeline.py is not in the launcher's root: {root}"
assert "site-packages" in str(root), f"launcher root is not the installed package location: {root}"
py_compile.compile(str(root / "run_pipeline.py"), doraise=True)      # not imported: with no TM payload in the environment it reads config/config.yaml at import time (a real fit passes its own boot YAML and payload)
print("launcher root:", root)
