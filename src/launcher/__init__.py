"""Launch ML engine as a black box and read what it leaves behind.

An engine (a TM Desktop, TM's own Execution Engine) hands ML engine a strict TM payload and a boot YAML, runs ``run_pipeline.py``
as a subprocess, tails the ledger it writes and turns the documents into step snapshots. The payload builder lives HERE, next to
the parser that judges it, and the ledger reader next to the document format, so they change together. What an engine does with
the snapshots (units, queues, mailboxes, placements) is the engine's own business and stays out of this module.
"""

from src.launcher.checkpoint import CheckpointConfigError, config_from_checkpoint_bytes, imitation_model_from_checkpoint_bytes, read_physical_version
from src.launcher.check import check_config, scan_plate
from src.launcher.early_stop import FitSpecError, apply_early_stop
from src.launcher.payload import FamilyFitSpec, SupervisedFitSpec, build_boot_yaml, build_pipeline_payload
from src.launcher.runner import MeFitError, run_me_fit
from src.launcher.staged import KIND_PIPELINE, KIND_RUN, KIND_SUPERVISED, load_staged, me_launch_options, plate_root, resolve_plate, spec_from_staged
from src.launcher.tail import LedgerTail

__all__ = [
    "CheckpointConfigError", "FamilyFitSpec", "FitSpecError", "LedgerTail", "MeFitError", "SupervisedFitSpec", "apply_early_stop",
    "build_boot_yaml", "build_pipeline_payload", "KIND_PIPELINE", "KIND_RUN", "KIND_SUPERVISED", "check_config", "config_from_checkpoint_bytes", "imitation_model_from_checkpoint_bytes", "load_staged", "me_launch_options", "scan_plate", "plate_root",
    "read_physical_version", "resolve_plate", "run_me_fit", "spec_from_staged",
]
