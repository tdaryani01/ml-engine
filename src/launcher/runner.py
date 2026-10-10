"""Run one supervised fit in ME as a black box and turn its ledger into supervisor snapshots.

EE stages a payload, a boot YAML and (optionally) a starting checkpoint, launches ME's
``run_pipeline.py`` as a subprocess, and tails the ledger ME writes. EE never steps ME and never
decides early stop: ME's ``run.end`` document says how the fit ended.

Snapshot contract (what the supervisor already consumes from ``run_ml_engine_train``):
``{step, epoch, loss, train_loss, phase, version[, checkpoint_bytes]}`` with ``phase`` one of
``train`` | ``es_trip`` | ``stopped``. ``version`` is the ABSOLUTE trajectory (``base_traj`` +
ME's per-fit version), and every announced checkpoint is copied out at announcement time with its
absolute version, so ME's own checkpoint pruning can never dangle a handle.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from src.launcher.payload import SupervisedFitSpec, build_boot_yaml, build_pipeline_payload
from src.launcher.tail import LedgerTail

# On Stop there is nothing to wait for: every announced checkpoint was already copied out. Ask
# politely, then kill, so Stop stays as prompt as the legacy loop.
_STOP_GRACE_S = 0.5
_EXIT_GRACE_S = 15.0


ML_ENGINE_ROOT = Path(__file__).resolve().parents[2]


class MeFitError(RuntimeError):
    """ME exited without recording how the fit ended."""


def _restamp(doc: Any, absolute_version: int) -> bytes:
    """Checkpoint document bytes carrying the ABSOLUTE version (what restore reads back)."""
    from src.ledger import document_to_bytes

    body = dict(doc.body)
    body["version"] = int(absolute_version)
    return document_to_bytes(dataclasses.replace(doc, body=body, version=int(absolute_version)))


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=_EXIT_GRACE_S)
        except subprocess.TimeoutExpired:
            pass  # a process slow to die is the OS's to reap; it must not turn a fit that already finished into a failed one


def run_me_fit(
    spec: SupervisedFitSpec,
    *,
    work_dir: Path,
    should_stop: Callable[[], bool],
    checkpoint_path: Path | None = None,
    base_traj: int = 0,
    extra_env: dict[str, str] | None = None,
    python: str | None = None,
    poll_s: float = 0.02,
    max_wall_s: float | None = None,
) -> Iterator[dict[str, Any]]:
    root = ML_ENGINE_ROOT  # the checkout this module lives in
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(parents=True, exist_ok=True)

    restore: Path | None = None
    if checkpoint_path is not None:
        restore = work / "restore.ckpt"
        restore.write_bytes(Path(checkpoint_path).read_bytes())
    payload_path = work / "payload.json"
    payload_path.write_text(json.dumps(build_pipeline_payload(spec, work_dir=work, restore_checkpoint_path=restore)))
    boot_path = work / "boot.yaml"
    boot_path.write_text(build_boot_yaml(work))

    env = dict(os.environ)
    env.update(extra_env or {})
    env["ML_ENGINE_TM_PAYLOAD"] = str(payload_path)
    env["ML_ENGINE_BOOT_YAML"] = str(boot_path)
    env["ML_ENGINE_CONFIG_SOURCE"] = "training_manager"
    env["PYTHONUNBUFFERED"] = "1"

    log_path = work / "me.log"
    log = open(log_path, "wb")  # noqa: SIM115 - closed in finally
    proc = subprocess.Popen(  # noqa: S603
        [python or sys.executable, "run_pipeline.py"],
        cwd=str(root),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    tail = LedgerTail(work / "out" / "ledger")
    step = 0
    epoch = 0
    last_train = 1.0
    last_val: float | None = None
    last_extra: dict[str, float] | None = None  # named numbers a family reports beside the losses (held-out accuracy, arrival rate, ...)
    last_version = int(base_traj)
    started = time.monotonic()
    ended: dict[str, Any] | None = None
    pending_cp: dict[int, Any] = {}  # relative version -> checkpoint doc (cadence/final candidates)
    held: dict[str, Any] | None = None
    held_rel = -1
    saw_complete = False  # a supervised fit writes step.complete per step; a closed-loop fit writes only step.metrics

    def _snap(phase: str, rel_version: int, doc: Any | None = None) -> dict[str, Any]:
        s: dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "loss": float(last_val if last_val is not None else last_train),
            "train_loss": float(last_train),
            "phase": phase,
            "version": int(base_traj) + int(rel_version),
            "val_measured": last_val is not None,
        }
        if last_extra:
            s["extra_metrics"] = dict(last_extra)
        # A checkpoint is published only with a validation measurement behind it. A supervised fit validates at the end of an epoch, so a cadence checkpoint written before the first epoch of this
        # launch has ended has none, and "loss" above is then the training loss standing in for it: published, it looked like a validation loss far below every real one and was picked as the best checkpoint.
        if doc is not None and last_val is not None:
            s["checkpoint_bytes"] = _restamp(doc, int(base_traj) + int(rel_version))
        return s

    try:
        while True:
            if should_stop():
                _terminate(proc)
                yield _snap("stopped", max(0, last_version - int(base_traj)))
                return
            if max_wall_s is not None and time.monotonic() - started > max_wall_s:
                _terminate(proc)
                raise MeFitError(f"ME fit exceeded max_wall_s={max_wall_s}")

            docs = tail.poll()
            for d in docs:
                # ME writes far faster than the supervisor consumes: a Stop must not wait for the
                # backlog of already-written records, so it is checked per record, not per batch.
                if should_stop():
                    _terminate(proc)
                    yield _snap("stopped", max(0, last_version - int(base_traj)))
                    return
                t = d.doc_type
                if t == "step.complete" or (t == "step.metrics" and not saw_complete):
                    # Flush the previous step (it can no longer receive a checkpoint), hold this one:
                    # ME writes a cadence checkpoint just AFTER the step that produced it.
                    if t == "step.complete":
                        saw_complete = True
                    if held is not None:
                        yield held
                    step += 1
                    m = (d.body.get("metrics") or {}) if t == "step.complete" else d.body
                    last_train = float(m.get("train_loss", last_train))
                    if t == "step.metrics" and m.get("val_loss") is not None:
                        last_val = float(m["val_loss"])
                        epoch += 1
                    if t == "step.metrics" and isinstance(m.get("extra_metrics"), dict):
                        last_extra = {str(k): float(v) for k, v in m["extra_metrics"].items()}
                    rel = int(d.version or step)
                    last_version = int(base_traj) + rel
                    cp = pending_cp.pop(rel, None)
                    held = _snap("train", rel, cp if cp is not None and last_version % spec.checkpoint_every == 0 else None)
                    held_rel = rel
                elif t == "step.metrics":
                    epoch += 1
                    v = d.body.get("val_loss")
                    if v is not None:
                        last_val = float(v)
                    if isinstance(d.body.get("extra_metrics"), dict):
                        last_extra = {str(k): float(x) for k, x in d.body["extra_metrics"].items()}
                elif t == "checkpoint":
                    rel = int(d.version or 0)
                    if rel <= 0:
                        continue
                    abs_v = int(base_traj) + rel
                    if held is not None and held_rel == rel and abs_v % spec.checkpoint_every == 0:
                        if held.get("val_measured"):
                            held["checkpoint_bytes"] = _restamp(d, abs_v)
                    else:
                        pending_cp[rel] = d
                        if len(pending_cp) > 8:
                            pending_cp.pop(min(pending_cp))
                elif t == "run.end":
                    ended = dict(d.body)
            if held is not None and not docs:
                yield held  # idle: do not sit on a step snapshot (live latency)
                held = None
            if ended is not None:
                break
            if proc.poll() is not None and not docs:
                # Process gone: drain once more, then decide.
                late = tail.poll()
                if not late:
                    break
                continue
            time.sleep(poll_s)

        if ended is None:
            tail_text = log_path.read_text(errors="replace")[-1500:]
            raise MeFitError(f"ME exited (code {proc.poll()}) without a run.end document:\n{tail_text}")

        if held is not None:
            yield held
            held = None
        reason = str(ended.get("reason") or "success")
        trip_rel = max(0, last_version - int(base_traj))
        if reason == "es_trip":
            # ME restored its BEST weights. Announce them at the trip step's version so the version
            # axis never runs backwards; the reported loss is the best validation loss.
            doc = _checkpoint_doc(work / "out" / "ledger", pending_cp, ended.get("best_version"))
            best_val = ended.get("best_val_loss")
            if best_val is not None:
                last_val = float(best_val)
            if ended.get("published_train_loss") is not None:  # the train loss of the SAME round as the val loss above
                last_train = float(ended["published_train_loss"])
            yield {**_snap("es_trip", trip_rel, doc), "run_end": ended}
        else:
            final_rel = ended.get("final_version")
            doc = _checkpoint_doc(work / "out" / "ledger", pending_cp, final_rel)
            if ended.get("published_val_loss") is not None and ended.get("published_train_loss") is not None:
                # A family whose final checkpoint holds its BEST weights reports that round's losses (not the last round's).
                last_val, last_train = float(ended["published_val_loss"]), float(ended["published_train_loss"])
            yield {**_snap("train", int(final_rel) if final_rel is not None else trip_rel, doc), "run_end": ended}
    finally:
        _terminate(proc)
        log.close()


def _checkpoint_doc(ledger_dir: Path, pending: dict[int, Any], rel_version: Any) -> Any | None:
    """The checkpoint document for ``rel_version`` (memory window first, then ME's checkpoint file)."""
    if rel_version is None:
        return None
    rel = int(rel_version)
    if rel in pending:
        return pending[rel]
    path = ledger_dir / "checkpoints" / f"main_v{rel}.bin"
    if not path.is_file():
        return None
    from src.ledger import document_from_bytes
    from src.ledger_wire import iter_framed_records

    records = iter_framed_records(path.read_bytes())
    return document_from_bytes(records[0]) if records else None
