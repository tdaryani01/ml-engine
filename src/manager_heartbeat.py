"""Non-blocking heartbeats to the sibling Training Manager control plane.

Never blocks the training thread: interval gate + at-most-one daemon worker.
Heartbeat JSON replies may include dial-out ``commands``; those are stashed on a
local queue for the engine to apply at safe points (not on the HTTP worker for
heavy restore work beyond enqueue).
"""
from __future__ import annotations

import json
import logging
import pickle
import queue
import random
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

_log = logging.getLogger(__name__)

# DIAGNOSTIC-ONLY: NDJSON claim/ledger trace. Path survives diagnostics_output deletes.
_DIAG_PATH = Path("/tmp/ml_engine_run_diag.ndjson")


def _diag(event: str, **fields: Any) -> None:
    rec = {"ts": time.time(), "event": event, **fields}
    try:
        with open(_DIAG_PATH, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def new_pool_session_id() -> str:
    """Anonymous pool worker id (numeric), not a durable agent / model id."""
    return str(random.randint(10_000_000, 99_999_999))


@dataclass(frozen=True)
class ManagerCommand:
    id: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass(frozen=True)
class ManagerHeartbeatConfig:
    enabled: bool = False
    uri: str = "http://127.0.0.1:8000"
    # Pool worker id only. Empty → random numeric at construct. Never a model id.
    instance_id: str = ""
    kind: str = "engine"
    label: str | None = None
    # Optional identity / debug only — control is dial-out via heartbeat.
    advertise_url: str = "http://127.0.0.1:0"
    capabilities: tuple[str, ...] = (
        "train_step",
        "ledger",
        "start",
        "pause",
        "resume",
        "restore",
        "shutdown",
        "cancel",
    )
    interval_s: float = 10.0
    timeout_s: float = 0.5
    idle_sleep_s: float = 10.0
    park_when_idle: bool = True


class ManagerHeartbeat:
    """TM client: pool capacity always; agent HB only while a job is leased.

    BL-006a / BL-011: workers register as pool sessions (random numeric id).
    A job already carries ``model_id`` (TM agent, set at enqueue). Claim binds
    this **worker** (pool session) to that job; the worker then heartbeats the
    agent and stamps the ledger with ``job.model_id`` — never with the pool id.
    """

    def __init__(self, cfg: ManagerHeartbeatConfig) -> None:
        pool_id = str(cfg.instance_id or "").strip() or new_pool_session_id()
        if pool_id != cfg.instance_id:
            cfg = ManagerHeartbeatConfig(
                enabled=cfg.enabled,
                uri=cfg.uri,
                instance_id=pool_id,
                kind=cfg.kind,
                label=cfg.label,
                advertise_url=cfg.advertise_url,
                capabilities=cfg.capabilities,
                interval_s=cfg.interval_s,
                timeout_s=cfg.timeout_s,
                idle_sleep_s=cfg.idle_sleep_s,
                park_when_idle=cfg.park_when_idle,
            )
        self._cfg = cfg
        self._lock = threading.Lock()
        self._in_flight = False
        self._registered = False
        self._pool_registered = False
        self._job_bound = False
        self._job_id: str | None = None
        self._bound_model_id: str | None = None
        self._claim_token: str | None = None
        self._last_sent_at = 0.0
        self._metrics: dict[str, Any] = {}
        self._commands: queue.Queue[ManagerCommand] = queue.Queue()
        self._pending_acks: queue.Queue[tuple[str, bool, str | None]] = queue.Queue()
        self._desired_state: str | None = None
        self._seen_command_ids: set[str] = set()
        self._active_checkpoint: dict[str, Any] | None = None

    @property
    def pool_session_id(self) -> str:
        return self._cfg.instance_id

    @property
    def bound_model_id(self) -> str | None:
        with self._lock:
            return self._bound_model_id

    @property
    def job_bound(self) -> bool:
        with self._lock:
            return self._job_bound

    @property
    def job_id(self) -> str | None:
        with self._lock:
            return self._job_id

    def bind_job(
        self, job_id: str | None = None, *, model_id: str | None = None
    ) -> None:
        """Record lease: this pool worker holds ``job_id`` for TM agent ``model_id``.

        ``model_id`` comes from the job document (enqueue); it is not the worker id.
        """
        mid = str(model_id or "").strip()
        with self._lock:
            self._job_bound = True
            self._job_id = None if job_id is None else str(job_id)
            self._bound_model_id = mid or None
            self._registered = False

    def unbind_job(self) -> None:
        """Release worker↔job lease after ack/fail — pool session remains."""
        with self._lock:
            self._job_bound = False
            self._job_id = None
            self._bound_model_id = None
            self._registered = False
            self._desired_state = None
            self._claim_token = None

    def try_claim_work(self, *, lease_s: float | None = None) -> dict[str, Any] | None:
        """Claim one queued job: bind **this worker** to the job (BL-006c/011).

        The job already has ``model_id`` (TM agent). Claim sets the job's worker
        to this pool session. Blocking HTTP — not for the HB thread.
        """
        if not self._cfg.enabled or not self._cfg.uri:
            return None
        with self._lock:
            if self._job_bound:
                return None
            sid = self._cfg.instance_id
        base = self._cfg.uri.rstrip("/")
        body: dict[str, Any] = {"session_id": sid}
        if lease_s is not None:
            body["lease_s"] = float(lease_s)
        resp = self._post_json(
            f"{base}/api/work/claim",
            body,
            expect_commands=True,
        )
        if not isinstance(resp, dict):
            _diag("claim_http_empty", pool=sid, kind=self._cfg.kind, label=self._cfg.label)
            return None
        job = resp.get("job")
        if not isinstance(job, dict) or not job.get("job_id"):
            _diag("claim_no_job", pool=sid, kind=self._cfg.kind, resp_keys=list(resp.keys()))
            return None
        model_id = str(job.get("model_id") or "").strip()
        if not model_id:
            _log.warning("claim missing model_id — refusing bind")
            _diag("claim_missing_model_id", pool=sid, job_id=job.get("job_id"))
            return None
        token = str(job.get("claim_token") or "")
        jid = str(job["job_id"])
        # Job public should already expose worker_id == this pool session.
        worker_id = str(job.get("worker_id") or job.get("session_id") or sid)
        with self._lock:
            self._claim_token = token or None
        self.bind_job(jid, model_id=model_id)
        self.set_metrics(
            {
                "run_state": "training",
                "state": "training",
                "job_id": jid,
                "model_id": model_id,
                "worker_id": worker_id,
            }
        )
        cfg_keys = sorted((job.get("config") or {}).keys()) if isinstance(job.get("config"), dict) else []
        _diag(
            "claim_ok",
            pool=sid,
            worker_id=worker_id,
            job_id=jid,
            model_id=model_id,
            kind=self._cfg.kind,
            label=self._cfg.label,
            job_kind=job.get("kind"),
            config_keys=cfg_keys,
            pipeline=((job.get("config") or {}) if isinstance(job.get("config"), dict) else {}).get("meta"),
        )
        return dict(job)

    def ack_work(
        self, *, result: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Ack the bound job and unbind agent HB."""
        with self._lock:
            jid = self._job_id
            token = self._claim_token
        if not jid or not token:
            return None
        base = self._cfg.uri.rstrip("/")
        resp = self._post_json(
            f"{base}/api/work/{urllib.parse.quote(jid, safe='')}/ack",
            {"claim_token": token, "result": dict(result or {})},
            expect_commands=True,
        )
        self.unbind_job()
        self.set_metrics({"run_state": "idle", "state": "idle"})
        return resp if isinstance(resp, dict) else None

    def fail_work(
        self, *, error: str, requeue: bool = True
    ) -> dict[str, Any] | None:
        """Fail the bound job and unbind agent HB."""
        with self._lock:
            jid = self._job_id
            token = self._claim_token
        if not jid or not token:
            return None
        base = self._cfg.uri.rstrip("/")
        resp = self._post_json(
            f"{base}/api/work/{urllib.parse.quote(jid, safe='')}/fail",
            {"claim_token": token, "error": str(error), "requeue": bool(requeue)},
            expect_commands=True,
        )
        self.unbind_job()
        self.set_metrics({"run_state": "idle", "state": "failure"})
        return resp if isinstance(resp, dict) else None

    @property
    def desired_state(self) -> str | None:
        with self._lock:
            return self._desired_state

    def set_desired_state(self, desired: str | None) -> None:
        with self._lock:
            self._desired_state = None if desired is None else str(desired)

    @property
    def active_checkpoint(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._active_checkpoint is None else dict(self._active_checkpoint)

    def set_active_checkpoint(self, active: dict[str, Any] | None) -> None:
        """Local tip after restore — avoids stale HB active fighting the next auto/drain."""
        with self._lock:
            self._active_checkpoint = None if active is None else dict(active)

    def set_metrics(self, metrics: Mapping[str, Any] | None) -> None:
        if not metrics:
            return
        with self._lock:
            self._metrics.update(dict(metrics))

    def poll_commands(self) -> list[ManagerCommand]:
        """Non-blocking drain; skips ids already handled in this process."""
        out: list[ManagerCommand] = []
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                if cmd.id in self._seen_command_ids:
                    continue
            out.append(cmd)
        return out

    def mark_command_seen(self, command_id: str) -> None:
        with self._lock:
            self._seen_command_ids.add(command_id)

    def queue_ack(self, command_id: str, *, ok: bool, detail: str | None = None) -> None:
        self._pending_acks.put((command_id, ok, detail))

    def maybe_ping(self, *, force: bool = False) -> bool:
        """Schedule a ping if due (or always when force=True). Returns True if started."""
        if not self._cfg.enabled or not self._cfg.uri:
            return False
        now = time.monotonic()
        with self._lock:
            if self._in_flight:
                return False
            if not force and (now - self._last_sent_at) < float(self._cfg.interval_s):
                return False
            self._in_flight = True
            self._last_sent_at = now
            metrics = dict(self._metrics)
            job_bound = self._job_bound
            need_register = job_bound and not self._registered
            need_pool = not self._pool_registered
        t = threading.Thread(
            target=self._worker,
            args=(need_register, need_pool, metrics, job_bound),
            name="tm-heartbeat",
            daemon=True,
        )
        t.start()
        return True

    @property
    def interval_s(self) -> float:
        return float(self._cfg.interval_s)

    @property
    def cfg(self) -> ManagerHeartbeatConfig:
        return self._cfg

    def fetch_blob(self, blob_key: str) -> bytes | None:
        """Blocking blob GET — call only from a worker / safe-point helper, not hot path."""
        base = self._cfg.uri.rstrip("/")
        key = urllib.parse.quote(blob_key, safe="/")
        url = f"{base}/api/ledger/blobs/{key}"
        req = urllib.request.Request(url, method="GET", headers={"Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=max(5.0, float(self._cfg.timeout_s))) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            _log.warning("GET blob %s failed: %s", blob_key, exc)
            return None

    def put_blob(self, blob_key: str, data: bytes, *, timeout_s: float | None = None) -> bool:
        """Blocking blob PUT — upload checkpoint bytes before ledger doc append."""
        if not self._cfg.enabled or not self._cfg.uri:
            return False
        base = self._cfg.uri.rstrip("/")
        key = urllib.parse.quote(blob_key, safe="/")
        url = f"{base}/api/ledger/blobs/{key}"
        req = urllib.request.Request(
            url,
            data=bytes(data),
            method="PUT",
            headers={
                "Content-Type": "application/octet-stream",
                "Accept": "application/json",
            },
        )
        wait = float(timeout_s if timeout_s is not None else max(5.0, float(self._cfg.timeout_s)))
        try:
            with urllib.request.urlopen(req, timeout=wait) as resp:
                resp.read()
                return 200 <= int(resp.status) < 300
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            _log.warning("PUT blob %s failed: %s", blob_key, exc)
            return False

    def post_json(
        self, path: str, body: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> dict[str, Any] | None:
        """Blocking JSON POST to TM API (shadow/decide/outcome)."""
        if not self._cfg.enabled or not self._cfg.uri:
            return None
        base = self._cfg.uri.rstrip("/")
        url = f"{base}{path}" if path.startswith("/") else f"{base}/{path}"
        wait = float(timeout_s if timeout_s is not None else max(2.0, float(self._cfg.timeout_s)))
        data = json.dumps(dict(body)).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=wait) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                parsed = json.loads(raw.decode("utf-8"))
                return parsed if isinstance(parsed, dict) else {}
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, json.JSONDecodeError) as exc:
            _log.warning("POST %s failed: %s", path, exc)
            return None

    def get_json(self, path: str, *, timeout_s: float | None = None) -> dict[str, Any] | None:
        """Blocking JSON GET (session-health / onset)."""
        if not self._cfg.enabled or not self._cfg.uri:
            return None
        base = self._cfg.uri.rstrip("/")
        url = f"{base}{path}" if path.startswith("/") else f"{base}/{path}"
        wait = float(timeout_s if timeout_s is not None else max(2.0, float(self._cfg.timeout_s)))
        req = urllib.request.Request(
            url,
            method="GET",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=wait) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                parsed = json.loads(raw.decode("utf-8"))
                return parsed if isinstance(parsed, dict) else {}
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, json.JSONDecodeError) as exc:
            _log.warning("GET %s failed: %s", path, exc)
            return None

    def append_ledger_doc(
        self,
        *,
        doc_type: str,
        body: Mapping[str, Any],
        branch_id: str = "main",
        blob_key: str | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """Blocking POST to TM shared ledger. Returns True on HTTP success."""
        if not self._cfg.enabled or not self._cfg.uri:
            return False
        base = self._cfg.uri.rstrip("/")
        with self._lock:
            bound_model = self._bound_model_id
            job_id = self._job_id
            pool_id = self._cfg.instance_id
        # Tape belongs to the TM agent (job.model_id), not the pool worker id.
        instance_id = bound_model or pool_id
        payload: dict[str, Any] = {
            "instance_id": instance_id,
            "doc_type": str(doc_type),
            "body": dict(body),
            "branch_id": str(branch_id or "main"),
        }
        if blob_key:
            payload["blob_key"] = str(blob_key)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/ledger/docs",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        wait = float(timeout_s if timeout_s is not None else max(2.0, float(self._cfg.timeout_s)))
        try:
            with urllib.request.urlopen(req, timeout=wait) as resp:
                resp.read()
                ok = 200 <= int(resp.status) < 300
                _diag(
                    "ledger_append",
                    ok=ok,
                    doc_type=doc_type,
                    post_instance_id=instance_id,
                    bound_model_id=bound_model,
                    job_id=job_id,
                    version=(body or {}).get("version"),
                    blob_key=blob_key,
                    mismatch=bool(bound_model and bound_model != instance_id),
                )
                return ok
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            _log.warning("ledger append %s failed: %s", doc_type, exc)
            _diag(
                "ledger_append_fail",
                doc_type=doc_type,
                post_instance_id=instance_id,
                bound_model_id=bound_model,
                job_id=job_id,
                error=str(exc),
            )
            return False

    def _worker(
        self,
        need_register: bool,
        need_pool: bool,
        metrics: dict[str, Any],
        job_bound: bool = False,
    ) -> None:
        cfg = self._cfg
        base = cfg.uri.rstrip("/")
        try:
            # Pool capacity always. Agent HB only while job-bound to a TM model_id.
            self._sync_pool_session(base, need_pool=need_pool, metrics=metrics)
            # Pause/cancel may have released the job during pool HB.
            if not self.job_bound:
                return
            if not job_bound:
                return
            with self._lock:
                model_id = self._bound_model_id
            if not model_id:
                _log.warning("job-bound heartbeat without model_id — skip agent HB")
                return
            self._flush_acks(base)
            # Agent must already exist in TM (initialized there). Heartbeat only —
            # do not invent / upsert an agent from the worker process.
            if need_register:
                with self._lock:
                    self._registered = True
            body = self._post_json(
                f"{base}/api/instances/{urllib.parse.quote(model_id, safe='')}/heartbeat",
                {"metrics": metrics},
                expect_commands=True,
            )
            if isinstance(body, dict):
                desired = body.get("desired_state")
                if desired is not None:
                    with self._lock:
                        self._desired_state = str(desired) if desired else None
                active = body.get("active_checkpoint")
                if isinstance(active, dict):
                    with self._lock:
                        self._active_checkpoint = dict(active)
                elif "active_checkpoint" in body:
                    with self._lock:
                        self._active_checkpoint = None
                for raw in body.get("commands") or []:
                    if not isinstance(raw, dict):
                        continue
                    cid = str(raw.get("id") or "")
                    action = str(raw.get("action") or "")
                    if not cid or not action:
                        continue
                    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
                    self._commands.put(
                        ManagerCommand(
                            id=cid,
                            action=action,
                            payload=dict(payload or {}),
                            created_at=float(raw.get("created_at") or 0.0),
                        )
                    )
        except Exception as exc:  # noqa: BLE001 — fire-and-forget
            _log.debug("training-manager heartbeat failed: %s", exc)
        finally:
            with self._lock:
                self._in_flight = False

    @staticmethod
    def _pool_status_from_metrics(metrics: Mapping[str, Any]) -> str:
        raw = metrics.get("run_state") or metrics.get("state") or ""
        rs = str(raw).strip().lower()
        if rs in ("training", "running", "busy"):
            return "busy"
        return "idle"

    def _sync_pool_session(
        self, base: str, *, need_pool: bool, metrics: Mapping[str, Any]
    ) -> None:
        """Register + heartbeat this process as a TM pool worker (capacity only)."""
        cfg = self._cfg
        sid = cfg.instance_id
        caps = ["pool", str(cfg.kind or "engine")]
        for c in cfg.capabilities:
            if c not in caps:
                caps.append(str(c))
        if need_pool:
            ok = self._post_json(
                f"{base}/api/workers/register",
                {
                    "session_id": sid,
                    "caps": caps,
                    "meta": {"kind": cfg.kind, "label": cfg.label or f"pool-{sid}"},
                },
                expect_commands=False,
            )
            if ok is not None:
                with self._lock:
                    self._pool_registered = True
        status = self._pool_status_from_metrics(metrics)
        meta: dict[str, Any] = {"kind": cfg.kind}
        with self._lock:
            jid = self._job_id
        if jid:
            meta["job_id"] = jid
        body = self._post_json(
            f"{base}/api/workers/{urllib.parse.quote(sid, safe='')}/heartbeat",
            {"status": status, "meta": meta},
            expect_commands=True,
        )
        if body is None:
            # Session lost (TTL / restart) — re-register next ping.
            with self._lock:
                self._pool_registered = False
            return
        if isinstance(body, dict) and body.get("should_exit"):
            _log.info(
                "training-manager pool drain requested for session=%s", sid
            )
        if isinstance(body, dict) and body.get("release_job") and self.job_bound:
            # BL-006d: TM parked/cancelled — drop agent bind; pool stays.
            _log.info(
                "training-manager release_job for session=%s job=%s", sid, jid
            )
            self.unbind_job()
            self.set_metrics({"run_state": "idle", "state": "paused"})

    def _flush_acks(self, base: str) -> None:
        with self._lock:
            model_id = self._bound_model_id
        if not model_id:
            while True:
                try:
                    self._pending_acks.get_nowait()
                except queue.Empty:
                    return
            return
        while True:
            try:
                cid, ok, detail = self._pending_acks.get_nowait()
            except queue.Empty:
                return
            payload: dict[str, Any] = {"ok": ok}
            if detail:
                payload["detail"] = detail
            self._post_json(
                f"{base}/api/instances/{urllib.parse.quote(model_id, safe='')}/commands/{cid}/ack",
                payload,
                expect_commands=False,
            )

    def _post_json(
        self, url: str, body: dict[str, Any], *, expect_commands: bool
    ) -> dict[str, Any] | bool | None:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self._cfg.timeout_s)) as resp:
                raw = resp.read()
                if not expect_commands:
                    return True
                if not raw:
                    return {}
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            _log.debug("POST %s failed: %s", url, exc)
            return None


def decode_checkpoint_blob(data: bytes) -> dict[str, Any]:
    """Unpack importer checkpoint blob into a restore_model_checkpoint body.

    Blobs are wire-encoded (raw ndarray buffers) so numpy 1.x/2.x venvs interop.
    Legacy plain pickles of ndarray objects still attempt to load for old MinIO keys.
    """
    obj = pickle.loads(data)
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint blob is {type(obj)!r}, expected dict")
    return _from_wire(obj)


def encode_checkpoint_blob(body: Mapping[str, Any]) -> bytes:
    """Pickle a checkpoint body with ndarray wire encoding for MinIO."""
    return pickle.dumps(_to_wire(dict(body)), protocol=pickle.HIGHEST_PROTOCOL)


def _to_wire(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data": arr.tobytes(),
        }
    if isinstance(value, dict):
        return {str(k): _to_wire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_wire(v) for v in value]
    if isinstance(value, tuple):
        return [_to_wire(v) for v in value]
    return value


def _from_wire(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            import numpy as np

            dtype = np.dtype(value["dtype"])
            shape = tuple(int(x) for x in value["shape"])
            buf = value["data"]
            if not isinstance(buf, (bytes, bytearray, memoryview)):
                raise TypeError("ndarray wire data must be bytes")
            return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return {str(k): _from_wire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_wire(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_from_wire(v) for v in value)
    return value


def maybe_from_settings(
    settings: Any | None,
    *,
    ledger_enabled: bool = True,
) -> ManagerHeartbeat | None:
    """Build a client from TrainingManagerSettings / mapping, or None if disabled."""
    if settings is None:
        return None
    if hasattr(settings, "enabled"):
        enabled = bool(settings.enabled)
        uri = str(getattr(settings, "uri", "") or "")
        instance_id = str(getattr(settings, "instance_id", "") or "")
        kind = str(getattr(settings, "kind", "engine"))
        label = getattr(settings, "label", None)
        advertise_url = str(getattr(settings, "advertise_url", "http://127.0.0.1:0"))
        caps = tuple(
            getattr(
                settings,
                "capabilities",
                ("train_step", "ledger", "start", "pause", "resume", "restore", "shutdown", "cancel"),
            )
            or ()
        )
        interval_s = float(getattr(settings, "interval_s", 10.0))
        timeout_s = float(getattr(settings, "timeout_s", 0.5))
        idle_sleep_s = float(getattr(settings, "idle_sleep_s", 10.0))
        park_when_idle = bool(getattr(settings, "park_when_idle", True))
    elif isinstance(settings, Mapping):
        enabled = bool(settings.get("enabled", False))
        uri = str(settings.get("uri", "") or "")
        instance_id = str(settings.get("instance_id", "") or "")
        kind = str(settings.get("kind", "engine"))
        label = settings.get("label")
        advertise_url = str(settings.get("advertise_url", "http://127.0.0.1:0"))
        caps = tuple(
            settings.get("capabilities")
            or (
                "train_step",
                "ledger",
                "start",
                "pause",
                "resume",
                "restore",
                "shutdown",
                "cancel",
            )
        )
        interval_s = float(settings.get("interval_s", 10.0))
        timeout_s = float(settings.get("timeout_s", 0.5))
        idle_sleep_s = float(settings.get("idle_sleep_s", 10.0))
        park_when_idle = bool(settings.get("park_when_idle", True))
    else:
        return None
    if not enabled or not uri:
        return None

    pool_id = instance_id.strip() or new_pool_session_id()
    base_label = str(label) if label else f"pool-{pool_id}"
    if ledger_enabled:
        final_caps = caps if caps else (
            "train_step",
            "ledger",
            "start",
            "pause",
            "resume",
            "restore",
            "shutdown",
            "cancel",
        )
        final_label = base_label
        seed_metrics: dict[str, Any] = {"ledger": "on"}
    else:
        final_caps = tuple(c for c in caps if c != "ledger") or (
            "train_step",
            "start",
            "pause",
            "resume",
            "restore",
            "shutdown",
            "cancel",
        )
        final_label = (
            base_label
            if "no ledger" in base_label.lower()
            else f"{base_label} (no ledger)"
        )
        seed_metrics = {"ledger": "off"}

    # Ensure control caps are advertised when TM is enabled.
    for need in ("start", "pause", "resume", "restore", "shutdown", "cancel"):
        if need not in final_caps:
            final_caps = final_caps + (need,)

    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri=uri,
            instance_id=pool_id,
            kind=kind,
            label=final_label,
            advertise_url=advertise_url,
            capabilities=final_caps,
            interval_s=interval_s,
            timeout_s=timeout_s,
            idle_sleep_s=idle_sleep_s,
            park_when_idle=park_when_idle,
        )
    )
    hb.set_metrics(seed_metrics)
    return hb
