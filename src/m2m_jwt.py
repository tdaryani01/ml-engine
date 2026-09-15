# OAuth2 client_credentials → short-lived JWT (Authentik M2M for TM API).
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

_M2M_ENV_KEYS = (
    "TM_M2M_TOKEN_URL",
    "TM_M2M_CLIENT_ID",
    "TM_M2M_USERNAME",
    "TM_M2M_PASSWORD",
    "TM_M2M_SCOPE",
    "TM_WORKER_M2M_USERNAME",
    "TM_WORKER_M2M_PASSWORD",
    "OIDC_CLIENT_ID",
)
_m2m_env_loaded = False


def _strip_env_value(raw: str) -> str:
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def load_m2m_env_files(*, force: bool = False) -> list[str]:
    """Load TM_M2M_* from local / sibling training-manager .env (no overwrite).

    Host ``run_pipeline`` often has no M2M exports; secrets live in TM ``.env``
    after ``setup-fleet-m2m.sh``. Returns paths that contributed at least one key.
    """
    global _m2m_env_loaded
    if _m2m_env_loaded and not force:
        return []
    _m2m_env_loaded = True

    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / ".env",  # ml-engine/.env
        here.parents[2] / "training-manager" / ".env",  # sibling repo
    ]
    touched: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            _log.warning("m2m env read failed path=%s err=%s", path, exc)
            continue
        n = 0
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, _, val = s.partition("=")
            key = key.strip()
            if key not in _M2M_ENV_KEYS:
                continue
            if key in os.environ and str(os.environ.get(key) or "").strip():
                continue
            os.environ[key] = _strip_env_value(val)
            n += 1
        if n:
            touched.append(str(path))
            _log.info("m2m env loaded keys=%s from %s", n, path)
    return touched


@dataclass
class ClientCredentialsConfig:
    token_url: str
    client_id: str
    username: str
    password: str
    scope: str = "openid profile groups"
    skew_s: float = 60.0


class ClientCredentialsTokenSource:
    """Fetch + cache Authentik access tokens for the engine → TM API."""

    def __init__(self, cfg: ClientCredentialsConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    @classmethod
    def from_mapping(cls, raw: dict | None) -> ClientCredentialsTokenSource | None:
        """Build from training_manager.m2m dict or env (TM_M2M_*)."""
        load_m2m_env_files()
        raw = dict(raw or {})
        token_url = str(
            raw.get("token_url")
            or os.environ.get("TM_M2M_TOKEN_URL")
            or ""
        ).strip()
        # Host engines must not use Docker host-gateway URL.
        if "host.docker.internal" in token_url:
            token_url = token_url.replace("host.docker.internal", "127.0.0.1")
        client_id = str(
            raw.get("client_id")
            or os.environ.get("TM_M2M_CLIENT_ID")
            or os.environ.get("OIDC_CLIENT_ID")
            or ""
        ).strip()
        username = str(
            raw.get("username")
            or os.environ.get("TM_WORKER_M2M_USERNAME")
            or os.environ.get("TM_M2M_USERNAME")
            or ""
        ).strip()
        password = str(
            raw.get("password")
            or os.environ.get("TM_WORKER_M2M_PASSWORD")
            or os.environ.get("TM_M2M_PASSWORD")
            or ""
        ).strip()
        if not (token_url and client_id and username and password):
            return None
        scope = str(
            raw.get("scope") or os.environ.get("TM_M2M_SCOPE") or "openid profile groups"
        ).strip()
        return cls(
            ClientCredentialsConfig(
                token_url=token_url,
                client_id=client_id,
                username=username,
                password=password,
                scope=scope,
            )
        )

    def get_token(self) -> str:
        with self._lock:
            now = time.time()
            if self._access_token and now < self._expires_at - float(self._cfg.skew_s):
                return self._access_token
            token, expires_in = self._fetch()
            self._access_token = token
            self._expires_at = now + max(30.0, float(expires_in))
            return token

    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.get_token()}"}

    def _fetch(self) -> tuple[str, float]:
        body = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self._cfg.client_id,
                "username": self._cfg.username,
                "password": self._cfg.password,
                "scope": self._cfg.scope,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._cfg.token_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 — IdP URL
                raw = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"m2m token fetch failed: {exc}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError("m2m token response not an object")
        token = str(raw.get("access_token") or "").strip()
        if not token:
            raise RuntimeError(f"m2m token missing access_token: {raw.get('error')}")
        return token, float(raw.get("expires_in") or 300)
