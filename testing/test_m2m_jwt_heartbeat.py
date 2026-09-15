# BL-014g: engine attaches M2M Bearer on gated TM calls.
from __future__ import annotations

import json
from unittest.mock import patch

from src.manager_heartbeat import ManagerHeartbeat, ManagerHeartbeatConfig
from src.m2m_jwt import ClientCredentialsConfig, ClientCredentialsTokenSource


def test_regression_heartbeat_get_json_sends_bearer():
    src = ClientCredentialsTokenSource(
        ClientCredentialsConfig(
            token_url="http://idp.test/token",
            client_id="cid",
            username="tm-worker",
            password="secret",
        )
    )
    src._access_token = "machine-jwt-test"
    src._expires_at = 9e12

    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="55443322",
            m2m={"token_url": "x", "client_id": "y", "username": "z", "password": "w"},
        )
    )
    hb._token_source = src

    captured: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"onsets":[]}'

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _Resp()

    with patch("urllib.request.urlopen", fake_urlopen):
        out = hb.get_json("/api/instances/agent-1/session-health", timeout_s=1.0)
    assert out == {"onsets": []}
    assert captured["url"].endswith("/api/instances/agent-1/session-health")
    assert captured["headers"].get("authorization") == "Bearer machine-jwt-test"


def test_regression_m2m_from_env(monkeypatch):
    monkeypatch.setenv("TM_M2M_TOKEN_URL", "http://idp/token")
    monkeypatch.setenv("TM_M2M_CLIENT_ID", "cid")
    monkeypatch.setenv("TM_M2M_USERNAME", "tm-worker")
    monkeypatch.setenv("TM_M2M_PASSWORD", "pw")
    src = ClientCredentialsTokenSource.from_mapping({})
    assert src is not None
    assert src._cfg.username == "tm-worker"


def test_regression_m2m_loads_sibling_env_file(tmp_path, monkeypatch):
    import src.m2m_jwt as m2m

    monkeypatch.delenv("TM_M2M_TOKEN_URL", raising=False)
    monkeypatch.delenv("TM_M2M_CLIENT_ID", raising=False)
    monkeypatch.delenv("TM_M2M_USERNAME", raising=False)
    monkeypatch.delenv("TM_M2M_PASSWORD", raising=False)
    monkeypatch.delenv("TM_WORKER_M2M_USERNAME", raising=False)
    monkeypatch.delenv("TM_WORKER_M2M_PASSWORD", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TM_M2M_TOKEN_URL=http://127.0.0.1:9090/application/o/token/",
                "TM_M2M_CLIENT_ID=fleet-client",
                "TM_WORKER_M2M_USERNAME=tm-worker",
                "TM_WORKER_M2M_PASSWORD=sekrit",
                "TM_M2M_SCOPE='openid profile groups'",
                "UNRELATED=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    m2m._m2m_env_loaded = False
    # Point loader at tmp by monkeypatching candidates via cwd-relative trick:
    # load from an injected path by temporarily replacing load_m2m_env_files body path list.
    orig = m2m.load_m2m_env_files

    def _load(*, force: bool = False):
        m2m._m2m_env_loaded = True
        text = env_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, _, val = s.partition("=")
            key = key.strip()
            if key not in m2m._M2M_ENV_KEYS:
                continue
            if key in __import__("os").environ and str(
                __import__("os").environ.get(key) or ""
            ).strip():
                continue
            __import__("os").environ[key] = m2m._strip_env_value(val)
        return [str(env_file)]

    monkeypatch.setattr(m2m, "load_m2m_env_files", _load)
    src = ClientCredentialsTokenSource.from_mapping(None)
    assert src is not None
    assert src._cfg.client_id == "fleet-client"
    assert src._cfg.username == "tm-worker"
    assert src._cfg.password == "sekrit"
    assert "openid" in src._cfg.scope
    monkeypatch.setattr(m2m, "load_m2m_env_files", orig)
