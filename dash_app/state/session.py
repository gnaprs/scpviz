"""Session storage with Redis backend and in-memory fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import pickle
import time
from threading import Lock
from typing import Any, Dict, Optional

try:
    from redis import Redis
except Exception:  # pragma: no cover - optional dependency in local dev
    Redis = None  # type: ignore[assignment]


@dataclass
class SessionState:
    """Mutable state tracked for each browser session."""

    pdata: Any = None
    uploads: Dict[str, str] = field(default_factory=dict)
    edited_svgs: Dict[str, str] = field(default_factory=dict)
    last_log: str = ""
    last_access_ts: float = field(default_factory=time.time)
    lock: Lock = field(default_factory=Lock, repr=False)


_SESSION_TTL_SECONDS = int(os.getenv("SCPVIZ_SESSION_TTL_SECONDS", str(6 * 60 * 60)))
_REDIS_URL = os.getenv("REDIS_URL", "").strip()
_REDIS_PREFIX = os.getenv("SCPVIZ_REDIS_PREFIX", "scpviz")

_SESSION_LOCK = Lock()
_SESSIONS: Dict[str, SessionState] = {}
_REDIS_CLIENT: Optional[Redis] = None
_USE_REDIS = False


def _state_to_payload(state: SessionState) -> bytes:
    payload = {
        "pdata": state.pdata,
        "uploads": state.uploads,
        "edited_svgs": state.edited_svgs,
        "last_log": state.last_log,
        "last_access_ts": state.last_access_ts,
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _payload_to_state(payload: bytes) -> SessionState:
    obj = pickle.loads(payload)
    return SessionState(
        pdata=obj.get("pdata"),
        uploads=dict(obj.get("uploads", {})),
        edited_svgs=dict(obj.get("edited_svgs", {})),
        last_log=str(obj.get("last_log", "")),
        last_access_ts=float(obj.get("last_access_ts", time.time())),
    )


def _redis_session_key(session_id: str) -> str:
    return f"{_REDIS_PREFIX}:session:{session_id}"


def _redis_lock_key(session_id: str) -> str:
    return f"{_REDIS_PREFIX}:lock:{session_id}"


def _init_redis() -> None:
    global _REDIS_CLIENT, _USE_REDIS
    if not _REDIS_URL or Redis is None:
        _USE_REDIS = False
        return
    try:
        client = Redis.from_url(_REDIS_URL, decode_responses=False)
        client.ping()
        _REDIS_CLIENT = client
        _USE_REDIS = True
    except Exception:
        _REDIS_CLIENT = None
        _USE_REDIS = False


_init_redis()


def _redis_get_state(session_id: str, create: bool) -> Optional[SessionState]:
    if _REDIS_CLIENT is None:
        return None
    raw = _REDIS_CLIENT.get(_redis_session_key(session_id))
    if raw is None:
        if not create:
            return None
        state = SessionState()
        _redis_set_state(session_id, state)
        return state
    try:
        state = _payload_to_state(raw)
    except Exception:
        state = SessionState()
    state.last_access_ts = time.time()
    _REDIS_CLIENT.expire(_redis_session_key(session_id), _SESSION_TTL_SECONDS)
    return state


def _redis_set_state(session_id: str, state: SessionState) -> None:
    if _REDIS_CLIENT is None:
        return
    state.last_access_ts = time.time()
    data = _state_to_payload(state)
    _REDIS_CLIENT.setex(_redis_session_key(session_id), _SESSION_TTL_SECONDS, data)


def _purge_expired_sessions_locked(now_ts: float) -> None:
    expired = [
        session_id
        for session_id, session in _SESSIONS.items()
        if (now_ts - session.last_access_ts) > _SESSION_TTL_SECONDS and not session.lock.locked()
    ]
    for session_id in expired:
        del _SESSIONS[session_id]


def _get_or_create_memory_session(session_id: str) -> SessionState:
    now_ts = time.time()
    _purge_expired_sessions_locked(now_ts)
    session = _SESSIONS.get(session_id)
    if session is None:
        session = SessionState()
        _SESSIONS[session_id] = session
    session.last_access_ts = now_ts
    return session


def _with_memory_session(session_id: str, fn):
    """Run function with per-session lock acquired without lock gap."""
    with _SESSION_LOCK:
        session = _get_or_create_memory_session(session_id)
        session.lock.acquire()
    try:
        session.last_access_ts = time.time()
        return fn(session)
    finally:
        session.lock.release()


def get_session(session_id: str) -> SessionState:
    """Return existing session state or create a new one."""
    if not session_id:
        raise ValueError("session_id is required")

    if _USE_REDIS and _REDIS_CLIENT is not None:
        lock = _REDIS_CLIENT.lock(_redis_lock_key(session_id), timeout=30, blocking_timeout=5)
        with lock:
            state = _redis_get_state(session_id, create=True)
            return state if state is not None else SessionState()

    with _SESSION_LOCK:
        return _get_or_create_memory_session(session_id)


def get_pdata(session_id: str) -> Optional[Any]:
    """Return pAnnData object for session if available."""
    if _USE_REDIS and _REDIS_CLIENT is not None:
        state = _redis_get_state(session_id, create=False)
        return state.pdata if state else None

    return _with_memory_session(session_id, lambda s: s.pdata)


def set_pdata(session_id: str, pdata: Any) -> None:
    """Store pAnnData object for a session."""
    if _USE_REDIS and _REDIS_CLIENT is not None:
        lock = _REDIS_CLIENT.lock(_redis_lock_key(session_id), timeout=60, blocking_timeout=10)
        with lock:
            state = _redis_get_state(session_id, create=True) or SessionState()
            state.pdata = pdata
            _redis_set_state(session_id, state)
        return

    def _set(session: SessionState) -> None:
        session.pdata = pdata

    _with_memory_session(session_id, _set)


def set_upload_path(session_id: str, key: str, file_path: str) -> None:
    """Store uploaded temp file path by logical key."""
    if _USE_REDIS and _REDIS_CLIENT is not None:
        lock = _REDIS_CLIENT.lock(_redis_lock_key(session_id), timeout=30, blocking_timeout=5)
        with lock:
            state = _redis_get_state(session_id, create=True) or SessionState()
            state.uploads[key] = file_path
            _redis_set_state(session_id, state)
        return

    def _set(session: SessionState) -> None:
        session.uploads[key] = file_path

    _with_memory_session(session_id, _set)


def get_upload_path(session_id: str, key: str) -> Optional[str]:
    """Get uploaded temp file path by logical key."""
    if _USE_REDIS and _REDIS_CLIENT is not None:
        state = _redis_get_state(session_id, create=False)
        return state.uploads.get(key) if state else None

    return _with_memory_session(session_id, lambda s: s.uploads.get(key))


def set_last_log(session_id: str, message: str) -> None:
    """Persist last operation message for UI display."""
    if _USE_REDIS and _REDIS_CLIENT is not None:
        lock = _REDIS_CLIENT.lock(_redis_lock_key(session_id), timeout=30, blocking_timeout=5)
        with lock:
            state = _redis_get_state(session_id, create=True) or SessionState()
            state.last_log = message
            _redis_set_state(session_id, state)
        return

    def _set(session: SessionState) -> None:
        session.last_log = message

    _with_memory_session(session_id, _set)


def get_last_log(session_id: str) -> str:
    """Return latest log message for session."""
    if _USE_REDIS and _REDIS_CLIENT is not None:
        state = _redis_get_state(session_id, create=False)
        return state.last_log if state else ""

    return _with_memory_session(session_id, lambda s: s.last_log)


def set_edited_svg(session_id: str, plot_key: str, svg_text: str) -> None:
    """Persist edited SVG markup for a plot key in a session."""
    if not plot_key:
        raise ValueError("plot_key is required")
    if _USE_REDIS and _REDIS_CLIENT is not None:
        lock = _REDIS_CLIENT.lock(_redis_lock_key(session_id), timeout=30, blocking_timeout=5)
        with lock:
            state = _redis_get_state(session_id, create=True) or SessionState()
            if svg_text:
                state.edited_svgs[plot_key] = svg_text
            else:
                state.edited_svgs.pop(plot_key, None)
            _redis_set_state(session_id, state)
        return

    def _set(session: SessionState) -> None:
        if svg_text:
            session.edited_svgs[plot_key] = svg_text
        else:
            session.edited_svgs.pop(plot_key, None)

    _with_memory_session(session_id, _set)


def get_edited_svg(session_id: str, plot_key: str) -> str:
    """Return edited SVG markup for a plot key if present."""
    if not plot_key:
        return ""
    if _USE_REDIS and _REDIS_CLIENT is not None:
        state = _redis_get_state(session_id, create=False)
        return state.edited_svgs.get(plot_key, "") if state else ""
    return _with_memory_session(session_id, lambda s: s.edited_svgs.get(plot_key, ""))

