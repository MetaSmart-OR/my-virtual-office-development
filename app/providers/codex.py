"""Codex App Server provider adapter for My Virtual Office.

Uses only the public CLI surface: `codex app-server --stdio`.
No Codex internal files are read.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


# ── Module-level subprocess state ────────────────────────────────────────────

_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()   # guards _proc start/reset
_turn_lock = threading.Lock()   # one active turn at a time
_active_thread_id: str | None = None  # thread ID of the current/last turn
_pending_approvals: dict[str, dict] = {}  # itemId → {"params": ..., "rpc_id": ...}


def _get_proc() -> subprocess.Popen | None:
    return _proc


def _reset_proc() -> None:
    global _proc
    _proc = None


def _start_proc(binary: str) -> subprocess.Popen | None:
    """Launch `codex app-server --stdio` and complete the initialize handshake."""
    global _proc
    try:
        p = subprocess.Popen(
            [binary, "app-server", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError:
        return None

    init_msg = (
        json.dumps({
            "method": "initialize",
            "id": 0,
            "params": {"clientInfo": {"name": "my-virtual-office", "version": "1.0"}},
        }) + "\n"
    ).encode("utf-8")

    try:
        p.stdin.write(init_msg)
    except OSError:
        p.kill()
        return None

    # Read stdout until we get the id=0 response (10s deadline)
    fd = p.stdout.fileno()
    buf = bytearray()
    deadline = time.time() + 10

    while time.time() < deadline:
        if p.poll() is not None:
            return None
        ready = select.select([fd], [], [], 0.5)[0]
        if ready:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf.extend(chunk)

        while b"\n" in buf:
            idx = buf.index(b"\n")
            line = bytes(buf[:idx]).strip()
            del buf[:idx + 1]
            if not line:
                continue
            try:
                resp = json.loads(line)
                if resp.get("id") == 0:
                    return p
            except json.JSONDecodeError:
                continue

    try:
        p.kill()
    except Exception:
        pass
    return None


_DEBUG = os.environ.get("VO_CODEX_DEBUG", "").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        import sys
        print(f"[CODEX-DEBUG][{tag}] {msg}", file=sys.stderr, flush=True)


def _send_rpc(proc: subprocess.Popen, method: str, params: dict, req_id: int) -> None:
    msg = (json.dumps({"method": method, "id": req_id, "params": params}) + "\n").encode("utf-8")
    _dbg("SEND", f"id={req_id} method={method} params={json.dumps(params)[:200]}")
    proc.stdin.write(msg)


# ── Provider dataclass ────────────────────────────────────────────────────────

@dataclass
class CodexProvider:
    """Provider adapter for the OpenAI Codex CLI App Server."""

    binary: str | None = None
    home: str | None = None
    enabled: bool = True

    provider_kind: str = "codex"
    provider_type: str = "runtime"

    def __post_init__(self) -> None:
        self.binary = os.path.expanduser(
            self.binary
            or os.environ.get("VO_CODEX_BIN")
            or shutil.which("codex")
            or "codex"
        )
        self.home = os.path.expanduser(
            self.home
            or os.environ.get("VO_CODEX_HOME")
            or "~/.codex"
        )

    def _resolved_binary(self) -> str | None:
        b = self.binary
        if not b:
            return None
        if os.path.isabs(b):
            return b if os.path.isfile(b) and os.access(b, os.X_OK) else None
        found = shutil.which(b)
        return found if found and os.access(found, os.X_OK) else None

    def is_available(self) -> bool:
        return bool(self.enabled and self._resolved_binary())

    def discover_agents(self) -> list[dict[str, Any]]:
        if not self.is_available():
            return []
        return [{
            "id": "codex",
            "statusKey": "codex",
            "providerKind": "codex",
            "providerType": "runtime",
            "name": "Codex",
            "emoji": "⚡",
            "role": "Coding Agent",
            "model": "gpt-5.3-codex",
            "workspace": self.home or "",
            "lastActiveAt": None,
            "capabilities": ["chat", "status"],
        }]

    def test(self) -> dict[str, Any]:
        binary = self._resolved_binary()
        if not binary:
            return {
                "ok": False,
                "version": "",
                "authenticated": False,
                "authMethod": None,
                "error": f"Codex binary not found: {self.binary}",
            }

        version = ""
        try:
            r = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = ((r.stdout or "") + (r.stderr or "")).strip().split("\n")[0].strip()
        except Exception as exc:
            return {
                "ok": False,
                "version": "",
                "authenticated": False,
                "authMethod": None,
                "error": str(exc),
            }

        authenticated = False
        auth_method = None

        if os.environ.get("OPENAI_API_KEY"):
            authenticated = True
            auth_method = "api_key"
        elif self.home and os.path.isdir(self.home):
            fpath = os.path.join(self.home, "auth.json")
            if os.path.isfile(fpath):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                    # OAuth login: tokens.access_token present
                    tokens = data.get("tokens") or {}
                    if isinstance(tokens, dict) and tokens.get("access_token"):
                        authenticated = True
                        auth_method = "oauth"
                    # API key stored in auth.json by `codex login --api-key`
                    elif data.get("OPENAI_API_KEY") or data.get("apiKey") or data.get("api_key"):
                        authenticated = True
                        auth_method = "api_key"
                except Exception:
                    pass

        error = None
        if not authenticated:
            error = (
                "Codex binary found but not authenticated. "
                "Run 'codex login' on the host or set OPENAI_API_KEY."
            )

        return {
            "ok": authenticated,
            "version": version,
            "authenticated": authenticated,
            "authMethod": auth_method,
            "error": error,
        }

    def get_or_create_thread_id(self, agent_key: str, status_dir: str) -> str:
        path = os.path.join(status_dir, "codex-threads.json")
        try:
            with open(path, encoding="utf-8") as f:
                threads = json.load(f)
        except (OSError, json.JSONDecodeError):
            threads = {}

        return threads.get(agent_key)  # None for new threads; server assigns ID on first turn

    def _update_thread_id(self, agent_key: str, thread_id: str, status_dir: str) -> None:
        path = os.path.join(status_dir, "codex-threads.json")
        threads: dict[str, str] = {}
        try:
            with open(path, encoding="utf-8") as f:
                threads = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        threads[agent_key] = thread_id
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(threads, f)
        except OSError:
            pass

    def get_history(self, thread_id: str) -> list[dict[str, Any]]:
        if not thread_id or not self.home:
            return []
        sessions_base = os.path.join(self.home, "sessions")
        if not os.path.isdir(sessions_base):
            return []

        # Filename format: rollout-<datetime>-<threadId>.jsonl
        thread_file = None
        for root, _, files in os.walk(sessions_base):
            for fname in files:
                if fname.endswith(f"-{thread_id}.jsonl") or fname == f"{thread_id}.jsonl":
                    thread_file = os.path.join(root, fname)
                    break
            if thread_file:
                break
        if not thread_file:
            return []

        messages: list[dict[str, Any]] = []
        try:
            with open(thread_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("type") != "response_item":
                        continue
                    payload = item.get("payload") or {}
                    if payload.get("type") != "message":
                        continue
                    role = payload.get("role", "")
                    if role not in ("user", "assistant"):
                        continue
                    content = payload.get("content", [])
                    if isinstance(content, list):
                        text = "".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict) and c.get("type") in ("input_text", "output_text", "text")
                        )
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = ""
                    if not text or text.lstrip().startswith("<"):
                        continue
                    ts_str = item.get("timestamp", "")
                    ts = 0
                    if ts_str:
                        try:
                            from datetime import datetime as _dt, timezone as _tz
                            ts = int(_dt.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp())
                        except Exception:
                            ts = 0
                    messages.append({"role": role, "text": text, "ts": ts})
        except OSError:
            pass
        return messages

    # ── Streaming / App Server methods ────────────────────────────────────────

    def _get_or_start_proc(self) -> subprocess.Popen | None:
        global _proc
        with _proc_lock:
            if _proc is None or _proc.poll() is not None:
                _proc = None
                binary = self._resolved_binary()
                if not binary:
                    return None
                _proc = _start_proc(binary)
        return _proc

    def stream_message(
        self,
        thread_id: str | None,
        message: str,
        sse_write: Callable[[dict], None],
        agent_id: str,
        agent_key: str | None = None,
        status_dir: str | None = None,
    ) -> None:
        proc = self._get_or_start_proc()
        if proc is None:
            sse_write({"type": "error", "text": "Failed to start Codex App Server"})
            return

        with _turn_lock:
            server_thread_id = self._do_stream(proc, thread_id, message, sse_write, agent_id)

        if server_thread_id and server_thread_id != thread_id and agent_key and status_dir:
            self._update_thread_id(agent_key, server_thread_id, status_dir)

    def _do_stream(
        self,
        proc: subprocess.Popen,
        thread_id: str | None,
        message: str,
        sse_write: Callable[[dict], None],
        agent_id: str,
    ) -> str | None:
        """Returns server-assigned thread ID (persisted by caller), or None on fatal error."""
        import gateway_presence  # local import — matches server.py pattern

        fd = proc.stdout.fileno()
        buf = bytearray()
        req_id = int(time.time() * 1000) % 100000

        # Phase 1: start or resume thread
        if thread_id:
            _send_rpc(proc, "thread/resume", {"threadId": thread_id}, req_id)
        else:
            _send_rpc(proc, "thread/start", {"threadId": str(uuid.uuid4())}, req_id)

        # Phase 1b: read until server confirms thread ID (thread/started or thread/resumed)
        server_thread_id: str | None = None
        deadline = time.time() + 15
        while time.time() < deadline and server_thread_id is None:
            if proc.poll() is not None:
                sse_write({"type": "error", "text": "Codex process died during thread init"})
                gateway_presence.set_manual_override(agent_id, "idle", "")
                return None
            if select.select([fd], [], [], 0.5)[0]:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if chunk:
                    buf.extend(chunk)
            while b"\n" in buf:
                idx = buf.index(b"\n")
                raw = bytes(buf[:idx]).strip()
                del buf[:idx + 1]
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if ev.get("error"):
                    err = ev["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    sse_write({"type": "error", "text": f"Thread init failed: {msg}"})
                    gateway_presence.set_manual_override(agent_id, "idle", "")
                    return None
                # thread/started or thread/resumed notification
                if ev.get("method") in ("thread/started", "thread/resumed"):
                    t = (ev.get("params") or {}).get("thread") or {}
                    server_thread_id = t.get("id") or thread_id
                    break
                # result response containing thread object
                if ev.get("id") == req_id and isinstance(ev.get("result"), dict):
                    t = ev["result"].get("thread") or {}
                    if t.get("id"):
                        server_thread_id = t["id"]
                        break

        if server_thread_id is None:
            if thread_id:
                server_thread_id = thread_id  # resuming: assume our stored ID is valid
            else:
                sse_write({"type": "error", "text": "Timed out waiting for thread ID from Codex"})
                gateway_presence.set_manual_override(agent_id, "idle", "")
                return None

        # Store at module level so respond_approval (different instance) can read it
        global _active_thread_id
        _active_thread_id = server_thread_id

        req_id += 1

        # Phase 2: start the turn with the server-confirmed thread ID
        _send_rpc(proc, "turn/start", {
            "threadId": server_thread_id,
            "input": [{"type": "text", "text": message}],
        }, req_id)

        # Phase 3: event loop
        last_keepalive = time.time()

        while True:
            if proc.poll() is not None:
                sse_write({"type": "error", "text": "Codex process exited unexpectedly"})
                gateway_presence.set_manual_override(agent_id, "idle", "")
                _reset_proc()
                return server_thread_id

            now = time.time()
            if now - last_keepalive > 20:
                gateway_presence.set_manual_override(agent_id, "working", "Working")
                last_keepalive = now

            ready = select.select([fd], [], [], 1.0)[0]
            if ready:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    sse_write({"type": "error", "text": "Codex read error"})
                    gateway_presence.set_manual_override(agent_id, "idle", "")
                    _reset_proc()
                    return server_thread_id
                if not chunk:
                    sse_write({"type": "error", "text": "Codex process closed stdout"})
                    gateway_presence.set_manual_override(agent_id, "idle", "")
                    _reset_proc()
                    return server_thread_id
                buf.extend(chunk)

            while b"\n" in buf:
                idx = buf.index(b"\n")
                line = bytes(buf[:idx]).strip()
                del buf[:idx + 1]
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    _dbg("RECV", f"bad JSON: {line[:200]}")
                    continue
                _dbg("RECV", f"method={event.get('method')} id={event.get('id')} keys={list(event.keys())}")
                if "error" in event and event.get("id") is not None:
                    err = event["error"]
                    err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    _dbg("RECV", f"RPC error id={event['id']}: {err_msg}")
                    continue
                if self._handle_event(event, sse_write, agent_id):
                    return server_thread_id

    def _handle_event(
        self,
        event: dict,
        sse_write: Callable[[dict], None],
        agent_id: str,
    ) -> bool:
        """Map one App Server event to SSE + presence. Returns True when turn is complete."""
        import gateway_presence

        method = event.get("method", "")
        _dbg("EVENT", f"method={method!r} params_keys={list((event.get('params') or {}).keys())}")
        params = event.get("params") or {}

        if method == "turn/started":
            gateway_presence.set_manual_override(agent_id, "working", "Working")
            sse_write({"type": "start"})

        elif method == "item/started":
            item_type = params.get("type", "")
            if item_type == "agentMessage":
                gateway_presence.set_manual_override(agent_id, "working", "Responding...")
                sse_write({"type": "thinking"})
            elif item_type in ("commandExecution", "tool_call"):
                gateway_presence.set_manual_override(agent_id, "working", "Running command")
                sse_write({"type": "tool", "name": "Running command"})

        elif method == "item/agentMessage/delta":
            delta = params.get("delta", "")
            gateway_presence.set_manual_override(agent_id, "working", "Responding...")
            sse_write({"type": "delta", "text": delta})

        elif method == "item/commandExecution/requestApproval":
            _dbg("APPROVAL", f"requestApproval rpc_id={event.get('id')} params={json.dumps(params)}")
            approval_id = (
                params.get("itemId")
                or params.get("approvalId")
                or str(uuid.uuid4())
            )
            # Store both request params and the JSON-RPC id so we can respond correctly
            _pending_approvals[approval_id] = {"params": params, "rpc_id": event.get("id")}
            reason = params.get("reason") or params.get("description") or "command execution"
            gateway_presence.set_manual_override(agent_id, "working", "Waiting for approval")
            sse_write({"type": "approval", "id": approval_id, "reason": reason})

        elif method == "item/completed":
            sse_write({"type": "item_done"})

        elif method == "turn/completed":
            output = params.get("output") or []
            full_text = ""
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", "")
                if isinstance(content, str):
                    full_text += content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            full_text += part.get("text", "") or part.get("output_text", "")
            gateway_presence.set_manual_override(agent_id, "idle", "")
            sse_write({"type": "done", "text": full_text})
            return True

        return False

    def respond_approval(self, approval_id: str, choice: str) -> dict[str, Any]:
        proc = _get_proc()
        _dbg("APPROVAL", f"approvalId={approval_id!r} choice={choice!r} proc_alive={proc is not None and proc.poll() is None}")
        if proc is None or proc.poll() is not None:
            return {"ok": False, "error": "No active Codex process"}
        try:
            pending = _pending_approvals.pop(approval_id, {})
            rpc_id = pending.get("rpc_id")
            if rpc_id is None:
                return {"ok": False, "error": f"No pending approval for id={approval_id!r}"}
            decision = "accept" if choice == "allow" else "decline"
            msg = (json.dumps({"id": rpc_id, "result": {"decision": decision}}) + "\n").encode("utf-8")
            _dbg("APPROVAL", f"responding to rpc_id={rpc_id} with decision={decision!r}")
            proc.stdin.write(msg)
            proc.stdin.flush()
            return {"ok": True}
        except Exception as exc:
            _dbg("APPROVAL", f"send failed: {exc}")
            return {"ok": False, "error": str(exc)}
