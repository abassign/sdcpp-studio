"""
VRAM Arbiter - Subprocess-level VRAM coordination.

Manages exclusive GPU access between tagging subprocesses and generation.
Uses a JSON file as atomic semaphore with active subprocess tracking.

Key insight: VRAM is consumed by subprocesses (llama-cli, analyze_cpp.py, stable-diffusion.cpp),
not by the HTTP servers. The arbiter must manage subprocess lifecycle, not server ports.
"""
import os
import json
import time
import subprocess
import threading
import requests
from typing import Optional

SEMAPHORE_PATH = "/tmp/vram_semaphore.json"
POLL_INTERVAL = 2  # seconds

# States
STATE_IDLE = "idle"
STATE_TAGGING = "tagging"       # Tagging subprocess holds VRAM
STATE_IA_RUNNING = "ia_running" # IA/generation holds VRAM
STATE_HANDOFF_TAG_TO_IA = "handoff_tag_to_ia"  # Tagging must release VRAM


class VramArbiter:
    """Singleton VRAM arbiter for subprocess-level coordination."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._poller_running = False
        self._poller_thread = None
        self._init_semaphore()

    def _init_semaphore(self):
        if not os.path.exists(SEMAPHORE_PATH):
            self._write_state({
                "state": STATE_IDLE,
                "token_holder": None,
                "ia_wants_vram": False,
                "tag_wants_vram": False,
                "active_subprocess_pid": None,
                "last_updated": time.time()
            })

    def _read_state(self) -> dict:
        try:
            with open(SEMAPHORE_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_state(self, state: dict):
        state["last_updated"] = time.time()
        tmp_path = SEMAPHORE_PATH + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2)
        os.rename(tmp_path, SEMAPHORE_PATH)

    def start_poller(self):
        with self._lock:
            if self._poller_running:
                return
            self._poller_running = True
            self._poller_thread = threading.Thread(target=self._poller_loop, daemon=True)
            self._poller_thread.start()
            print("[VramArbiter] Poller started", flush=True)

    def stop_poller(self):
        self._poller_running = False
        if self._poller_thread:
            self._poller_thread.join(timeout=5)

    def _poller_loop(self):
        while self._poller_running:
            try:
                self._tick()
            except Exception as e:
                print(f"[VramArbiter] Poller error: {e}", flush=True)
            time.sleep(POLL_INTERVAL)

    def _tick(self):
        state = self._read_state()
        if not state:
            return

        current_state = state.get("state", STATE_IDLE)

        # Check if the active subprocess is still alive
        pid = state.get("active_subprocess_pid")
        process_alive = False
        if pid:
            process_alive = self._is_process_alive(pid)

        # If subprocess died unexpectedly, clean up
        if pid and not process_alive:
            print(f"[VramArbiter] Subprocess {pid} died, cleaning up", flush=True)
            state["active_subprocess_pid"] = None
            if current_state != STATE_IDLE:
                state["state"] = STATE_IDLE
                state["token_holder"] = None
                state["ia_wants_vram"] = False
                state["tag_wants_vram"] = False

        # Process state transitions
        if current_state == STATE_TAGGING and not process_alive:
            # Tagging subprocess finished
            state["state"] = STATE_IDLE
            state["token_holder"] = None

        elif current_state == STATE_TAGGING:
            # Tagging is running - check if IA wants VRAM
            if state.get("ia_wants_vram"):
                print(f"[VramArbiter] IA wants VRAM, initiating tag→ia handoff", flush=True)
                state["state"] = STATE_HANDOFF_TAG_TO_IA
                state["token_holder"] = None

        elif current_state == STATE_HANDOFF_TAG_TO_IA:
            # Waiting for tagging subprocess to die
            if not process_alive:
                state["state"] = STATE_IDLE
                state["ia_wants_vram"] = False
                print(f"[VramArbiter] Tagging subprocess died, VRAM free for IA", flush=True)

        elif current_state == STATE_IA_RUNNING and not process_alive:
            # IA/generation finished
            state["state"] = STATE_IDLE
            state["token_holder"] = None

        self._write_state(state)

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def kill_subprocess(self, pid: int, timeout: int = 10):
        """Force kill a subprocess and wait for it to die."""
        try:
            os.kill(pid, 9)  # SIGKILL
            # Wait for process to die
            start = time.time()
            while time.time() - start < timeout:
                if not self._is_process_alive(pid):
                    return True
                time.sleep(0.5)
            return False
        except (OSError, ProcessLookupError):
            return True  # Already dead

    # --- Public API ---

    def request_vram(self, requester: str, subprocess_pid: Optional[int] = None) -> bool:
        """
        Request VRAM access.
        requester: "ia" or "tag"
        subprocess_pid: PID of the subprocess that will use VRAM
        Returns True if granted immediately.
        """
        state = self._read_state()
        current_state = state.get("state", STATE_IDLE)

        if requester == "ia":
            if current_state == STATE_IDLE:
                state["state"] = STATE_IA_RUNNING
                state["token_holder"] = "ia"
                state["ia_wants_vram"] = False
                state["active_subprocess_pid"] = subprocess_pid
                self._write_state(state)
                return True
            else:
                state["ia_wants_vram"] = True
                self._write_state(state)
                return False

        elif requester == "tag":
            if current_state == STATE_IDLE:
                state["state"] = STATE_TAGGING
                state["token_holder"] = "tag"
                state["tag_wants_vram"] = False
                state["active_subprocess_pid"] = subprocess_pid
                self._write_state(state)
                return True
            else:
                state["tag_wants_vram"] = True
                self._write_state(state)
                return False

        return False

    def release_vram(self, requester: str):
        """Release VRAM access."""
        state = self._read_state()
        state["state"] = STATE_IDLE
        state["token_holder"] = None
        state["active_subprocess_pid"] = None
        if requester == "ia":
            state["ia_wants_vram"] = False
        elif requester == "tag":
            state["tag_wants_vram"] = False
        self._write_state(state)

    def wait_for_vram(self, requester: str, subprocess_pid: Optional[int] = None, timeout: int = 120) -> bool:
        """Block until VRAM is granted. Returns True if granted."""
        start = time.time()
        while time.time() - start < timeout:
            if self.request_vram(requester, subprocess_pid):
                return True
            time.sleep(POLL_INTERVAL)
        return False

    def force_kill_and_acquire(self, requester: str, timeout: int = 30) -> bool:
        """
        Force kill the current VRAM holder and acquire the token.
        Used by generation (highest priority).
        """
        state = self._read_state()
        pid = state.get("active_subprocess_pid")

        if pid:
            print(f"[VramArbiter] Force killing subprocess {pid} for {requester}", flush=True)
            self.kill_subprocess(pid, timeout=timeout)

        # Grant token
        state["state"] = STATE_IA_RUNNING if requester == "ia" else STATE_TAGGING
        state["token_holder"] = requester
        state["ia_wants_vram"] = False
        state["tag_wants_vram"] = False
        state["active_subprocess_pid"] = None
        self._write_state(state)
        return True

    def get_status(self) -> dict:
        return self._read_state()


# Module-level singleton
_arbiter = None

def get_arbiter() -> VramArbiter:
    global _arbiter
    if _arbiter is None:
        _arbiter = VramArbiter()
    return _arbiter
