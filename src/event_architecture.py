"""Model-configurable event subscriptions and persistent Agent Zoo forks."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time
from typing import Any
import uuid

from agent_utils.core.events import (
    Event,
    IoCompleted,
    IoFailed,
    SessionRestored,
    TurnSettled,
)
from agent_utils.core.io import FsStat, IoRequest, OpId
from agent_utils.core.pipeline import EmitFn, Feature
from agent_utils.core.state import DurableSpec, State
from agent_utils.core.tool import Tool, tool
from agent_utils.features.buffers.tools import resolve_session_path
from agent_utils.features.core import IoFence
from agent_utils.features.interrupts import InterruptCheck
from agent_zoo.durability import (
    PluginConfigBox,
    PluginConfigStore,
    PluginStorage,
    QueuedWorkerDriver,
    build_checkpoint_document,
    session_context,
    session_cwd,
    session_state,
)
from agent_zoo.events import (
    EmitNotice,
    ForkSessionOp,
    ProjectionRequested,
    PublishView,
    SessionArtifact,
    SessionSource,
    SessionSpec,
    ShutdownRequested,
    SpawnSession,
    SuspendRequested,
)
from agent_zoo.features.settings import Setting, SettingError


DURABLE_STATE_KEY = "event_architecture"
RUNTIME_STATE_KEY = "event_architecture"
FORK_ARTIFACT_NAME = "fork_lifecycle"
FORK_ENABLE_ENV = "AZO_EVENT_ARCHITECTURE_FORK_ENABLED"
CONFIG_PATH_ENV = "AZO_EVENT_ARCHITECTURE_CONFIG"
PLUGIN_NAME = "azo-plugin-event-architecture"
CONFIG_FILENAME = "event_architecture.json"
CONFIG_REL_PATH = f"config/{CONFIG_FILENAME}"
PERSISTENT_FORK_MARKER = '<azo-event-architecture fork-lifecycle="persistent">'
EPHEMERAL_FORK_MARKER = '<azo-event-architecture fork-lifecycle="ephemeral">'
PERSISTENT_FORK_GUIDANCE = (
    "This is a persistent, long-running fork. Remain available for future work and "
    "handle tasks as they arrive; use RLMs for bounded sub-tasks as useful."
)
EPHEMERAL_FORK_GUIDANCE = (
    "This is an ephemeral, single-task fork. Complete the assigned work, use RLMs "
    "for bounded sub-tasks as useful, then call suspend_session() twice as directed."
)
MIN_INTERVAL_SECONDS = 1.0
MAX_CONCURRENT_COMMANDS = 4
DEFAULT_OUTPUT_LIMIT_BYTES = 16_384
VALID_TRIGGERS = {"rising", "change", "each"}
VALID_FORK_LIFECYCLES = {"persistent", "ephemeral"}
SUSPEND_CONFIRMATION_MESSAGE = (
    "You have requested to suspend this session, have you completed all work "
    "originally allocated to this fork? Call suspend_session() again to confirm."
)


# ---------------------------------------------------------------------------
# Durable & Ephemeral Per-Session State
# ---------------------------------------------------------------------------


@dataclass
class EventArchitectureState:
    """Durable per-session state for event subscriptions and fork confirmation."""

    subscriptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    suspend_confirmation_pending: bool = False
    fork_lifecycle: str = ""


@dataclass
class EventArchitectureRuntime:
    """Strictly ephemeral per-session runtime state (never serialized)."""

    fork_ops: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_suspend: bool = False
    driver_generation: int = 0
    last_synced_signature: Any = None
    sync_op_ids: set[str] = field(default_factory=set)


EVENT_ARCHITECTURE_DURABLE_SPEC = DurableSpec(
    name=DURABLE_STATE_KEY,
    cls=EventArchitectureState,
    version=1,
    on_fork="reset",
)


def _event_durable(state: Any) -> EventArchitectureState:
    return state.durable(EVENT_ARCHITECTURE_DURABLE_SPEC)


def _event_runtime(state: Any) -> EventArchitectureRuntime:
    return state.runtime(RUNTIME_STATE_KEY, EventArchitectureRuntime)


def _current_session_id(state: Any) -> str:
    return str(session_context(state).session_id or "").strip()


def _fork_lineage_matches(state: Any, inherited_parent_id: str) -> bool:
    if not inherited_parent_id:
        return True
    if state is None:
        return False
    smeta = session_state(state)
    return (
        str(smeta.kind or "").strip().lower() == "fork"
        and str(smeta.parent_session_id or "").strip() == inherited_parent_id
    )


# ---------------------------------------------------------------------------
# Typed IO Requests, Events & Driver-Private Command Watch Reactor
# ---------------------------------------------------------------------------


class CommandObservation:
    """Immutable-by-convention result passed from watch workers to the pipeline."""

    __slots__ = (
        "subscription_id",
        "revision",
        "session_id",
        "generation",
        "completed_at",
        "stdout",
        "stderr",
        "returncode",
        "error",
    )

    def __init__(
        self,
        subscription_id: str,
        revision: str,
        completed_at: float,
        stdout: str = "",
        stderr: str = "",
        returncode: int | None = None,
        error: str = "",
        session_id: str = "",
        generation: int = 0,
    ) -> None:
        self.subscription_id = subscription_id
        self.revision = revision
        self.session_id = session_id
        self.generation = generation
        self.completed_at = completed_at
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.error = error


@dataclass
class SyncCommandWatches(IoRequest):
    """Typed Sans-I/O request synchronizing background command watches."""

    session_id: str = ""
    subscriptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    cleanup: bool = False


@dataclass
class CommandWatchTriggered(Event):
    """Event pushed by ``CommandWatchDriver`` when a subscribed command completes."""

    observation: CommandObservation
    generation: int = 0


class _BoundedByteTail:
    """Thread-safe byte tail with a strict in-memory size bound."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1024, int(limit))
        self._data = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            self._data.extend(chunk)
            overflow = len(self._data) - self.limit
            if overflow > 0:
                del self._data[:overflow]

    def text(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", errors="replace")


class CommandWatchReactor:
    """Driver-private background scheduler running subscribed shell commands."""

    def __init__(self, *, on_observation: Any = None) -> None:
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._on_observation = on_observation
        self._events: queue.Queue[CommandObservation] = queue.Queue(maxsize=128)
        self._specs: dict[str, dict[str, Any]] = {}
        self._workers: dict[str, dict[str, Any]] = {}
        self._next_due: dict[str, float] = {}
        self._session_id = ""
        self._generation = 0
        self._thread: threading.Thread | None = None

    def set_on_observation(self, callback: Any) -> None:
        with self._lock:
            self._on_observation = callback

    def sync(self, session_id: str, specs: dict[str, dict[str, Any]]) -> int:
        """Replace the reactor snapshot, cancel obsolete command groups, return generation."""
        copied = {
            subscription_id: _reactor_spec(spec)
            for subscription_id, spec in specs.items()
            if bool(spec.get("enabled", True))
        }
        now = time.monotonic()
        processes: list[subprocess.Popen[bytes]] = []
        with self._lock:
            session_changed = session_id != self._session_id
            if session_changed:
                self._session_id = session_id
                self._generation += 1
                self._next_due.clear()
            previous = self._specs
            self._specs = copied
            for subscription_id in list(self._next_due):
                if subscription_id not in copied:
                    self._next_due.pop(subscription_id, None)
            for subscription_id, spec in copied.items():
                old = previous.get(subscription_id)
                if session_changed or old is None or old.get("revision") != spec.get("revision"):
                    self._next_due[subscription_id] = now
                else:
                    self._next_due.setdefault(subscription_id, now)
            for worker in self._workers.values():
                current = copied.get(str(worker["subscription_id"]))
                obsolete = (
                    int(worker["generation"]) != self._generation
                    or current is None
                    or str(current.get("revision") or "") != str(worker["revision"])
                )
                if obsolete:
                    worker["cancelled"] = True
                    process = worker.get("process")
                    if process is not None:
                        processes.append(process)
            if copied:
                self._ensure_thread_locked()
            generation = self._generation
        if session_changed:
            self._discard_events()
        for process in processes:
            _terminate_process_group(process)
        self._wake.set()
        return generation

    def drain(self) -> list[CommandObservation]:
        """Return all buffered observations without blocking."""
        observations: list[CommandObservation] = []
        while True:
            try:
                observations.append(self._events.get_nowait())
            except queue.Empty:
                return observations

    @property
    def current_generation(self) -> int:
        with self._lock:
            return self._generation

    def _discard_events(self) -> None:
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                return

    def cleanup(self) -> None:
        """Stop scheduling and terminate every active command process group."""
        self._stop.set()
        self._wake.set()
        with self._lock:
            processes = []
            self._generation += 1
            self._session_id = ""
            for worker in self._workers.values():
                worker["cancelled"] = True
                process = worker.get("process")
                if process is not None:
                    processes.append(process)
            self._specs = {}
            self._next_due.clear()
        for process in processes:
            _terminate_process_group(process)
        scheduler_thread = self._thread
        if scheduler_thread is not None and scheduler_thread is not threading.current_thread():
            scheduler_thread.join()
        with self._lock:
            self._thread = None
            worker_threads = [
                worker.get("thread")
                for worker in self._workers.values()
                if worker.get("thread") is not None
            ]
        for worker_thread in worker_threads:
            if worker_thread is not threading.current_thread():
                worker_thread.join()
        self._discard_events()

    def _ensure_thread_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="azo-command-watch-scheduler",
            daemon=True,
        )
        self._thread.start()

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            launch: list[tuple[str, dict[str, Any], int]] = []
            wait_seconds = 1.0
            now = time.monotonic()
            with self._lock:
                due_times: list[float] = []
                active_subscription_ids = {
                    str(worker["subscription_id"])
                    for worker in self._workers.values()
                }
                available = max(0, MAX_CONCURRENT_COMMANDS - len(self._workers))
                for subscription_id, spec in self._specs.items():
                    due = self._next_due.get(subscription_id, now)
                    if subscription_id not in active_subscription_ids and due <= now and available > 0:
                        token = uuid.uuid4().hex
                        self._workers[token] = {
                            "subscription_id": subscription_id,
                            "revision": str(spec.get("revision") or ""),
                            "generation": self._generation,
                            "process": None,
                            "thread": None,
                            "cancelled": False,
                        }
                        launch.append((token, dict(spec), self._generation))
                        available -= 1
                    elif subscription_id not in active_subscription_ids:
                        due_times.append(due)
                if due_times:
                    wait_seconds = max(0.05, min(1.0, min(due_times) - now))

            for token, spec, generation in launch:
                worker_thread = threading.Thread(
                    target=self._run_subscription,
                    args=(token, spec, generation),
                    name=f"azo-command-watch-{token[:8]}",
                    daemon=True,
                )
                with self._lock:
                    worker = self._workers.get(token)
                    if worker is None:
                        continue
                    worker["thread"] = worker_thread
                worker_thread.start()

            self._wake.wait(wait_seconds)
            self._wake.clear()

    def _run_subscription(self, token: str, spec: dict[str, Any], generation: int) -> None:
        subscription_id = str(self._worker_value(token, "subscription_id") or "")
        session_id = str(self._session_id_for_generation(generation) or "")
        observation = _execute_command(
            subscription_id,
            spec,
            session_id=session_id,
            generation=generation,
            on_process=lambda process: self._register_process(token, process),
            is_cancelled=lambda: self._worker_cancelled(token),
        )
        emit_callback = None
        should_deliver = False
        with self._lock:
            worker = self._workers.pop(token, None)
            current = self._specs.get(subscription_id)
            current_generation = self._generation
            if (
                worker is not None
                and not bool(worker.get("cancelled", False))
                and generation == current_generation
                and current is not None
                and str(current.get("revision") or "") == str(spec.get("revision") or "")
            ):
                interval = float(current.get("interval_seconds", MIN_INTERVAL_SECONDS))
                self._next_due[subscription_id] = time.monotonic() + interval
                self._put_observation(observation)
                emit_callback = self._on_observation
                should_deliver = True
        if should_deliver and callable(emit_callback):
            try:
                emit_callback(observation)
            except Exception:
                pass
        self._wake.set()

    def _worker_value(self, token: str, key: str) -> Any:
        with self._lock:
            worker = self._workers.get(token)
            return None if worker is None else worker.get(key)

    def _worker_cancelled(self, token: str) -> bool:
        with self._lock:
            worker = self._workers.get(token)
            return worker is None or bool(worker.get("cancelled", False)) or self._stop.is_set()

    def _register_process(self, token: str, process: subprocess.Popen[bytes]) -> None:
        terminate = False
        with self._lock:
            worker = self._workers.get(token)
            if worker is None or bool(worker.get("cancelled", False)) or self._stop.is_set():
                terminate = True
            else:
                worker["process"] = process
        if terminate:
            _terminate_process_group(process)

    def _session_id_for_generation(self, generation: int) -> str:
        with self._lock:
            return self._session_id if generation == self._generation else ""

    def _put_observation(self, observation: CommandObservation) -> None:
        try:
            self._events.put_nowait(observation)
            return
        except queue.Full:
            pass
        try:
            self._events.get_nowait()
        except queue.Empty:
            pass
        try:
            self._events.put_nowait(observation)
        except queue.Full:
            pass


class CommandWatchDriver(QueuedWorkerDriver):
    """Process-bound driver owning ``CommandWatchReactor`` and pushing watch events."""

    request_types = (SyncCommandWatches,)
    durability = "process_bound"
    worker_name = "azo-command-watch-driver"
    cancel_queued_on_close = True

    def __init__(self) -> None:
        super().__init__()
        self._reactor = CommandWatchReactor(on_observation=self._on_observation)

    def _on_observation(self, observation: CommandObservation) -> None:
        with self._cond:
            if self._closing or self._closed:
                return
        self._safe_emit(
            CommandWatchTriggered(
                observation=observation,
                generation=int(observation.generation),
            )
        )

    def is_fenced(self, op_id: OpId) -> bool:
        with self._cond:
            return self._closed or self._closing or op_id in self._cancelled

    def quiesce(
        self,
        *,
        policy: str = "terminate",
        deadline: float | None = None,
    ) -> list[Event]:
        self.cancel_pending()
        events = super().quiesce(policy=policy, deadline=deadline)
        self._reactor.cleanup()
        return events

    def on_close(self) -> None:
        self._reactor.cleanup()

    def execute(self, req: IoRequest) -> dict[str, Any]:
        if not isinstance(req, SyncCommandWatches):
            raise TypeError(
                f"CommandWatchDriver expected SyncCommandWatches, got {type(req).__name__}"
            )
        if req.cleanup:
            self._reactor.cleanup()
            return {
                "generation": self._reactor.current_generation,
                "session_id": "",
                "cleaned": True,
            }
        generation = self._reactor.sync(req.session_id, req.subscriptions)
        return {
            "generation": generation,
            "session_id": req.session_id,
            "cleaned": False,
        }


def _reactor_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Copy only fields required by background command execution."""
    return {
        "revision": str(spec.get("revision") or ""),
        "command": str(spec.get("command") or ""),
        "cwd": str(spec.get("cwd") or ""),
        "interval_seconds": float(spec.get("interval_seconds", MIN_INTERVAL_SECONDS)),
        "timeout_seconds": float(spec.get("timeout_seconds", 10.0)),
        "output_limit_bytes": int(spec.get("output_limit_bytes", DEFAULT_OUTPUT_LIMIT_BYTES)),
        "enabled": bool(spec.get("enabled", True)),
    }


def _execute_command(
    subscription_id: str,
    spec: dict[str, Any],
    *,
    session_id: str = "",
    generation: int = 0,
    on_process=None,
    is_cancelled=None,
) -> CommandObservation:
    """Execute one shell command with bounded pipe capture and group cleanup."""
    command = str(spec.get("command") or "")
    cwd = str(spec.get("cwd") or "") or None
    timeout_seconds = max(0.1, float(spec.get("timeout_seconds", 10.0)))
    output_limit = max(1024, int(spec.get("output_limit_bytes", DEFAULT_OUTPUT_LIMIT_BYTES)))
    completed_at = time.time()
    process: subprocess.Popen[bytes] | None = None
    stdout_tail = _BoundedByteTail(output_limit)
    stderr_tail = _BoundedByteTail(output_limit)
    readers: list[threading.Thread] = []

    try:
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("command watch was cancelled before launch")
        process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        if on_process is not None:
            on_process(process)
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("failed to open command output pipes")
        readers = [
            threading.Thread(
                target=_read_pipe,
                args=(process.stdout, stdout_tail),
                name=f"azo-command-stdout-{subscription_id[:8]}",
                daemon=True,
            ),
            threading.Thread(
                target=_read_pipe,
                args=(process.stderr, stderr_tail),
                name=f"azo-command-stderr-{subscription_id[:8]}",
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
        try:
            returncode = process.wait(timeout=timeout_seconds)
            error = ""
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            returncode = process.returncode
            error = f"command timed out after {timeout_seconds:g}s"
        finally:
            _terminate_process_group(process)
            for reader in readers:
                reader.join(timeout=1.0)

        return CommandObservation(
            subscription_id=subscription_id,
            revision=str(spec.get("revision") or ""),
            session_id=session_id,
            generation=generation,
            completed_at=time.time(),
            stdout=stdout_tail.text(),
            stderr=stderr_tail.text(),
            returncode=returncode,
            error=error,
        )
    except Exception as exc:
        if process is not None:
            _terminate_process_group(process)
        for reader in readers:
            reader.join(timeout=1.0)
        return CommandObservation(
            subscription_id=subscription_id,
            revision=str(spec.get("revision") or ""),
            session_id=session_id,
            generation=generation,
            completed_at=completed_at,
            stdout=stdout_tail.text(),
            stderr=stderr_tail.text(),
            returncode=process.returncode if process is not None else None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _read_pipe(pipe, tail: _BoundedByteTail) -> None:
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                return
            tail.append(chunk)
    except (OSError, ValueError):
        return
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate the shell's process group, including surviving descendants."""
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    process.poll()
    deadline = time.monotonic() + 0.5
    while _process_group_exists(process_group_id) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.02)
    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            pass


# ---------------------------------------------------------------------------
# Pure Sans-I/O Subscription Helpers & InterruptCheck
# ---------------------------------------------------------------------------


def _subscription_store(state: Any) -> dict[str, dict[str, Any]]:
    return _event_durable(state).subscriptions


def _owned_subscriptions(state: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    """Return this session's subscriptions without mutating cloned records."""
    session_id = _current_session_id(state)
    if not session_id:
        return "", {}
    return session_id, {
        subscription_id: spec
        for subscription_id, spec in _subscription_store(state).items()
        if str(spec.get("owner_session_id") or "") == session_id
    }


def _sync_signature(session_id: str, subscriptions: dict[str, dict[str, Any]]) -> Any:
    return (
        session_id,
        tuple(
            sorted(
                (sub_id, str(spec.get("revision") or ""), bool(spec.get("enabled", True)))
                for sub_id, spec in subscriptions.items()
            )
        ),
    )


def _submit_watch_sync(
    state: Any,
    *,
    force: bool = False,
    cleanup: bool = False,
    hold_readiness: str | bool | None = None,
) -> OpId | None:
    io_table = getattr(state, "io", None)
    if io_table is None:
        return None
    rt = _event_runtime(state)
    if cleanup:
        rt.last_synced_signature = None
        op_id = io_table.submit(
            SyncCommandWatches(session_id="", subscriptions={}, cleanup=True),
            tag="command_watch_sync",
        )
        rt.sync_op_ids.add(op_id)
        return op_id

    session_id, owned = _owned_subscriptions(state)
    sig = _sync_signature(session_id, owned)
    if not force and sig == rt.last_synced_signature:
        return None
    rt.last_synced_signature = sig
    specs = {
        subscription_id: _reactor_spec(spec)
        for subscription_id, spec in owned.items()
        if bool(spec.get("enabled", True))
    }
    op_id = io_table.submit(
        SyncCommandWatches(session_id=session_id, subscriptions=specs, cleanup=False),
        tag="command_watch_sync",
        hold_readiness=hold_readiness,
    )
    rt.sync_op_ids.add(op_id)
    return op_id


def _find_subscription(
    subscriptions: dict[str, dict[str, Any]],
    query: str,
) -> tuple[str, dict[str, Any]] | None:
    if query in subscriptions:
        return query, subscriptions[query]
    matches = [
        (subscription_id, spec)
        for subscription_id, spec in subscriptions.items()
        if str(spec.get("name") or "") == query
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _validate_subscription_inputs(
    *,
    name: str,
    command: str,
    match_regex: str,
    trigger: str,
    interval_seconds: float,
    timeout_seconds: float,
) -> tuple[str, str, str, str, float, float]:
    clean_name = str(name or "").strip()
    clean_command = str(command or "").strip()
    clean_regex = str(match_regex or "")
    clean_trigger = str(trigger or "rising").strip().lower()
    if not clean_name:
        raise ValueError("name must not be empty")
    if not clean_command:
        raise ValueError("command must not be empty")
    if clean_trigger not in VALID_TRIGGERS:
        raise ValueError(f"trigger must be one of {sorted(VALID_TRIGGERS)}")
    try:
        re.compile(clean_regex)
    except re.error as exc:
        raise ValueError(f"invalid match_regex: {exc}") from exc
    interval = max(MIN_INTERVAL_SECONDS, float(interval_seconds))
    timeout = max(0.1, float(timeout_seconds))
    return clean_name, clean_command, clean_regex, clean_trigger, interval, timeout


def _observation_text(spec: dict[str, Any], observation: CommandObservation) -> str:
    source = str(spec.get("match_stream") or "stdout")
    if source == "stderr":
        return observation.stderr
    if source == "both":
        return (
            observation.stdout
            + ("\n" if observation.stdout and observation.stderr else "")
            + observation.stderr
        )
    return observation.stdout


def _evaluate_observation(
    spec: dict[str, Any],
    observation: CommandObservation,
) -> str | None:
    runtime = spec.setdefault("runtime", {})
    runtime["last_run_at"] = observation.completed_at
    runtime["last_returncode"] = observation.returncode
    runtime["runs"] = int(runtime.get("runs", 0)) + 1

    if observation.error:
        previous_error = str(runtime.get("last_error") or "")
        runtime["last_error"] = observation.error
        if bool(spec.get("wake_on_error", False)) and observation.error != previous_error:
            runtime["fires"] = int(runtime.get("fires", 0)) + 1
            return _format_error_interrupt(spec, observation)
        return None

    runtime["last_error"] = ""
    text = _observation_text(spec, observation)
    pattern = re.compile(str(spec.get("match_regex") or ""), re.MULTILINE)
    match = pattern.search(text)
    matched = match is not None
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    initialized = bool(runtime.get("initialized", False))
    previous_matched = bool(runtime.get("last_matched", False))
    previous_digest = str(runtime.get("last_output_digest") or "")
    trigger = str(spec.get("trigger") or "rising")

    should_fire = False
    if initialized or bool(spec.get("fire_initial", False)):
        if trigger == "rising":
            should_fire = matched and not previous_matched
        elif trigger == "change":
            should_fire = matched and digest != previous_digest
        elif trigger == "each":
            should_fire = matched

    runtime["initialized"] = True
    runtime["last_matched"] = matched
    runtime["last_output_digest"] = digest
    runtime["last_stdout"] = observation.stdout[-1000:]
    runtime["last_stderr"] = observation.stderr[-1000:]

    if not should_fire:
        return None

    runtime["fires"] = int(runtime.get("fires", 0)) + 1
    matched_text = match.group(0) if match is not None else ""
    return _format_match_interrupt(spec, observation, matched_text)


def _format_match_interrupt(
    spec: dict[str, Any],
    observation: CommandObservation,
    matched_text: str,
) -> str:
    event_id = f"evt_{uuid.uuid4().hex[:16]}"
    lines = [
        "--- EXTERNAL COMMAND EVENT ---",
        f"event_id: {event_id}",
        f"subscription: {spec.get('name')} ({spec.get('id')})",
        f"owner_session_id: {spec.get('owner_session_id')}",
        f"trigger: {spec.get('trigger')}",
        f"returncode: {observation.returncode}",
        f"command: {spec.get('command')}",
        f"matched: {matched_text[:1000]}",
    ]
    if observation.stdout:
        lines.extend(["stdout:", observation.stdout[-4000:]])
    if observation.stderr:
        lines.extend(["stderr:", observation.stderr[-2000:]])
    lines.append("--- END EXTERNAL COMMAND EVENT ---")
    return "\n".join(lines)


def _format_error_interrupt(
    spec: dict[str, Any],
    observation: CommandObservation,
) -> str:
    return "\n".join(
        [
            "--- COMMAND WATCH ERROR ---",
            f"subscription: {spec.get('name')} ({spec.get('id')})",
            f"owner_session_id: {spec.get('owner_session_id')}",
            f"command: {spec.get('command')}",
            f"error: {observation.error}",
            "--- END COMMAND WATCH ERROR ---",
        ]
    )


class CommandWatchCheck(InterruptCheck):
    """Pure Sans-I/O watch coordinator synchronizing via ``SyncCommandWatches``."""

    check_name = "command_watch_subscriptions"
    cooldown = 0
    handles = (
        SessionRestored,
        CommandWatchTriggered,
        TurnSettled,
        ShutdownRequested,
        IoCompleted,
        IoFailed,
    )
    optional_reads = set()
    writes = set()

    def __init__(self, *, inherited_parent_id: str = "") -> None:
        super().__init__()
        self._inherited_parent_id = inherited_parent_id

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        del emit
        if isinstance(event, ShutdownRequested):
            _submit_watch_sync(state, cleanup=True)
            return

        if isinstance(event, SessionRestored):
            rt = _event_runtime(state)
            rt.fork_ops.clear()
            rt.pending_suspend = False
            rt.sync_op_ids.clear()
            rt.last_synced_signature = None
            if not _fork_lineage_matches(state, self._inherited_parent_id):
                return
            _submit_watch_sync(state, force=True, hold_readiness="event_architecture")
            return

        if isinstance(event, (IoCompleted, IoFailed)):
            rt = _event_runtime(state)
            if event.op_id not in rt.sync_op_ids:
                return
            rt.sync_op_ids.discard(event.op_id)
            if isinstance(event, IoCompleted) and isinstance(event.value, dict):
                gen = int(event.value.get("generation") or 0)
                if gen > rt.driver_generation:
                    rt.driver_generation = gen
            return

        if not _fork_lineage_matches(state, self._inherited_parent_id):
            return

        if isinstance(event, TurnSettled):
            _submit_watch_sync(state, force=False)
            return

        if isinstance(event, CommandWatchTriggered):
            observation = event.observation
            rt = _event_runtime(state)
            obs_gen = int(event.generation or observation.generation or 0)
            if rt.driver_generation and obs_gen and obs_gen < rt.driver_generation:
                return
            session_id, subscriptions = _owned_subscriptions(state)
            if observation.session_id and observation.session_id != session_id:
                return
            spec = subscriptions.get(observation.subscription_id)
            if spec is None:
                return
            if str(spec.get("revision") or "") != observation.revision:
                return
            message = _evaluate_observation(spec, observation)
            if message:
                self._fire(state, message)

    def status_line(self, state: Any) -> str:
        session_id, subscriptions = _owned_subscriptions(state)
        enabled = sum(bool(spec.get("enabled", True)) for spec in subscriptions.values())
        return f"  command watches: {enabled} enabled for session {session_id or '(unbound)'}"


# ---------------------------------------------------------------------------
# Fork Lifecycle & Suspension
# ---------------------------------------------------------------------------


def _fork_launch_message(text: str, lifecycle: str) -> str:
    if lifecycle == "ephemeral":
        marker = EPHEMERAL_FORK_MARKER
        guidance = EPHEMERAL_FORK_GUIDANCE
    else:
        marker = PERSISTENT_FORK_MARKER
        guidance = PERSISTENT_FORK_GUIDANCE
    return f"{marker}\n{guidance}\n\n{text}"


def _content_texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        ]
    return []


def _fork_lifecycle_from_session_inputs(state: Any) -> str:
    smeta = session_state(state)
    if str(smeta.kind or "").strip().lower() != "fork":
        return ""
    durable_lifecycle = str(_event_durable(state).fork_lifecycle or "").strip().lower()
    if durable_lifecycle in VALID_FORK_LIFECYCLES:
        return durable_lifecycle
    parent_session_id = str(smeta.parent_session_id or "").strip()
    raw_inputs = getattr(state, "_azo_session_inputs", None)
    if not isinstance(raw_inputs, (list, tuple)):
        return ""
    lifecycle = ""
    for item in raw_inputs:
        if isinstance(item, SessionArtifact):
            name, data, meta = item.name, item.data, item.metadata
        elif isinstance(item, dict):
            name = str(item.get("name") or "")
            data = item.get("data")
            meta = item.get("metadata")
        else:
            continue
        if name != FORK_ARTIFACT_NAME:
            continue
        payload = data if isinstance(data, dict) else (meta if isinstance(meta, dict) else {})
        artifact_parent = str(payload.get("parent_session_id") or "").strip()
        if artifact_parent and artifact_parent != parent_session_id:
            continue
        candidate = str(payload.get("lifecycle") or "").strip().lower()
        if candidate in VALID_FORK_LIFECYCLES:
            lifecycle = candidate
    return lifecycle


def _latest_fork_lifecycle(state: Any) -> str:
    lifecycle = ""
    for entry in getattr(state, "entries", []) or []:
        for message in getattr(entry, "messages", []) or []:
            if str(message.get("role") or "") != "user":
                continue
            for text in _content_texts(message.get("content")):
                persistent_at = text.rfind(PERSISTENT_FORK_MARKER)
                ephemeral_at = text.rfind(EPHEMERAL_FORK_MARKER)
                if persistent_at >= 0 or ephemeral_at >= 0:
                    lifecycle = (
                        "ephemeral"
                        if ephemeral_at > persistent_at
                        else "persistent"
                    )
    if lifecycle:
        return lifecycle
    return _fork_lifecycle_from_session_inputs(state)


def _ephemeral_fork_authorized(state: Any) -> bool:
    return _latest_fork_lifecycle(state) == "ephemeral"


def _fork_ops(state: Any) -> dict[str, dict[str, Any]]:
    return _event_runtime(state).fork_ops


def _build_fork_lifecycle_artifact(
    lifecycle: str,
    *,
    parent_session_id: str,
) -> SessionArtifact:
    return SessionArtifact(
        name=FORK_ARTIFACT_NAME,
        kind="json",
        data={
            "lifecycle": lifecycle,
            "parent_session_id": parent_session_id,
            "plugin": PLUGIN_NAME,
        },
        content_type="application/json",
        metadata={
            "plugin": PLUGIN_NAME,
            "lifecycle": lifecycle,
            "parent_session_id": parent_session_id,
        },
    )


def _build_fork_session_spec(
    state: Any,
    *,
    config: dict[str, Any] | None,
    child_session_id: str,
    parent_session_id: str,
    first_user_message: str,
    title: str = "",
    description: str = "",
) -> SessionSpec:
    cfg = config if isinstance(config, dict) else getattr(state, "config", None)
    cfg = cfg if isinstance(cfg, dict) else {}
    ctx = session_context(state, config=cfg)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    pilot_cfg = cfg.get("pilot") if isinstance(cfg.get("pilot"), dict) else {}

    project = (
        str(
            ctx.project_name
            or cfg.get("project_name")
            or "default"
        ).strip()
        or "default"
    )
    pipeline_key = str(ctx.pipeline_key or "default").strip() or "default"
    return SessionSpec(
        project=project,
        workdir=session_cwd(state),
        model=str(llm_cfg.get("default") or getattr(state, "active_model", "") or "").strip(),
        pipeline_key=pipeline_key,
        env_name=str(pilot_cfg.get("_env_name") or ""),
        config_path=str(ctx.config_path or ""),
        config_sets=tuple(ctx.config_sets or ()),
        state_root=str(ctx.state_root or ""),
        store_root=str(ctx.store_root or ""),
        local_state_root=str(ctx.local_state_root or ""),
        kind="fork",
        title=str(title or "").strip() or f"fork {child_session_id[:8]}",
        description=str(description or "").strip()
        or (f"Fork of {parent_session_id[:8]}" if parent_session_id else ""),
        parent_session_id=parent_session_id,
        first_user_message=first_user_message,
        spawn_policy="fork",
        startup_mode="normal",
        lifetime="independent",
        extra_env=((FORK_ENABLE_ENV, parent_session_id),) if parent_session_id else (),
    )


def _extract_fork_checkpoint(
    value: Any,
    *,
    fallback_title: str = "",
) -> tuple[str, str, str | None, str, str, str]:
    receipt = value.get("receipt") if isinstance(value, dict) else value
    if receipt is not None and getattr(receipt, "local_saved", True) is False:
        err = str(getattr(receipt, "error", "") or "fork checkpoint was not saved locally")
        return "", "", None, fallback_title, "", err

    def _pick(field_name: str, fallback: str = "") -> str:
        raw = value.get(field_name) if isinstance(value, dict) else getattr(value, field_name, None)
        if not raw and receipt is not None and receipt is not value:
            raw = getattr(receipt, field_name, None)
        return str(raw or fallback or "").strip()

    child_id = _pick("session_id")
    rev_id = _pick("revision_id")
    inst_id = _pick("instance_id") or None
    title = _pick("title", fallback_title)
    description = _pick("description")
    if not child_id:
        return "", rev_id, inst_id, title, description, "fork checkpoint did not return a child session ID"
    return child_id, rev_id, inst_id, title, description, ""


class ForkSessionCoordinator:
    """Coordinate ``ForkSessionOp(spawn_child=False)`` -> ``SpawnSession`` -> ``PublishView(attach=...)``."""

    handles = (IoCompleted, IoFailed, SessionRestored)
    optional_reads = {
        "config",
        "active_model",
        "cwd",
        "session_cwd",
        "terminal_cwd",
    }
    writes = set()

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        inherited_parent_id: str = "",
    ) -> None:
        self.config = config if isinstance(config, dict) else None
        self._inherited_parent_id = inherited_parent_id

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        if isinstance(event, SessionRestored):
            _fork_ops(state).clear()
            return
        if not isinstance(event, (IoCompleted, IoFailed)):
            return
        if not _fork_lineage_matches(state, self._inherited_parent_id):
            return
        ops = _fork_ops(state)
        if event.op_id not in ops:
            return
        pending = ops.pop(event.op_id)
        stage = str(pending.get("stage") or "")
        if stage == "fork_checkpoint":
            self._handle_fork_checkpoint(event, pending, ops, state, emit)
        elif stage == "spawn_session":
            self._handle_spawn_session(event, pending, state, emit)

    def _handle_fork_checkpoint(
        self,
        event: IoCompleted | IoFailed,
        pending: dict[str, Any],
        ops: dict[str, dict[str, Any]],
        state: State,
        emit: EmitFn,
    ) -> None:
        if isinstance(event, IoFailed):
            emit(EmitNotice(f"Failed to fork agent session: {getattr(event, 'error', '')}", level="error"))
            emit(ProjectionRequested())
            return

        child_id, revision_id, checkpoint_instance_id, title, description, error = (
            _extract_fork_checkpoint(event.value, fallback_title=str(pending.get("title") or ""))
        )
        if error:
            emit(EmitNotice(f"Failed to fork agent session: {error}", level="error"))
            emit(ProjectionRequested())
            return

        parent_session_id = str(pending.get("parent_session_id") or "")
        lifecycle = str(pending.get("lifecycle") or "persistent")
        spec = _build_fork_session_spec(
            state,
            config=self.config,
            child_session_id=child_id,
            parent_session_id=parent_session_id,
            first_user_message=str(pending.get("first_user_message") or ""),
            title=title,
            description=description,
        )
        spawn_op_id = state.io.submit(
            SpawnSession(
                session_id=child_id,
                source=SessionSource(
                    kind="resume",
                    session_id=child_id,
                    revision_id=revision_id,
                    instance_id=checkpoint_instance_id,
                ),
                spec=spec,
                wait_ready=False,
                watch=False,
                launch_metadata={
                    "plugin": PLUGIN_NAME,
                    "fork_lifecycle": lifecycle,
                    "parent_session_id": parent_session_id,
                },
                inputs=(_build_fork_lifecycle_artifact(lifecycle, parent_session_id=parent_session_id),),
            ),
            tag="event_architecture_spawn",
        )
        ops[spawn_op_id] = {
            "stage": "spawn_session",
            "child_session_id": child_id,
            "parent_session_id": parent_session_id,
            "lifecycle": lifecycle,
            "kind": "fork",
        }

    def _handle_spawn_session(
        self,
        event: IoCompleted | IoFailed,
        pending: dict[str, Any],
        state: State,
        emit: EmitFn,
    ) -> None:
        if isinstance(event, IoFailed):
            emit(
                EmitNotice(
                    f"Failed to spawn forked agent session: {getattr(event, 'error', '')}",
                    level="error",
                )
            )
            emit(ProjectionRequested())
            return

        val = event.value
        get = val.get if isinstance(val, dict) else lambda key, default="": getattr(val, key, default)
        child_id = str(get("session_id", "") or pending.get("child_session_id") or "").strip()
        logs_dir = str(get("logs_dir", "") or get("logs", "") or "").strip()
        stderr_path = str(get("stderr_path", "") or "").strip()
        if not logs_dir and stderr_path:
            logs_dir = os.path.dirname(stderr_path)

        if child_id:
            state.io.submit(
                PublishView(attach={"session_id": child_id, "kind": str(pending.get("kind") or "fork")}),
                tag="frontend",
            )

        lifecycle = str(pending.get("lifecycle") or "persistent")
        short_id = child_id[:8] if child_id else ""
        msg = f"Spawned {lifecycle} fork session · {short_id}" if short_id else f"Spawned {lifecycle} fork session."
        if logs_dir:
            msg += f" · logs: {logs_dir}"
        emit(EmitNotice(msg, level="info"))
        emit(ProjectionRequested())


@tool()
def fork_agent(message: str, lifecycle: str = "persistent", state=None) -> str:
    """Create an independent worker agent that can respond to future events.

    Forks cannot be messaged by the calling agent and do not return task results
    or output buffers to it. Use this tool to construct autonomous, event-driven
    workflows. Use ``submit_rlm`` instead for delegated work that should return a
    result or output buffer.

    Args:
        message: Initial instructions supplied to the independent worker.
        lifecycle: ``persistent`` creates a long-running worker that remains available for future events; ``ephemeral`` creates an independent worker for a bounded task that may suspend itself when finished.
    """
    text = str(message or "").strip()
    if not text:
        raise ValueError("message must not be empty")
    normalized_lifecycle = str(lifecycle or "").strip().lower()
    if normalized_lifecycle not in VALID_FORK_LIFECYCLES:
        raise ValueError("lifecycle must be 'persistent' or 'ephemeral'")
    if state is None or getattr(state, "io", None) is None:
        raise RuntimeError("fork_agent requires a live Agent Zoo session")
    session_id = _current_session_id(state)
    if not session_id:
        raise RuntimeError("fork_agent requires a bound live session ID")

    launch_message = _fork_launch_message(text, normalized_lifecycle)
    title = str(session_state(state).title or "")
    source_doc = build_checkpoint_document(state)

    op_id = state.io.submit(
        ForkSessionOp(
            target_session_id="",
            source_document=source_doc,
            first_user_message=launch_message,
            title=title,
            spawn_policy="fork",
            spawn_child=False,
        ),
        tag="event_architecture_fork",
    )
    _fork_ops(state)[op_id] = {
        "stage": "fork_checkpoint",
        "parent_session_id": session_id,
        "lifecycle": normalized_lifecycle,
        "first_user_message": launch_message,
        "title": title,
    }
    return (
        "Fork requested through the live session control path. "
        f"Lifecycle: {normalized_lifecycle}. Parent session: {session_id}. "
        "The new agent will receive the supplied message."
    )


class SuspendSession(Tool):
    """Ephemeral-fork-only graceful suspension tool."""

    name = "suspend_session"
    optional_reads = {"entries", "last_tool_calls"}
    writes = set()

    def available(self, state: State) -> bool:
        return _ephemeral_fork_authorized(state)

    def call(self, state: State) -> str:
        """Suspend this ephemeral fork after a required two-call confirmation."""
        durable = _event_durable(state)
        previous_calls = list(getattr(state, "last_tool_calls", []) or [])
        previous_was_confirmation = (
            len(previous_calls) == 1
            and previous_calls[0].name == "suspend_session"
            and previous_calls[0].result == SUSPEND_CONFIRMATION_MESSAGE
            and not bool(previous_calls[0].error)
        )
        current_calls = []
        entries = list(getattr(state, "entries", []) or [])
        if entries:
            current_calls = list(getattr(entries[-1], "tool_calls", []) or [])
        if not current_calls:
            current_calls = list(getattr(state, "pending_tool_calls", []) or [])
        current_is_only_suspend = (
            len(current_calls) == 1 and current_calls[0].name == "suspend_session"
        )
        if (
            not durable.suspend_confirmation_pending
            or not previous_was_confirmation
            or not current_is_only_suspend
        ):
            durable.suspend_confirmation_pending = True
            return SUSPEND_CONFIRMATION_MESSAGE

        durable.suspend_confirmation_pending = False
        if state is None or getattr(state, "io", None) is None:
            raise RuntimeError("suspend_session requires a live Agent Zoo session")
        _event_runtime(state).pending_suspend = True
        return "Ephemeral fork suspension confirmed; saving and suspending this session."


class EphemeralSuspendHook:
    """Emit ``SuspendRequested`` on ``TurnSettled`` after ``suspend_session`` consolidates."""

    _singleton = True
    handles = (TurnSettled,)
    writes = {
        "done",
        "shutdown",
        "shutdown_reason",
        "shutdown_save",
        "shutdown_registry_status",
        "shutdown_spawned_policy",
    }

    def __init__(self, *, inherited_parent_id: str = "") -> None:
        self._inherited_parent_id = inherited_parent_id

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        if not isinstance(event, TurnSettled):
            return
        if not _fork_lineage_matches(state, self._inherited_parent_id):
            return
        rt = _event_runtime(state)
        if not rt.pending_suspend:
            return
        rt.pending_suspend = False
        state.done = True
        state.shutdown = True
        state.shutdown_reason = "suspend"
        state.shutdown_save = True
        state.shutdown_registry_status = "stopped"
        state.shutdown_spawned_policy = "leave"
        emit(
            SuspendRequested(
                reason="ephemeral fork completed",
                save=True,
                spawned_policy="leave",
                registry_status="stopped",
            )
        )


suspend_session = SuspendSession()


# ---------------------------------------------------------------------------
# Pure Sans-I/O Watch Subscription Tools
# ---------------------------------------------------------------------------


@tool(optional_reads={"cwd", "session_cwd", "terminal_cwd", "home_dir"})
def subscribe_interrupt(
    name: str,
    command: str,
    match_regex: str,
    interval_seconds: float = 30.0,
    trigger: str = "rising",
    timeout_seconds: float = 10.0,
    match_stream: str = "stdout",
    fire_initial: bool = False,
    wake_on_error: bool = False,
    cwd: str = "",
    state=None,
) -> Iterator[FsStat]:
    """Poll a shell command and wake this session when its output triggers.

    Commands run asynchronously under ``/bin/bash -lc``. Polling for a
    subscription is serialized: the next interval begins after the prior command
    finishes. Subscriptions are strictly owned by the creating session ID;
    forked sessions must create their own subscriptions.

    Args:
        name: Human-readable unique subscription name for this session.
        command: Shell command to execute on every poll.
        match_regex: Python regular expression applied to selected command output.
        interval_seconds: Seconds to wait after each completed poll; minimum 1.
        trigger: ``rising`` for false-to-true, ``change`` for changed matching
            output, or ``each`` for every matching poll.
        timeout_seconds: Maximum runtime for each command invocation.
        match_stream: Output to match: ``stdout``, ``stderr``, or ``both``.
        fire_initial: Whether an already-matching first observation should wake.
        wake_on_error: Wake once when the command runner reports a new error.
        cwd: Working directory. Empty uses the session's current working directory.
    """
    (
        clean_name,
        clean_command,
        clean_regex,
        clean_trigger,
        interval,
        timeout,
    ) = _validate_subscription_inputs(
        name=name,
        command=command,
        match_regex=match_regex,
        trigger=trigger,
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    stream = str(match_stream or "stdout").strip().lower()
    if stream not in {"stdout", "stderr", "both"}:
        raise ValueError("match_stream must be stdout, stderr, or both")

    session_id, subscriptions = _owned_subscriptions(state)
    if not session_id:
        raise RuntimeError("subscribe_interrupt requires a live session ID")
    if any(str(spec.get("name") or "") == clean_name for spec in subscriptions.values()):
        raise ValueError(f"subscription name already exists: {clean_name}")

    raw_cwd = str(cwd or "").strip()
    resolved_cwd = resolve_session_path(raw_cwd, state) if raw_cwd else session_cwd(state)
    if not resolved_cwd:
        raise ValueError("cwd is required when session has no working directory")

    stat_result = yield FsStat(resolved_cwd)
    if not getattr(stat_result, "exists", False) or not getattr(stat_result, "is_dir", False):
        raise ValueError(f"cwd is not a directory: {resolved_cwd}")
    canonical_cwd = str(getattr(stat_result, "path", "") or resolved_cwd)

    subscription_id = f"watch_{uuid.uuid4().hex[:12]}"
    spec = {
        "id": subscription_id,
        "name": clean_name,
        "owner_session_id": session_id,
        "command": clean_command,
        "match_regex": clean_regex,
        "match_stream": stream,
        "interval_seconds": interval,
        "timeout_seconds": timeout,
        "output_limit_bytes": DEFAULT_OUTPUT_LIMIT_BYTES,
        "trigger": clean_trigger,
        "fire_initial": bool(fire_initial),
        "wake_on_error": bool(wake_on_error),
        "enabled": True,
        "cwd": canonical_cwd,
        "created_at": time.time(),
        "revision": uuid.uuid4().hex,
        "runtime": {},
    }
    _subscription_store(state)[subscription_id] = spec
    _submit_watch_sync(state, force=True)
    return "Registered command interrupt subscription:\n" + json.dumps(
        {key: value for key, value in spec.items() if key != "runtime"},
        indent=2,
        sort_keys=True,
    )


@tool()
def list_interrupt_subscriptions(state=None) -> str:
    """List command interrupt subscriptions owned by this session."""
    session_id, subscriptions = _owned_subscriptions(state)
    if not subscriptions:
        return f"No command interrupt subscriptions for session {session_id or '(unbound)'}."
    rows = []
    for subscription_id, spec in sorted(
        subscriptions.items(), key=lambda item: str(item[1].get("name") or "")
    ):
        runtime = dict(spec.get("runtime") or {})
        rows.append(
            {
                "id": subscription_id,
                "name": spec.get("name"),
                "owner_session_id": spec.get("owner_session_id"),
                "command": spec.get("command"),
                "match_regex": spec.get("match_regex"),
                "trigger": spec.get("trigger"),
                "interval_seconds": spec.get("interval_seconds"),
                "runtime": runtime,
            }
        )
    return json.dumps(rows, indent=2, sort_keys=True)


@tool()
def remove_interrupt_subscription(subscription: str, state=None) -> str:
    """Remove one command interrupt subscription by ID or exact name.

    Args:
        subscription: Subscription ID or exact session-local subscription name.
    """
    query = str(subscription or "").strip()
    if not query:
        raise ValueError("subscription must not be empty")
    session_id, subscriptions = _owned_subscriptions(state)
    found = _find_subscription(subscriptions, query)
    if found is None:
        raise ValueError(f"subscription not found for session {session_id}: {query}")
    subscription_id, spec = found
    _subscription_store(state).pop(subscription_id, None)
    _submit_watch_sync(state, force=True)
    return f"Removed subscription {spec.get('name')} ({subscription_id}) from session {session_id}."


# ---------------------------------------------------------------------------
# Host Build-Time Config Loading & Pure Sans-I/O Settings via PluginConfigBox
# ---------------------------------------------------------------------------

_CONFIG_LOAD_ERROR = "Cannot read plugin configuration"


def _plugin_storage(*, state_root: str | Path | None = None) -> PluginStorage:
    return PluginStorage(
        PLUGIN_NAME,
        state_root=state_root,
        config_filename=CONFIG_REL_PATH,
    )


def _plugin_config_store(*, state_root: str | Path | None = None) -> PluginConfigStore:
    override = str(os.environ.get(CONFIG_PATH_ENV, "") or "").strip()
    if override:
        return PluginConfigStore(Path(override).expanduser())
    return _plugin_storage(state_root=state_root).config


def _configured_enablement(config: Any) -> bool | None:
    section = (config or {}).get("event_architecture")
    if not isinstance(section, dict) or "enabled" not in section:
        return None
    value = section.get("enabled")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return value is True


def _validate_config_doc(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("Plugin configuration must be an object")
    if "event_architecture" in document and not isinstance(document["event_architecture"], dict):
        raise ValueError("Plugin configuration section must be an object")
    return document


def _load_config_box(*, state_root: str | Path | None = None) -> PluginConfigBox:
    store = _plugin_config_store(state_root=state_root)
    box = PluginConfigBox(
        PLUGIN_NAME,
        config_filename=CONFIG_REL_PATH,
        state_root=state_root,
        validate=_validate_config_doc,
        store=store,
    )
    if box.load_error is None and box.path.is_file():
        try:
            if not box.path.read_text(encoding="utf-8").strip():
                box.load_error = _CONFIG_LOAD_ERROR
        except OSError:
            box.load_error = _CONFIG_LOAD_ERROR
    elif box.load_error is not None:
        box.load_error = _CONFIG_LOAD_ERROR
    return box


def _registration_mode(
    config: Any,
    *,
    config_box: PluginConfigBox | None = None,
) -> tuple[bool, str]:
    """Return ``(should_register, inherited_parent_id)`` without inspecting session state."""
    explicit = _configured_enablement(config)
    if explicit is not None:
        return explicit, ""
    box = config_box if config_box is not None else _load_config_box()
    if box.load_error is None:
        saved = _configured_enablement(box.data)
        if saved is not None:
            return saved, ""
    inherited_parent_id = str(os.environ.get(FORK_ENABLE_ENV, "") or "").strip()
    if inherited_parent_id:
        return True, inherited_parent_id
    return False, ""


def _plugin_enabled(
    config: Any,
    session: Any = None,
    *,
    config_box: PluginConfigBox | None = None,
) -> bool:
    should_register, inherited_parent_id = _registration_mode(config, config_box=config_box)
    if not should_register:
        return False
    if not inherited_parent_id or session is None:
        return True
    return _fork_lineage_matches(getattr(session, "state", None), inherited_parent_id)


class _GatedTool(Tool):
    """Wrap a ``Tool`` with a per-state fork-lineage gate for dynamic ``/load`` support."""

    def __init__(self, inner: Tool, *, inherited_parent_id: str) -> None:
        self._inner = inner
        self._inherited_parent_id = inherited_parent_id
        self.name = inner.name
        self.description = inner.description
        self.schema = inner.schema
        self.io_tag = inner.io_tag
        self.reads = set(inner.reads)
        self.writes = set(inner.writes)
        self.optional_reads = set(inner.optional_reads)
        self.init = dict(inner.init)
        if hasattr(inner, "fn"):
            self.fn = inner.fn
            self._fn_wants_state = getattr(inner, "_fn_wants_state", False)

    def available(self, state: State) -> bool:
        return (
            _fork_lineage_matches(state, self._inherited_parent_id)
            and self._inner.available(state)
        )

    def call(self, state: State, **kwargs: Any) -> Any:
        return self._inner.call(state, **kwargs)


def _register_settings(builder: Any, box: PluginConfigBox) -> None:
    def get_enabled(_ctx: Any) -> bool:
        doc = box.require_loaded()
        return _configured_enablement(doc) is True

    def set_enabled(ctx: Any, value: Any) -> None:
        box.require_loaded()
        if type(value) is not bool:
            raise SettingError("Expected a boolean")

        def _updater(current: dict[str, Any]) -> dict[str, Any]:
            doc = dict(current) if isinstance(current, dict) else {}
            section = doc.get("event_architecture")
            if section is not None and not isinstance(section, dict):
                raise SettingError("Plugin configuration section must be an object")
            updated_section = dict(section) if isinstance(section, dict) else {}
            updated_section["enabled"] = value
            doc["event_architecture"] = updated_section
            return doc

        box.mutate(ctx, _updater, tag="event_architecture.config")

    builder.add(
        Feature(
            "event_architecture_settings",
            components=[
                Setting(
                    "event_architecture.enabled",
                    "Enabled by default",
                    "bool",
                    get_enabled,
                    set=set_enabled,
                    group="Event architecture",
                    apply="reload",
                    help=(
                        "Saved default for ordinary sessions, not current fork status. "
                        "Absent means false here; legacy fork inheritance remains until explicitly saved. "
                        "Reload to apply; explicit launch configuration may override this default."
                    ),
                ),
            ],
            order=[Setting],
        )
    )


def register_features(builder, *, config):
    """Register model tools when enabled or eligible for fork inheritance (no session capture)."""
    box = _load_config_box()
    _register_settings(builder, box)
    should_register, inherited_parent_id = _registration_mode(config, config_box=box)
    if not should_register:
        return

    base_tools: list[Tool] = [
        fork_agent,
        suspend_session,
        subscribe_interrupt,
        list_interrupt_subscriptions,
        remove_interrupt_subscription,
    ]
    tools: list[Tool] = (
        [_GatedTool(t, inherited_parent_id=inherited_parent_id) for t in base_tools]
        if inherited_parent_id
        else base_tools
    )
    components = [
        ForkSessionCoordinator(config, inherited_parent_id=inherited_parent_id),
        *tools[:2],
        EphemeralSuspendHook(inherited_parent_id=inherited_parent_id),
        *tools[2:],
        CommandWatchCheck(inherited_parent_id=inherited_parent_id),
    ]
    builder.add(
        Feature(
            name="event_architecture",
            components=components,
            order=[IoFence, ForkSessionCoordinator, Tool, {EphemeralSuspendHook, CommandWatchCheck}],
            drivers=[CommandWatchDriver],
            durable_state=EVENT_ARCHITECTURE_DURABLE_SPEC,
        )
    )
