"""Model-configurable event subscriptions and persistent Agent Zoo forks."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from agent_utils import Feature, Tool, tool
from agent_utils.components import InterruptCheck
from agent_utils.session_cwd import get_session_cwd


STATE_ATTR = "azo_event_subscriptions"
REACTOR_STATE_ATTR = "azo_event_reactor_runtime"
SUSPEND_CONFIRM_ATTR = "azo_event_suspend_confirmation_pending"
FORK_ENABLE_ENV = "AZO_EVENT_ARCHITECTURE_FORK_ENABLED"
CONFIG_PATH_ENV = "AZO_EVENT_ARCHITECTURE_CONFIG"
PLUGIN_NAME = "azo-plugin-event-architecture"
CONFIG_FILENAME = "event_architecture.json"
PERSISTENT_FORK_MARKER = "<azo-event-architecture fork-lifecycle=\"persistent\">"
EPHEMERAL_FORK_MARKER = "<azo-event-architecture fork-lifecycle=\"ephemeral\">"
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


class CommandObservation:
    """Immutable-by-convention result passed from workers to the backend thread."""

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
    """Run subscribed commands outside the agent backend thread."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._events: queue.Queue[CommandObservation] = queue.Queue(maxsize=128)
        self._specs: dict[str, dict[str, Any]] = {}
        self._workers: dict[str, dict[str, Any]] = {}
        self._next_due: dict[str, float] = {}
        self._session_id = ""
        self._generation = 0
        self._thread: threading.Thread | None = None

    def sync(self, session_id: str, specs: dict[str, dict[str, Any]]) -> None:
        """Replace the reactor snapshot and cancel obsolete command groups."""
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
        if session_changed:
            self._discard_events()
        for process in processes:
            _terminate_process_group(process)
        self._wake.set()

    def drain(self) -> list[CommandObservation]:
        """Return all completed observations without blocking."""
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
            worker_threads = [
                worker.get("thread")
                for worker in self._workers.values()
                if worker.get("thread") is not None
            ]
        for worker_thread in worker_threads:
            if worker_thread is not threading.current_thread():
                worker_thread.join()
        self._discard_events()

    def save_state(self) -> dict[str, Any]:
        """Persist no scheduler, thread, queue, or subprocess state."""
        return {}

    def restore_state(self, _data: dict[str, Any]) -> None:
        """Discard live work before subscriptions are synchronized after restore."""
        self.cleanup()

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


def _reactor_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Copy only fields required by background command execution."""
    return {
        "revision": str(spec.get("revision") or ""),
        "command": str(spec.get("command") or ""),
        "cwd": str(spec.get("cwd") or ""),
        "interval_seconds": float(spec.get("interval_seconds", MIN_INTERVAL_SECONDS)),
        "timeout_seconds": float(spec.get("timeout_seconds", 10.0)),
        "output_limit_bytes": int(spec.get("output_limit_bytes", DEFAULT_OUTPUT_LIMIT_BYTES)),
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


_REACTOR = CommandWatchReactor()


def _current_session_id(state: Any) -> str:
    run_db = getattr(state, "run_db", None)
    session_id = str(getattr(run_db, "session_id", "") or "").strip()
    if session_id:
        return session_id
    session_id = str(getattr(state, "_session_id", "") or "").strip()
    if session_id:
        return session_id
    session_io = getattr(state, "session_io", None)
    return str(getattr(session_io, "session_id", "") or "").strip()


def _subscription_store(state: Any) -> dict[str, dict[str, Any]]:
    store = getattr(state, STATE_ATTR, None)
    if not isinstance(store, dict):
        store = {}
        setattr(state, STATE_ATTR, store)
    return store


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
        return observation.stdout + ("\n" if observation.stdout and observation.stderr else "") + observation.stderr
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
    """Synchronize subscriptions and deliver matching command observations."""

    check_name = "command_watch_subscriptions"
    cooldown = 0
    optional_reads = {STATE_ATTR, REACTOR_STATE_ATTR, "run_db", "_session_id", "session_io"}
    writes = {STATE_ATTR, REACTOR_STATE_ATTR}

    def __init__(self, reactor: CommandWatchReactor):
        self.reactor = reactor
        self.init = {STATE_ATTR: dict, REACTOR_STATE_ATTR: lambda: reactor}

    def __call__(self, state):
        setattr(state, REACTOR_STATE_ATTR, self.reactor)
        session_id, subscriptions = _owned_subscriptions(state)
        self.reactor.sync(session_id, subscriptions)
        current_generation = int(getattr(self.reactor, "current_generation", 0))

        for observation in self.reactor.drain():
            if current_generation and observation.generation != current_generation:
                continue
            if observation.session_id and observation.session_id != session_id:
                continue
            spec = subscriptions.get(observation.subscription_id)
            if spec is None:
                continue
            if str(spec.get("revision") or "") != observation.revision:
                continue
            message = _evaluate_observation(spec, observation)
            if message:
                self._fire(state, message)
        return state

    def status_line(self, state) -> str:
        session_id, subscriptions = _owned_subscriptions(state)
        enabled = sum(bool(spec.get("enabled", True)) for spec in subscriptions.values())
        return f"  command watches: {enabled} enabled for session {session_id or '(unbound)'}"


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
    return lifecycle


def _ephemeral_fork_authorized(state: Any) -> bool:
    return _latest_fork_lifecycle(state) == "ephemeral"


@tool(optional_reads={"session_io", "run_db", "_session_id"})
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
    session_io = getattr(state, "session_io", None)
    if session_io is None or not callable(getattr(session_io, "send", None)):
        raise RuntimeError("fork_agent requires a live Agent Zoo session")
    session_id = _current_session_id(state)
    if not session_id:
        raise RuntimeError("fork_agent requires a bound live session ID")

    session_io.send(
        "slash_command",
        {"raw": f"/fork {_fork_launch_message(text, normalized_lifecycle)}"},
        client_id="azo-plugin-event-architecture",
    )
    return (
        "Fork requested through the live session control path. "
        f"Lifecycle: {normalized_lifecycle}. Parent session: {session_id}. "
        "The new agent will receive the supplied message."
    )


class SuspendSession(Tool):
    """Ephemeral-fork-only graceful suspension tool."""

    optional_reads = {"entries", "last_tool_calls", "session_io", SUSPEND_CONFIRM_ATTR}
    writes = {
        SUSPEND_CONFIRM_ATTR,
        "done",
        "shutdown_reason",
        "shutdown_save",
        "shutdown_registry_status",
        "shutdown_spawned_policy",
        "shutdown",
    }
    init = {SUSPEND_CONFIRM_ATTR: bool}

    @staticmethod
    def fn(state=None) -> str:
        """Suspend this ephemeral fork after a required two-call confirmation."""
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
        confirmation_pending = bool(getattr(state, SUSPEND_CONFIRM_ATTR, False))
        if (
            not confirmation_pending
            or not previous_was_confirmation
            or not current_is_only_suspend
        ):
            setattr(state, SUSPEND_CONFIRM_ATTR, True)
            return SUSPEND_CONFIRMATION_MESSAGE

        setattr(state, SUSPEND_CONFIRM_ATTR, False)
        session_io = getattr(state, "session_io", None)
        if session_io is None or not callable(getattr(session_io, "send", None)):
            raise RuntimeError("suspend_session requires a live Agent Zoo session")
        session_io.send(
            "control",
            {"type": "suspend", "reason": "ephemeral fork completed"},
            client_id="azo-plugin-event-architecture",
        )
        state.done = True
        state.shutdown_reason = "suspend"
        state.shutdown_save = True
        state.shutdown_registry_status = "stopped"
        state.shutdown_spawned_policy = "leave"
        state.shutdown = True
        return "Ephemeral fork suspension confirmed; saving and suspending this session."

    def available(self, state) -> bool:
        return _ephemeral_fork_authorized(state)


suspend_session = SuspendSession()


@tool(optional_reads={STATE_ATTR, "run_db", "_session_id", "session_io"}, writes={STATE_ATTR})
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
) -> str:
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

    base_cwd = Path(get_session_cwd(state)).expanduser()
    resolved_cwd = Path(cwd).expanduser() if str(cwd or "").strip() else base_cwd
    if not resolved_cwd.is_absolute():
        resolved_cwd = base_cwd / resolved_cwd
    resolved_cwd = resolved_cwd.resolve()
    if not resolved_cwd.is_dir():
        raise ValueError(f"cwd is not a directory: {resolved_cwd}")

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
        "cwd": str(resolved_cwd),
        "created_at": time.time(),
        "revision": uuid.uuid4().hex,
        "runtime": {},
    }
    _subscription_store(state)[subscription_id] = spec
    _REACTOR.sync(session_id, _owned_subscriptions(state)[1])
    return "Registered command interrupt subscription:\n" + json.dumps(
        {key: value for key, value in spec.items() if key != "runtime"},
        indent=2,
        sort_keys=True,
    )


@tool(optional_reads={STATE_ATTR, "run_db", "_session_id", "session_io"})
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


@tool(optional_reads={STATE_ATTR, "run_db", "_session_id", "session_io"}, writes={STATE_ATTR})
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
    _REACTOR.sync(session_id, _owned_subscriptions(state)[1])
    return f"Removed subscription {spec.get('name')} ({subscription_id}) from session {session_id}."


def _agent_zoo_state_root() -> Path:
    for env_name in ("AGENT_ZOO_HOME", "AGENT_ZOO_STATE_ROOT", "AGENT_ZOO_INSTALL_ROOT"):
        value = str(os.environ.get(env_name, "") or "").strip()
        if value:
            return Path(value).expanduser()
    xdg_data = str(os.environ.get("XDG_DATA_HOME", "") or "").strip()
    base = Path(xdg_data).expanduser() if xdg_data else Path.home() / ".local" / "share"
    return base / "agent-zoo"


def _plugin_config_path() -> Path:
    override = str(os.environ.get(CONFIG_PATH_ENV, "") or "").strip()
    if override:
        return Path(override).expanduser()
    return (
        _agent_zoo_state_root()
        / "plugin-configs"
        / PLUGIN_NAME
        / "config"
        / CONFIG_FILENAME
    )


def _load_installed_config() -> dict[str, Any]:
    path = _plugin_config_path()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


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


def _effective_configured_enablement(config: Any) -> bool | None:
    explicit = _configured_enablement(config)
    if explicit is not None:
        return explicit
    return _configured_enablement(_load_installed_config())


def _registration_session_id(session: Any) -> str:
    state = getattr(session, "state", None)
    return _current_session_id(state) if state is not None else ""


def _plugin_enabled(config: Any, session: Any) -> bool:
    configured = _effective_configured_enablement(config)
    if configured is not None:
        return configured
    state = getattr(session, "state", None)
    session_kind = str(getattr(state, "_session_kind", "") or "").strip().lower()
    parent_session_id = str(getattr(state, "_parent_session_id", "") or "").strip()
    inherited_parent_id = str(os.environ.get(FORK_ENABLE_ENV, "") or "").strip()
    return (
        session_kind == "fork"
        and bool(parent_session_id)
        and inherited_parent_id == parent_session_id
    )


def register_features(builder, *, session, config):
    """Register model tools only when explicitly enabled or inherited by a fork."""
    configured = _effective_configured_enablement(config)
    enabled = _plugin_enabled(config, session)
    current_session_id = _registration_session_id(session)
    if enabled and current_session_id:
        os.environ[FORK_ENABLE_ENV] = current_session_id
    elif configured is not None or not enabled:
        os.environ.pop(FORK_ENABLE_ENV, None)
    if not enabled:
        _REACTOR.cleanup()
        return

    components = [
        fork_agent,
        suspend_session,
        subscribe_interrupt,
        list_interrupt_subscriptions,
        remove_interrupt_subscription,
        CommandWatchCheck(_REACTOR),
    ]
    builder.add(
        Feature(
            name="event_architecture",
            components=components,
        )
    )
