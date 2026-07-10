from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utils import Entry, State, ToolCall


PLUGIN_PATH = Path(__file__).parents[1] / "src" / "event_architecture.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("event_architecture_test", PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "AZO_EVENT_ARCHITECTURE_CONFIG",
        str(tmp_path / "missing-event-architecture.json"),
    )
    return load_plugin()


class FakeSessionIO:
    def __init__(self, session_id: str = "session-a") -> None:
        self.session_id = session_id
        self.sent = []

    def send(self, channel, payload, *, client_id=""):
        self.sent.append((channel, payload, client_id))
        return SimpleNamespace(channel=channel, payload=payload, client_id=client_id)


def make_state(session_id: str = "session-a") -> State:
    state = State()
    state._session_id = session_id
    state.run_db = SimpleNamespace(session_id=session_id)
    state.session_io = FakeSessionIO(session_id)
    state.pending_tool_calls = []
    state.pending_interrupts = []
    state._interrupt_state = {}
    state.tool_schemas = {}
    state.terminal_cwd = str(Path.cwd())
    return state


def authorize_ephemeral_fork(plugin, state: State) -> None:
    state._session_kind = "fork"
    state.entries.append(
        Entry(
            messages=[
                {
                    "role": "user",
                    "content": f"{plugin.EPHEMERAL_FORK_MARKER}\nassigned work",
                }
            ],
            index=len(state.entries),
            step=state.step,
        )
    )


def call_tool(tool, state: State, **kwargs):
    tc = ToolCall(
        id="call-1",
        name=tool.name,
        raw_args="{}",
        parsed_args=kwargs,
    )
    state.pending_tool_calls = [tc]
    tool(state)
    return tc


def test_fork_agent_routes_existing_slash_command(plugin):
    state = make_state("parent-session")

    tc = call_tool(plugin.fork_agent, state, message="Monitor issues independently.")

    assert tc.error is False
    assert "Fork requested" in tc.result
    assert len(state.session_io.sent) == 1
    channel, payload, client_id = state.session_io.sent[0]
    assert channel == "slash_command"
    assert client_id == "azo-plugin-event-architecture"
    assert payload["raw"].startswith(f"/fork {plugin.PERSISTENT_FORK_MARKER}\n")
    assert payload["raw"].endswith("\n\nMonitor issues independently.")


def test_fork_agent_defaults_to_persistent_lifecycle(plugin):
    state = make_state("parent-session")

    tc = call_tool(plugin.fork_agent, state, message="Remain available for work.")

    assert tc.error is False
    raw = state.session_io.sent[0][1]["raw"]
    assert raw.startswith(f"/fork {plugin.PERSISTENT_FORK_MARKER}\n")
    assert plugin.PERSISTENT_FORK_GUIDANCE in raw
    assert plugin.EPHEMERAL_FORK_MARKER not in raw
    assert raw.endswith("\n\nRemain available for work.")


def test_fork_agent_propagates_ephemeral_lifecycle(plugin):
    state = make_state("parent-session")

    tc = call_tool(
        plugin.fork_agent,
        state,
        message="Handle one issue and stop.",
        lifecycle="ephemeral",
    )

    assert tc.error is False
    assert "Lifecycle: ephemeral" in (tc.result or "")
    raw = state.session_io.sent[0][1]["raw"]
    assert raw.startswith(f"/fork {plugin.EPHEMERAL_FORK_MARKER}\n")
    assert plugin.EPHEMERAL_FORK_GUIDANCE in raw
    assert raw.endswith("\n\nHandle one issue and stop.")


def test_fork_agent_rejects_unknown_lifecycle(plugin):
    state = make_state("parent-session")

    tc = call_tool(
        plugin.fork_agent,
        state,
        message="Do work.",
        lifecycle="temporary",
    )

    assert tc.error is True
    assert "persistent" in (tc.result or "")
    assert "ephemeral" in (tc.result or "")
    assert state.session_io.sent == []




def test_latest_fork_lifecycle_overrides_cloned_ancestor(plugin):
    state = make_state("persistent-descendant")
    authorize_ephemeral_fork(plugin, state)
    state.entries.append(
        Entry(
            messages=[
                {
                    "role": "user",
                    "content": f"{plugin.PERSISTENT_FORK_MARKER}\nnew role",
                }
            ],
            index=len(state.entries),
            step=state.step,
        )
    )

    assert plugin._latest_fork_lifecycle(state) == "persistent"
    assert plugin.suspend_session.available(state) is False


def test_suspend_session_is_unavailable_to_persistent_fork(plugin):
    state = make_state("persistent-child")
    state._session_kind = "fork"
    state.last_tool_calls = []

    tc = call_tool(plugin.suspend_session, state)

    assert tc.error is True
    assert "not currently available" in (tc.result or "")
    assert not hasattr(state, plugin.SUSPEND_CONFIRM_ATTR)
    assert state.done is False


def test_suspend_session_requires_adjacent_confirmation(plugin):
    state = make_state("ephemeral-child")
    authorize_ephemeral_fork(plugin, state)
    state.last_tool_calls = []

    first = call_tool(plugin.suspend_session, state)

    assert first.error is False
    assert first.result == plugin.SUSPEND_CONFIRMATION_MESSAGE
    assert state.azo_event_suspend_confirmation_pending is True
    assert state.done is False

    state.last_tool_calls = [first]
    second = call_tool(plugin.suspend_session, state)

    assert second.error is False
    assert "suspension confirmed" in (second.result or "")
    assert state.azo_event_suspend_confirmation_pending is False
    assert state.done is True
    assert state.shutdown is True
    assert state.shutdown_reason == "suspend"
    assert state.shutdown_save is True
    assert state.shutdown_registry_status == "stopped"
    assert state.shutdown_spawned_policy == "leave"


def test_suspend_session_confirmation_resets_after_other_tool(plugin):
    state = make_state("ephemeral-child")
    authorize_ephemeral_fork(plugin, state)
    state.last_tool_calls = []
    first = call_tool(plugin.suspend_session, state)
    state.last_tool_calls = [
        ToolCall(
            id="other-call",
            name="list_tools",
            raw_args="{}",
            parsed_args={},
            result="ok",
        )
    ]

    retry = call_tool(plugin.suspend_session, state)

    assert first.result == plugin.SUSPEND_CONFIRMATION_MESSAGE
    assert retry.result == plugin.SUSPEND_CONFIRMATION_MESSAGE
    assert state.azo_event_suspend_confirmation_pending is True
    assert state.done is False


def test_suspend_session_rejects_confirmation_bundled_with_other_tool(plugin):
    state = make_state("ephemeral-child")
    authorize_ephemeral_fork(plugin, state)
    state.last_tool_calls = []
    first = call_tool(plugin.suspend_session, state)
    state.last_tool_calls = [first]
    state.entries = [
        *state.entries,
        SimpleNamespace(
            tool_calls=[
                ToolCall(
                    id="confirm-call",
                    name="suspend_session",
                    raw_args="{}",
                    parsed_args={},
                ),
                ToolCall(
                    id="other-call",
                    name="list_tools",
                    raw_args="{}",
                    parsed_args={},
                ),
            ]
        )
    ]

    retry = call_tool(plugin.suspend_session, state)

    assert retry.result == plugin.SUSPEND_CONFIRMATION_MESSAGE
    assert state.azo_event_suspend_confirmation_pending is True
    assert state.done is False


def test_owned_subscriptions_are_session_local_and_non_destructive(plugin):
    state = make_state("child-session")
    state._session_id = "stale-parent-session"
    state.session_io.session_id = "stale-parent-session"
    state.azo_event_subscriptions = {
        "parent": {
            "id": "parent",
            "name": "parent-only",
            "owner_session_id": "parent-session",
            "revision": "old-a",
            "runtime": {"runs": 3},
        },
        "child": {
            "id": "child",
            "name": "child-only",
            "owner_session_id": "child-session",
            "revision": "child-a",
            "runtime": {},
        },
    }

    session_id, owned = plugin._owned_subscriptions(state)

    assert session_id == "child-session"
    assert set(owned) == {"child"}
    assert set(state.azo_event_subscriptions) == {"parent", "child"}
    assert state.azo_event_subscriptions["parent"]["runtime"] == {"runs": 3}


def test_change_trigger_establishes_baseline_then_fires(plugin):
    spec = {
        "id": "watch-1",
        "name": "pr-list",
        "owner_session_id": "session-a",
        "command": "gh pr list",
        "match_regex": ".+",
        "match_stream": "stdout",
        "trigger": "change",
        "fire_initial": False,
        "runtime": {},
    }

    first = plugin.CommandObservation("watch-1", "rev", 1.0, stdout="pr-1\n", returncode=0)
    same = plugin.CommandObservation("watch-1", "rev", 2.0, stdout="pr-1\n", returncode=0)
    changed = plugin.CommandObservation("watch-1", "rev", 3.0, stdout="pr-1\npr-2\n", returncode=0)

    assert plugin._evaluate_observation(spec, first) is None
    assert plugin._evaluate_observation(spec, same) is None
    message = plugin._evaluate_observation(spec, changed)
    assert message is not None
    assert "EXTERNAL COMMAND EVENT" in message
    assert "pr-2" in message
    assert spec["runtime"]["fires"] == 1


def test_rising_trigger_rearms_after_nonmatch(plugin):
    spec = {
        "id": "watch-1",
        "name": "ready",
        "owner_session_id": "session-a",
        "command": "check",
        "match_regex": "READY",
        "match_stream": "stdout",
        "trigger": "rising",
        "fire_initial": True,
        "runtime": {},
    }

    matched = lambda when: plugin.CommandObservation(
        "watch-1", "rev", when, stdout="READY", returncode=0
    )
    clear = plugin.CommandObservation("watch-1", "rev", 2.0, stdout="waiting", returncode=0)

    assert plugin._evaluate_observation(spec, matched(1.0)) is not None
    assert plugin._evaluate_observation(spec, matched(1.5)) is None
    assert plugin._evaluate_observation(spec, clear) is None
    assert plugin._evaluate_observation(spec, matched(3.0)) is not None


def test_subscribe_list_and_remove_tool_lifecycle(plugin, monkeypatch, tmp_path):
    state = make_state("owner-session")
    sync_calls = []
    monkeypatch.setattr(plugin._REACTOR, "sync", lambda session_id, specs: sync_calls.append((session_id, dict(specs))))

    created = call_tool(
        plugin.subscribe_interrupt,
        state,
        name="new-prs",
        command="gh pr list --state open",
        match_regex=".+",
        interval_seconds=0.1,
        trigger="change",
        timeout_seconds=5,
        match_stream="both",
        fire_initial=False,
        wake_on_error=True,
        cwd=str(tmp_path),
    )

    assert created.error is False
    subscriptions = state.azo_event_subscriptions
    assert len(subscriptions) == 1
    subscription_id, spec = next(iter(subscriptions.items()))
    assert subscription_id.startswith("watch_")
    assert spec["owner_session_id"] == "owner-session"
    assert spec["interval_seconds"] == plugin.MIN_INTERVAL_SECONDS
    assert spec["cwd"] == str(tmp_path.resolve())
    assert sync_calls[-1][0] == "owner-session"

    listed = call_tool(plugin.list_interrupt_subscriptions, state)
    assert listed.error is False
    assert "new-prs" in listed.result
    assert "owner-session" in listed.result

    removed = call_tool(plugin.remove_interrupt_subscription, state, subscription="new-prs")
    assert removed.error is False
    assert subscription_id not in state.azo_event_subscriptions


def test_subscribe_rejects_invalid_regex(plugin, monkeypatch):
    state = make_state()
    monkeypatch.setattr(plugin._REACTOR, "sync", lambda *_args, **_kwargs: None)

    tc = call_tool(
        plugin.subscribe_interrupt,
        state,
        name="broken",
        command="true",
        match_regex="[",
    )

    assert tc.error is True
    assert "invalid match_regex" in tc.result


class FakeReactor:
    def __init__(self, observations, generation=3):
        self.observations = list(observations)
        self.synced = []
        self.current_generation = generation

    def sync(self, session_id, specs):
        self.synced.append((session_id, dict(specs)))

    def drain(self):
        observations = self.observations
        self.observations = []
        return observations


def test_interrupt_check_discards_stale_revision_and_fires_current(plugin):
    state = make_state("session-a")
    state.azo_event_subscriptions = {
        "watch-1": {
            "id": "watch-1",
            "name": "ready",
            "owner_session_id": "session-a",
            "command": "printf READY",
            "match_regex": "READY",
            "match_stream": "stdout",
            "trigger": "each",
            "fire_initial": True,
            "enabled": True,
            "revision": "current",
            "runtime": {},
        }
    }
    reactor = FakeReactor(
        [
            plugin.CommandObservation(
                "watch-1", "current", 1.0, stdout="READY", returncode=0,
                session_id="session-a", generation=2,
            ),
            plugin.CommandObservation(
                "watch-1", "current", 2.0, stdout="READY", returncode=0,
                session_id="other-session", generation=3,
            ),
            plugin.CommandObservation(
                "watch-1", "stale", 3.0, stdout="READY", returncode=0,
                session_id="session-a", generation=3,
            ),
            plugin.CommandObservation(
                "watch-1", "current", 4.0, stdout="READY", returncode=0,
                session_id="session-a", generation=3,
            ),
        ]
    )

    plugin.CommandWatchCheck(reactor)(state)

    assert reactor.synced[-1][0] == "session-a"
    assert len(state.pending_interrupts) == 1
    assert "READY" in state.pending_interrupts[0]


def test_execute_command_captures_bounded_tail(plugin, tmp_path):
    observation = plugin._execute_command(
        "watch-1",
        {
            "revision": "rev",
            "command": "python -c 'print(\"x\" * 5000)'",
            "cwd": str(tmp_path),
            "timeout_seconds": 5,
            "output_limit_bytes": 1024,
        },
    )

    assert observation.error == ""
    assert observation.returncode == 0
    assert 1000 <= len(observation.stdout) <= 1024
    assert set(observation.stdout.strip()) == {"x"}


def test_execute_command_timeout_kills_process_group(plugin, tmp_path):
    observation = plugin._execute_command(
        "watch-1",
        {
            "revision": "rev",
            "command": "sleep 5",
            "cwd": str(tmp_path),
            "timeout_seconds": 0.1,
            "output_limit_bytes": 1024,
        },
    )

    assert "timed out" in observation.error
    assert observation.returncode is not None


def test_execute_command_kills_background_descendant_after_shell_exit(plugin, tmp_path):
    launched = {}
    observation = plugin._execute_command(
        "watch-background",
        {
            "revision": "rev",
            "command": (
                "python -c 'import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)' &"
            ),
            "cwd": str(tmp_path),
            "timeout_seconds": 2.0,
            "output_limit_bytes": 1024,
        },
        on_process=lambda process: launched.setdefault("process", process),
    )

    process = launched["process"]
    assert observation.error == ""
    assert observation.returncode == 0
    assert not plugin._process_group_exists(process.pid)


class Builder:
    def __init__(self):
        self.features = []

    def add(self, feature):
        self.features.append(feature)


def registration_session(
    session_id: str,
    *,
    kind: str = "default",
    parent_session_id: str = "",
):
    state = make_state(session_id)
    state._session_kind = kind
    state._parent_session_id = parent_session_id
    return SimpleNamespace(state=state)


def test_register_features_is_disabled_by_default(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "unrelated-parent")

    plugin.register_features(
        builder,
        session=registration_session("root-session"),
        config={},
    )

    assert builder.features == []
    assert plugin.FORK_ENABLE_ENV not in plugin.os.environ


def test_installed_config_enables_tools_in_fresh_session(plugin, monkeypatch, tmp_path):
    config_path = tmp_path / "event_architecture.json"
    config_path.write_text(
        '{"event_architecture": {"enabled": true}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(plugin.CONFIG_PATH_ENV, str(config_path))
    builder = Builder()

    plugin.register_features(
        builder,
        session=registration_session("root-session"),
        config={},
    )

    assert [feature.name for feature in builder.features] == ["event_architecture"]


def test_installed_config_can_disable_tools(plugin, monkeypatch, tmp_path):
    config_path = tmp_path / "event_architecture.json"
    config_path.write_text(
        '{"event_architecture": {"enabled": false}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(plugin.CONFIG_PATH_ENV, str(config_path))
    builder = Builder()

    plugin.register_features(
        builder,
        session=registration_session("root-session"),
        config={},
    )

    assert builder.features == []


def test_explicit_config_overrides_installed_config(plugin, monkeypatch, tmp_path):
    config_path = tmp_path / "event_architecture.json"
    config_path.write_text(
        '{"event_architecture": {"enabled": true}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(plugin.CONFIG_PATH_ENV, str(config_path))
    builder = Builder()

    plugin.register_features(
        builder,
        session=registration_session("root-session"),
        config={"event_architecture": {"enabled": False}},
    )

    assert builder.features == []


def test_register_features_adds_tools_when_enabled(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.delenv(plugin.FORK_ENABLE_ENV, raising=False)
    plugin.register_features(
        builder,
        session=registration_session("root-session"),
        config={"event_architecture": {"enabled": True}},
    )

    assert plugin.os.environ[plugin.FORK_ENABLE_ENV] == "root-session"
    assert [feature.name for feature in builder.features] == ["event_architecture"]
    components = builder.features[0].components
    tool_names = {component.name for component in components if hasattr(component, "name")}
    assert {
        "fork_agent",
        "subscribe_interrupt",
        "list_interrupt_subscriptions",
        "remove_interrupt_subscription",
    }.issubset(tool_names)
    assert any(isinstance(component, plugin.CommandWatchCheck) for component in components)


def test_ephemeral_fork_registration_adds_suspend_tool(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "parent-session")
    session = registration_session(
        "child-session",
        kind="fork",
        parent_session_id="parent-session",
    )
    authorize_ephemeral_fork(plugin, session.state)

    plugin.register_features(builder, session=session, config={})

    components = builder.features[0].components
    tool_names = {
        component.name for component in components if hasattr(component, "name")
    }
    assert "suspend_session" in tool_names
    assert plugin.suspend_session.available(session.state) is True
    plugin.suspend_session(session.state)
    assert session.state.tool_schemas["suspend_session"]["available"] is True


def test_persistent_fork_registration_gates_suspend_tool(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "parent-session")
    session = registration_session(
        "child-session",
        kind="fork",
        parent_session_id="parent-session",
    )

    plugin.register_features(builder, session=session, config={})

    components = builder.features[0].components
    tool_names = {
        component.name for component in components if hasattr(component, "name")
    }
    assert "suspend_session" in tool_names
    assert plugin.suspend_session.available(session.state) is False
    plugin.suspend_session(session.state)
    assert session.state.tool_schemas["suspend_session"]["available"] is False


def test_nonfork_registration_excludes_suspend_tool(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.delenv(plugin.FORK_ENABLE_ENV, raising=False)
    session = registration_session("root-session")

    plugin.register_features(
        builder,
        session=session,
        config={"event_architecture": {"enabled": True}},
    )

    components = builder.features[0].components
    tool_names = {
        component.name for component in components if hasattr(component, "name")
    }
    assert "suspend_session" not in tool_names


def test_fork_session_inherits_and_rotates_launch_enablement(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "parent-session")
    session = registration_session(
        "child-session",
        kind="fork",
        parent_session_id="parent-session",
    )

    plugin.register_features(builder, session=session, config={})

    assert [feature.name for feature in builder.features] == ["event_architecture"]
    assert plugin.os.environ[plugin.FORK_ENABLE_ENV] == "child-session"


def test_fork_session_rejects_other_lineage_marker(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "different-parent")
    session = registration_session(
        "child-session",
        kind="fork",
        parent_session_id="parent-session",
    )

    plugin.register_features(builder, session=session, config={})

    assert builder.features == []
    assert plugin.FORK_ENABLE_ENV not in plugin.os.environ


def test_explicit_false_disables_fork_inheritance(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "parent-session")
    session = registration_session(
        "child-session",
        kind="fork",
        parent_session_id="parent-session",
    )

    plugin.register_features(
        builder,
        session=session,
        config={"event_architecture": {"enabled": False}},
    )

    assert builder.features == []
    assert plugin.FORK_ENABLE_ENV not in plugin.os.environ


def test_fork_enablement_propagates_through_child_process(plugin, monkeypatch):
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "parent-session")
    code = f"""
import importlib.util
import os
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('event_architecture_child', {str(PLUGIN_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class Builder:
    def __init__(self): self.features = []
    def add(self, feature): self.features.append(feature)
state = SimpleNamespace(
    run_db=SimpleNamespace(session_id='child-session'),
    _session_id='child-session',
    _session_kind='fork',
    _parent_session_id='parent-session',
)
builder = Builder()
module.register_features(builder, session=SimpleNamespace(state=state), config={{}})
print(','.join(feature.name for feature in builder.features))
print(os.environ.get(module.FORK_ENABLE_ENV, ''))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == ["event_architecture", "child-session"]


def test_real_reactor_scheduler_delivers_interrupt(plugin, tmp_path):
    state = make_state("session-live")
    state.azo_event_subscriptions = {
        "watch-live": {
            "id": "watch-live",
            "name": "live-ready",
            "owner_session_id": "session-live",
            "command": "printf READY",
            "match_regex": "READY",
            "match_stream": "stdout",
            "interval_seconds": 60.0,
            "timeout_seconds": 2.0,
            "output_limit_bytes": 1024,
            "trigger": "each",
            "fire_initial": True,
            "wake_on_error": True,
            "enabled": True,
            "cwd": str(tmp_path),
            "revision": "live-revision",
            "runtime": {},
        }
    }
    reactor = plugin.CommandWatchReactor()
    check = plugin.CommandWatchCheck(reactor)
    deadline = time.monotonic() + 3.0

    while time.monotonic() < deadline and not state.pending_interrupts:
        check(state)
        time.sleep(0.02)

    reactor.cleanup()
    assert state.pending_interrupts
    assert "live-ready" in state.pending_interrupts[0]
    assert "READY" in state.pending_interrupts[0]


def test_reactor_completion_queue_is_bounded(plugin):
    reactor = plugin.CommandWatchReactor()
    for index in range(200):
        reactor._put_observation(
            plugin.CommandObservation(
                f"watch-{index}", "rev", float(index), session_id="session-a"
            )
        )

    observations = reactor.drain()

    assert len(observations) == 128
    assert observations[0].subscription_id == "watch-72"
    assert observations[-1].subscription_id == "watch-199"


def test_reactor_removal_terminates_active_process(plugin, tmp_path):
    reactor = plugin.CommandWatchReactor()
    reactor.sync(
        "session-a",
        {
            "watch-long": {
                "revision": "rev",
                "command": "sleep 30",
                "cwd": str(tmp_path),
                "interval_seconds": 60.0,
                "timeout_seconds": 60.0,
                "output_limit_bytes": 1024,
                "enabled": True,
            }
        },
    )
    deadline = time.monotonic() + 3.0
    process = None
    while time.monotonic() < deadline and process is None:
        with reactor._lock:
            process = next(
                (
                    worker.get("process")
                    for worker in reactor._workers.values()
                    if worker.get("process") is not None
                ),
                None,
            )
        time.sleep(0.01)

    assert process is not None
    reactor.sync("session-a", {})
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.01)

    reactor.cleanup()
    assert process.poll() is not None


def test_session_generation_change_discards_queued_observations(plugin):
    reactor = plugin.CommandWatchReactor()
    reactor.sync("session-a", {})
    first_generation = reactor.current_generation
    reactor._put_observation(
        plugin.CommandObservation(
            "watch-a",
            "rev",
            1.0,
            session_id="session-a",
            generation=first_generation,
        )
    )

    reactor.sync("session-b", {})
    reactor.sync("session-a", {})

    assert reactor.current_generation == first_generation + 2
    assert reactor.drain() == []


def test_cleanup_waits_for_worker_in_launch_registration_window(
    plugin, monkeypatch, tmp_path
):
    reactor = plugin.CommandWatchReactor()
    real_popen = plugin.subprocess.Popen
    launch_entered = threading.Event()
    release_launch = threading.Event()
    launched = {}

    def delayed_popen(*args, **kwargs):
        launch_entered.set()
        assert release_launch.wait(timeout=3.0)
        process = real_popen(*args, **kwargs)
        launched["process"] = process
        return process

    monkeypatch.setattr(plugin.subprocess, "Popen", delayed_popen)
    reactor.sync(
        "session-a",
        {
            "watch-race": {
                "revision": "rev",
                "command": "sleep 30",
                "cwd": str(tmp_path),
                "interval_seconds": 60.0,
                "timeout_seconds": 60.0,
                "output_limit_bytes": 1024,
                "enabled": True,
            }
        },
    )
    assert launch_entered.wait(timeout=3.0)

    cleanup_thread = threading.Thread(target=reactor.cleanup)
    cleanup_thread.start()
    time.sleep(0.05)
    assert cleanup_thread.is_alive()
    release_launch.set()
    cleanup_thread.join(timeout=4.0)

    assert not cleanup_thread.is_alive()
    assert launched["process"].poll() is not None
    with reactor._lock:
        assert reactor._workers == {}


def test_replacement_waits_for_cancelled_generation_to_exit(
    plugin, monkeypatch, tmp_path
):
    reactor = plugin.CommandWatchReactor()
    real_terminate = plugin._terminate_process_group
    termination_entered = threading.Event()
    release_termination = threading.Event()

    def delayed_terminate(process):
        termination_entered.set()
        assert release_termination.wait(timeout=4.0)
        real_terminate(process)

    monkeypatch.setattr(plugin, "_terminate_process_group", delayed_terminate)
    old_spec = {
        "revision": "old",
        "command": "sleep 30",
        "cwd": str(tmp_path),
        "interval_seconds": 60.0,
        "timeout_seconds": 60.0,
        "output_limit_bytes": 1024,
        "enabled": True,
    }
    reactor.sync("session-a", {"watch-same": old_spec})
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        with reactor._lock:
            if any(worker.get("process") is not None for worker in reactor._workers.values()):
                break
        time.sleep(0.01)
    else:
        raise AssertionError("old generation did not launch")

    new_spec = dict(old_spec, revision="new", command="printf NEW")
    sync_thread = threading.Thread(
        target=lambda: reactor.sync("session-a", {"watch-same": new_spec})
    )
    sync_thread.start()
    assert termination_entered.wait(timeout=3.0)
    time.sleep(1.1)
    with reactor._lock:
        assert {worker["revision"] for worker in reactor._workers.values()} == {"old"}

    release_termination.set()
    sync_thread.join(timeout=4.0)
    assert not sync_thread.is_alive()
    deadline = time.monotonic() + 3.0
    observations = []
    while time.monotonic() < deadline and not observations:
        observations = reactor.drain()
        time.sleep(0.01)

    reactor.cleanup()
    assert [observation.revision for observation in observations] == ["new"]
