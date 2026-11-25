from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utils.core.events import (
    IoCompleted,
    IoFailed,
    RenderRequested,
    RunToolCall,
    SessionRestored,
    SessionStarted,
    TurnSettled,
)
from agent_utils.core.io import FsStat, SubmitOp
from agent_utils.core.pipeline import Feature, PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.core.state import Entry, State, ToolCall
from agent_utils.core.tool import Tool
from agent_utils.features.core import IoFence, SchemaPublisher
from agent_utils.runtime.drivers.fs import FsStatResult
from agent_utils.runtime.simulation import SimOutcome, SimulationRuntime
from agent_zoo.core.feature_state import session_state
from agent_zoo.drivers.sessions import SimSessionDriver
from agent_zoo.drivers.sessions.live import _build_runtime_launch
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
from agent_zoo.pipelines.default import build_default_builder
from agent_zoo.storage.store import CommitReceipt
from agent_zoo.testing.sans_io_guard import assert_pipeline_sans_io, audit_module_sans_io


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
    monkeypatch.setenv("AGENT_ZOO_STATE_ROOT", str(tmp_path / "state-root"))
    monkeypatch.delenv("AZO_EVENT_ARCHITECTURE_CONFIG", raising=False)
    monkeypatch.delenv("AZO_EVENT_ARCHITECTURE_FORK_ENABLED", raising=False)
    return load_plugin()


def make_state(session_id: str = "session-a", *, cwd: str = "") -> State:
    state = State(cwd=cwd or str(Path.cwd()))
    smeta = session_state(state)
    smeta.session_id = session_id
    state.pending_tool_calls = []
    state.pending_interrupts = []
    state._interrupt_state = {}
    state.tool_schemas = {}
    return state


def authorize_ephemeral_fork(plugin, state: State) -> None:
    session_state(state).kind = "fork"
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


def call_tool(tool, state: State, *, stat_result: FsStatResult | None = None, **kwargs):
    tc = ToolCall(
        id="call-1",
        name=tool.name,
        raw_args="{}",
        parsed_args=kwargs,
    )
    state.pending_tool_calls = [tc]
    emitted = []
    tool.handle(RunToolCall(tc), state, emitted.append)
    if tc.waiting_op is not None:
        # Drive FsStat generator tools in unit tests without touching real OS I/O.
        outbox = state.io.take_outbox()
        remaining = []
        for eff in outbox:
            if isinstance(eff, SubmitOp) and eff.op_id == tc.waiting_op and isinstance(eff.request, FsStat):
                res = stat_result or FsStatResult(
                    path=eff.request.path,
                    exists=True,
                    is_file=False,
                    is_dir=True,
                )
                tool.handle(IoCompleted(op_id=eff.op_id, value=res), state, emitted.append)
            else:
                remaining.append(eff)
        # Re-queue any non-FsStat effects (e.g. SyncCommandWatches) so caller can inspect them.
        if remaining:
            state.io._outbox.extend(remaining)
    state._emitted = emitted
    return tc


def publish_schemas(components, state: State) -> None:
    publisher = SchemaPublisher()
    publisher.bind_pipeline(tuple(components))
    publisher.handle(RenderRequested(), state, lambda _ev: None)


def test_static_and_pipeline_sans_io_audit(plugin, tmp_path):
    assert audit_module_sans_io(PLUGIN_PATH) == []

    session = Session(state=make_state("root-session", cwd=str(tmp_path)))
    config = {"event_architecture": {"enabled": True}}
    builder = build_default_builder(
        session,
        config=config,
        include_plugins=False,
        include_llm_drivers=False,
    )
    plugin.register_features(builder, session=session, config=config)
    pipeline = builder.build()
    pipeline.init_state(session.state)
    built_drivers = builder.build_drivers()
    assert any(isinstance(d, plugin.CommandWatchDriver) for d in built_drivers)
    assert plugin.DURABLE_STATE_KEY in pipeline.durable_specs
    assert pipeline.durable_specs[plugin.DURABLE_STATE_KEY].on_fork == "reset"
    assert_pipeline_sans_io(
        pipeline,
        session=session,
        events=(SessionStarted(), SessionRestored(), RenderRequested(), TurnSettled()),
    )


def test_fork_agent_submits_fork_session_op(plugin):
    state = make_state("parent-session")

    tc = call_tool(plugin.fork_agent, state, message="Monitor issues independently.")

    assert tc.error is False
    assert "Fork requested" in tc.result
    effects = state.io.take_outbox()
    assert len(effects) == 1
    assert isinstance(effects[0], SubmitOp)
    req = effects[0].request
    assert isinstance(req, ForkSessionOp)
    assert req.spawn_policy == "fork"
    assert req.spawn_child is False
    assert req.first_user_message.startswith(f"{plugin.PERSISTENT_FORK_MARKER}\n")
    assert req.first_user_message.endswith("\n\nMonitor issues independently.")
    assert effects[0].op_id in plugin._event_runtime(state).fork_ops
    # Ephemeral runtime ops never leak into serialized checkpoint documents.
    assert "fork_ops" not in json.dumps(req.source_document)


def test_fork_agent_defaults_to_persistent_lifecycle(plugin):
    state = make_state("parent-session")

    tc = call_tool(plugin.fork_agent, state, message="Remain available for work.")

    assert tc.error is False
    req = state.io.take_outbox()[0].request
    assert isinstance(req, ForkSessionOp)
    assert req.spawn_child is False
    assert req.first_user_message.startswith(f"{plugin.PERSISTENT_FORK_MARKER}\n")
    assert plugin.PERSISTENT_FORK_GUIDANCE in req.first_user_message
    assert plugin.EPHEMERAL_FORK_MARKER not in req.first_user_message
    assert req.first_user_message.endswith("\n\nRemain available for work.")


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
    req = state.io.take_outbox()[0].request
    assert isinstance(req, ForkSessionOp)
    assert req.spawn_child is False
    assert req.first_user_message.startswith(f"{plugin.EPHEMERAL_FORK_MARKER}\n")
    assert plugin.EPHEMERAL_FORK_GUIDANCE in req.first_user_message
    assert req.first_user_message.endswith("\n\nHandle one issue and stop.")


@pytest.mark.parametrize("lifecycle", ["persistent", "ephemeral"])
def test_fork_agent_unified_session_driver_pipeline(plugin, lifecycle, tmp_path):
    config = {
        "event_architecture": {"enabled": True},
        "llm": {"default": "openai/gpt-5"},
        "pilot": {"_env_name": "dev"},
        "_runtime_launch": {
            "config_path": "/tmp/azo.yaml",
            "config_sets": ["event_architecture.enabled=true"],
        },
    }
    session = Session(cwd=str(tmp_path))
    state = session.state
    smeta = session_state(state)
    smeta.session_id = "parent-session"
    smeta.project_name = "demo-project"
    smeta.title = "Parent Work"
    state._agent_zoo_context = {
        "project_name": "demo-project",
        "pipeline_key": "default",
        "state_root": str(tmp_path / "state"),
        "project_store_root": str(tmp_path / "store"),
    }
    state._persistence_local_root = str(tmp_path / "local")
    state.config = config
    state.pending_tool_calls = []

    seen_forks: list[ForkSessionOp] = []
    seen_spawns: list[SpawnSession] = []
    seen_views: list[PublishView] = []
    seen_notices: list[EmitNotice] = []
    seen_projections: list[ProjectionRequested] = []

    class _EventSpy:
        handles = (EmitNotice, ProjectionRequested)

        def handle(self, ev, _st, _emit):
            if isinstance(ev, EmitNotice):
                seen_notices.append(ev)
            elif isinstance(ev, ProjectionRequested):
                seen_projections.append(ev)

    builder = PipelineBuilder()
    builder.add(Feature("test_core", components=[IoFence()], order=[IoFence, Tool]))
    plugin.register_features(builder, session=session, config=config)
    builder.add(
        Feature(
            "test_order",
            components=[_EventSpy()],
            order=[plugin.ForkSessionCoordinator, _EventSpy, Tool],
        )
    )
    session.pipeline = builder.build()
    session.ensure_initialized()

    sim = SimulationRuntime(session)
    sim_sessions = SimSessionDriver().register(sim)

    def _capture_session_req(op_id, req, runtime):
        if isinstance(req, SpawnSession):
            seen_spawns.append(req)
        return sim_sessions.handle(op_id, req, runtime)

    sim.register_handler(SpawnSession, _capture_session_req)
    sim.register_handler(
        ForkSessionOp,
        lambda _op, req, _rt: (
            seen_forks.append(req),
            SimOutcome(
                value=CommitReceipt(
                    session_id="child-fork-1234",
                    instance_id="fork-ckpt-inst-1",
                    revision_id="rev-fork-1",
                    parent_revision_id=None,
                    local_saved=True,
                ),
                delay=0.01,
            ),
        )[1],
    )
    sim.register_handler(
        PublishView,
        lambda _op, req, _rt: (
            seen_views.append(req),
            SimOutcome(value=True, delay=0.0),
        )[1],
    )

    tc = ToolCall(
        id="call-fork-1",
        name="fork_agent",
        raw_args="{}",
        parsed_args={"message": "Handle incoming events.", "lifecycle": lifecycle},
    )
    state.pending_tool_calls = [tc]
    sim.inject(RunToolCall(tc))
    sim.run_until_idle()

    assert tc.error is False
    assert len(seen_forks) == 1
    assert seen_forks[0].spawn_child is False
    assert seen_forks[0].spawn_policy == "fork"

    assert len(seen_spawns) == 1
    spawn = seen_spawns[0]
    assert spawn.session_id == "child-fork-1234"
    assert spawn.source == SessionSource(
        kind="resume",
        session_id="child-fork-1234",
        revision_id="rev-fork-1",
        instance_id="fork-ckpt-inst-1",
    )
    assert isinstance(spawn.spec, SessionSpec)
    assert spawn.spec.project == "demo-project"
    assert spawn.spec.workdir == str(tmp_path)
    assert spawn.spec.model == "openai/gpt-5"
    assert spawn.spec.pipeline_key == "default"
    assert spawn.spec.env_name == "dev"
    assert spawn.spec.config_path == "/tmp/azo.yaml"
    assert spawn.spec.config_sets == ("event_architecture.enabled=true",)
    assert spawn.spec.kind == "fork"
    assert spawn.spec.spawn_policy == "fork"
    assert spawn.spec.parent_session_id == "parent-session"
    assert spawn.spec.lifetime == "independent"
    assert spawn.spec.extra_env == ((plugin.FORK_ENABLE_ENV, "parent-session"),)
    assert spawn.launch_metadata["fork_lifecycle"] == lifecycle
    assert spawn.launch_metadata["parent_session_id"] == "parent-session"

    launch = _build_runtime_launch(
        session_id=spawn.session_id,
        instance_id="runtime-inst-1",
        source=spawn.source,
        spec=spawn.spec,
    )
    assert launch.target.session_id == "child-fork-1234"
    assert launch.spawn_policy == "fork"
    assert launch.kind == "fork"

    assert len(spawn.inputs) == 1
    artifact = spawn.inputs[0]
    assert isinstance(artifact, SessionArtifact)
    assert artifact.name == plugin.FORK_ARTIFACT_NAME
    assert artifact.kind == "json"
    assert artifact.data["lifecycle"] == lifecycle
    assert artifact.data["parent_session_id"] == "parent-session"

    assert len(sim_sessions.sessions) == 1
    sim_child = next(iter(sim_sessions.sessions.values()))
    assert sim_child.session_id == "child-fork-1234"
    assert sim_child.inputs == spawn.inputs

    assert len(seen_views) == 1
    assert seen_views[0].attach == {"session_id": "child-fork-1234", "kind": "fork"}
    assert any(
        ev.level == "info" and f"Spawned {lifecycle} fork session" in ev.text
        for ev in seen_notices
    )
    assert len(seen_projections) >= 1
    assert plugin._event_runtime(state).fork_ops == {}


def test_fork_session_coordinator_handles_fork_and_spawn_failures(plugin):
    state = make_state("parent-session")
    coordinator = plugin.ForkSessionCoordinator()

    # 1. ForkSessionOp IoFailed
    tc = call_tool(plugin.fork_agent, state, message="Do work.", lifecycle="ephemeral")
    assert tc.error is False
    fork_op_id = state.io.take_outbox()[0].op_id
    emitted = []
    coordinator.handle(IoFailed(op_id=fork_op_id, error=RuntimeError("boom")), state, emitted.append)
    assert state.io.take_outbox() == []
    assert any(isinstance(ev, EmitNotice) and ev.level == "error" and "boom" in ev.text for ev in emitted)
    assert any(isinstance(ev, ProjectionRequested) for ev in emitted)

    # 2. ForkSessionOp returns unsaved CommitReceipt
    call_tool(plugin.fork_agent, state, message="Do work.")
    fork_op_id = state.io.take_outbox()[0].op_id
    emitted = []
    coordinator.handle(
        IoCompleted(
            op_id=fork_op_id,
            value=CommitReceipt(
                session_id="child-unsaved",
                instance_id="inst-1",
                revision_id=None,
                parent_revision_id=None,
                local_saved=False,
                error="disk full",
            ),
        ),
        state,
        emitted.append,
    )
    assert state.io.take_outbox() == []
    assert any(isinstance(ev, EmitNotice) and ev.level == "error" and "disk full" in ev.text for ev in emitted)

    # 3. ForkSessionOp succeeds, SpawnSession fails
    call_tool(plugin.fork_agent, state, message="Do work.")
    fork_op_id = state.io.take_outbox()[0].op_id
    emitted = []
    coordinator.handle(
        IoCompleted(
            op_id=fork_op_id,
            value={"session_id": "child-2", "revision_id": "rev-2", "instance_id": "inst-2"},
        ),
        state,
        emitted.append,
    )
    spawn_effects = state.io.take_outbox()
    assert len(spawn_effects) == 1
    assert isinstance(spawn_effects[0].request, SpawnSession)

    emitted = []
    coordinator.handle(
        IoFailed(op_id=spawn_effects[0].op_id, error=OSError("spawn failed")),
        state,
        emitted.append,
    )
    assert state.io.take_outbox() == []
    assert any(isinstance(ev, EmitNotice) and ev.level == "error" and "spawn failed" in ev.text for ev in emitted)
    assert any(isinstance(ev, ProjectionRequested) for ev in emitted)


def test_ephemeral_fork_authorized_from_session_artifact_inputs(plugin):
    artifact = plugin._build_fork_lifecycle_artifact("ephemeral", parent_session_id="parent-session")

    child = make_state("child-session")
    session_state(child).kind = "fork"
    session_state(child).parent_session_id = "parent-session"
    child._azo_session_inputs = (artifact,)
    assert plugin._latest_fork_lifecycle(child) == "ephemeral"
    assert plugin.suspend_session.available(child) is True

    # Mismatched parent lineage rejects the inherited artifact.
    unrelated = make_state("other-child")
    session_state(unrelated).kind = "fork"
    session_state(unrelated).parent_session_id = "different-parent"
    unrelated._azo_session_inputs = (artifact,)
    assert plugin._latest_fork_lifecycle(unrelated) == ""
    assert plugin.suspend_session.available(unrelated) is False

    # Root sessions ignore fork_lifecycle artifacts.
    root = make_state("root-session")
    session_state(root).kind = "default"
    root._azo_session_inputs = (artifact,)
    assert plugin._latest_fork_lifecycle(root) == ""
    assert plugin.suspend_session.available(root) is False

    # Newer transcript marker always overrides an inherited artifact.
    child.entries.append(
        Entry(
            messages=[{"role": "user", "content": f"{plugin.PERSISTENT_FORK_MARKER}\nnow persistent"}],
            index=0,
            step=0,
        )
    )
    assert plugin._latest_fork_lifecycle(child) == "persistent"
    assert plugin.suspend_session.available(child) is False


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
    assert state.io.take_outbox() == []


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
    session_state(state).kind = "fork"
    state.last_tool_calls = []

    tc = call_tool(plugin.suspend_session, state)

    assert tc.error is True
    assert "not currently available" in (tc.result or "")
    assert plugin._event_durable(state).suspend_confirmation_pending is False
    assert state.done is False


def test_suspend_session_requires_adjacent_confirmation_and_defers_to_turn_settled(plugin):
    state = make_state("ephemeral-child")
    authorize_ephemeral_fork(plugin, state)
    state.last_tool_calls = []
    hook = plugin.EphemeralSuspendHook()

    first = call_tool(plugin.suspend_session, state)

    assert first.error is False
    assert first.result == plugin.SUSPEND_CONFIRMATION_MESSAGE
    assert plugin._event_durable(state).suspend_confirmation_pending is True
    assert plugin._event_runtime(state).pending_suspend is False
    assert state.done is False
    assert not any(isinstance(ev, SuspendRequested) for ev in state._emitted)

    state.last_tool_calls = [first]
    second = call_tool(plugin.suspend_session, state)

    assert second.error is False
    assert "suspension confirmed" in (second.result or "")
    assert plugin._event_durable(state).suspend_confirmation_pending is False
    # Tool MUST NOT mutate state.done or emit SuspendRequested mid-RunToolCall!
    assert plugin._event_runtime(state).pending_suspend is True
    assert state.done is False
    assert not any(isinstance(ev, SuspendRequested) for ev in state._emitted)

    # On TurnSettled, EphemeralSuspendHook emits SuspendRequested cleanly.
    hook_emitted = []
    hook.handle(TurnSettled(), state, hook_emitted.append)
    assert plugin._event_runtime(state).pending_suspend is False
    assert state.done is True
    assert state.shutdown is True
    assert state.shutdown_reason == "suspend"
    assert state.shutdown_save is True
    assert state.shutdown_registry_status == "stopped"
    assert state.shutdown_spawned_policy == "leave"
    suspends = [ev for ev in hook_emitted if isinstance(ev, SuspendRequested)]
    assert len(suspends) == 1
    assert suspends[0].reason == "ephemeral fork completed"
    assert suspends[0].spawned_policy == "leave"


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
    assert plugin._event_durable(state).suspend_confirmation_pending is True
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
        ),
    ]

    retry = call_tool(plugin.suspend_session, state)

    assert retry.result == plugin.SUSPEND_CONFIRMATION_MESSAGE
    assert plugin._event_durable(state).suspend_confirmation_pending is True
    assert state.done is False


def test_owned_subscriptions_are_session_local_and_non_destructive(plugin):
    state = make_state("child-session")
    plugin._event_durable(state).subscriptions = {
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
    assert set(plugin._event_durable(state).subscriptions) == {"parent", "child"}
    assert plugin._event_durable(state).subscriptions["parent"]["runtime"] == {"runs": 3}


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


def test_subscribe_list_and_remove_tool_lifecycle(plugin, tmp_path):
    state = make_state("owner-session", cwd=str(tmp_path))

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
    subscriptions = plugin._event_durable(state).subscriptions
    assert len(subscriptions) == 1
    subscription_id, spec = next(iter(subscriptions.items()))
    assert subscription_id.startswith("watch_")
    assert spec["owner_session_id"] == "owner-session"
    assert spec["interval_seconds"] == plugin.MIN_INTERVAL_SECONDS
    assert spec["cwd"] == str(tmp_path)

    sync_effects = [
        eff.request for eff in state.io.take_outbox()
        if isinstance(eff, SubmitOp) and isinstance(eff.request, plugin.SyncCommandWatches)
    ]
    assert len(sync_effects) == 1
    assert sync_effects[0].session_id == "owner-session"
    assert subscription_id in sync_effects[0].subscriptions

    listed = call_tool(plugin.list_interrupt_subscriptions, state)
    assert listed.error is False
    assert "new-prs" in listed.result
    assert "owner-session" in listed.result

    removed = call_tool(plugin.remove_interrupt_subscription, state, subscription="new-prs")
    assert removed.error is False
    assert subscription_id not in plugin._event_durable(state).subscriptions
    remove_syncs = [
        eff.request for eff in state.io.take_outbox()
        if isinstance(eff, SubmitOp) and isinstance(eff.request, plugin.SyncCommandWatches)
    ]
    assert len(remove_syncs) == 1
    assert remove_syncs[0].subscriptions == {}


def test_subscribe_rejects_non_directory_via_fs_stat(plugin, tmp_path):
    state = make_state("owner-session", cwd=str(tmp_path))
    tc = call_tool(
        plugin.subscribe_interrupt,
        state,
        name="bad-cwd",
        command="true",
        match_regex=".*",
        cwd=str(tmp_path / "missing"),
        stat_result=FsStatResult(path=str(tmp_path / "missing"), exists=False, is_dir=False),
    )
    assert tc.error is True
    assert "cwd is not a directory" in (tc.result or "")


def test_subscribe_rejects_invalid_regex(plugin):
    state = make_state()

    tc = call_tool(
        plugin.subscribe_interrupt,
        state,
        name="broken",
        command="true",
        match_regex="[",
    )

    assert tc.error is True
    assert "invalid match_regex" in tc.result
    assert state.io.take_outbox() == []


def test_interrupt_check_discards_stale_revision_and_fires_current(plugin):
    state = make_state("session-a")
    plugin._event_durable(state).subscriptions = {
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
    check = plugin.CommandWatchCheck()

    # TurnSettled submits SyncCommandWatches; IoCompleted sets driver_generation=3.
    check.handle(TurnSettled(), state, lambda _ev: None)
    outbox = state.io.take_outbox()
    assert len(outbox) == 1
    assert isinstance(outbox[0].request, plugin.SyncCommandWatches)
    assert outbox[0].request.session_id == "session-a"
    check.handle(
        IoCompleted(op_id=outbox[0].op_id, value={"generation": 3, "session_id": "session-a"}),
        state,
        lambda _ev: None,
    )

    # Stale generation, wrong session, stale revision are all discarded; current fires.
    observations = [
        plugin.CommandObservation("watch-1", "current", 1.0, stdout="READY", returncode=0, session_id="session-a", generation=2),
        plugin.CommandObservation("watch-1", "current", 2.0, stdout="READY", returncode=0, session_id="other-session", generation=3),
        plugin.CommandObservation("watch-1", "stale", 3.0, stdout="READY", returncode=0, session_id="session-a", generation=3),
        plugin.CommandObservation("watch-1", "current", 4.0, stdout="READY", returncode=0, session_id="session-a", generation=3),
    ]
    for obs in observations:
        check.handle(
            plugin.CommandWatchTriggered(observation=obs, generation=obs.generation),
            state,
            lambda _ev: None,
        )

    assert len(state.pending_interrupts) == 1
    assert "READY" in state.pending_interrupts[0]

    # SessionRestored clears ephemeral fork ops and re-syncs watches.
    plugin._event_runtime(state).fork_ops["stale-op"] = {"stage": "fork_checkpoint"}
    check.handle(SessionRestored(), state, lambda _ev: None)
    assert plugin._event_runtime(state).fork_ops == {}
    restore_outbox = state.io.take_outbox()
    assert len(restore_outbox) == 1
    assert isinstance(restore_outbox[0].request, plugin.SyncCommandWatches)
    assert restore_outbox[0].request.cleanup is False

    # ShutdownRequested submits cleanup sync.
    check.handle(ShutdownRequested(), state, lambda _ev: None)
    shutdown_outbox = state.io.take_outbox()
    assert len(shutdown_outbox) == 1
    assert isinstance(shutdown_outbox[0].request, plugin.SyncCommandWatches)
    assert shutdown_outbox[0].request.cleanup is True


def test_execute_command_captures_bounded_tail(plugin, tmp_path):
    observation = plugin._execute_command(
        "watch-1",
        {
            "revision": "rev",
            "command": f"{sys.executable} -c 'print(\"x\" * 5000)'",
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
                f"{sys.executable} -c 'import signal,time; "
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
    smeta = session_state(state)
    smeta.kind = kind
    smeta.parent_session_id = parent_session_id
    return SimpleNamespace(state=state)


def test_register_features_is_disabled_by_default_and_does_not_mutate_env(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "unrelated-parent")

    plugin.register_features(
        builder,
        session=registration_session("root-session"),
        config={},
    )

    assert [feature.name for feature in builder.features] == ["event_architecture_settings"]
    assert plugin.os.environ.get(plugin.FORK_ENABLE_ENV) == "unrelated-parent"


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

    assert [feature.name for feature in builder.features] == ["event_architecture_settings", "event_architecture"]


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

    assert [feature.name for feature in builder.features] == ["event_architecture_settings"]


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

    assert [feature.name for feature in builder.features] == ["event_architecture_settings"]


def test_register_features_adds_tools_driver_and_durable_spec_when_enabled(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.delenv(plugin.FORK_ENABLE_ENV, raising=False)
    plugin.register_features(
        builder,
        session=registration_session("root-session"),
        config={"event_architecture": {"enabled": True}},
    )

    # Does NOT mutate process environment.
    assert plugin.FORK_ENABLE_ENV not in plugin.os.environ
    assert [feature.name for feature in builder.features] == ["event_architecture_settings", "event_architecture"]
    feature = builder.features[1]
    assert feature.drivers == [plugin.CommandWatchDriver]
    assert feature.durable_state == plugin.EVENT_ARCHITECTURE_DURABLE_SPEC
    components = feature.components
    tool_names = {component.name for component in components if hasattr(component, "name")}
    assert {
        "fork_agent",
        "subscribe_interrupt",
        "list_interrupt_subscriptions",
        "remove_interrupt_subscription",
    }.issubset(tool_names)
    assert any(isinstance(component, plugin.ForkSessionCoordinator) for component in components)
    assert any(isinstance(component, plugin.CommandWatchCheck) for component in components)
    assert any(isinstance(component, plugin.EphemeralSuspendHook) for component in components)


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

    components = builder.features[1].components
    tool_names = {
        component.name for component in components if hasattr(component, "name")
    }
    assert "suspend_session" in tool_names
    assert plugin.suspend_session.available(session.state) is True
    publish_schemas(components, session.state)
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

    components = builder.features[1].components
    tool_names = {
        component.name for component in components if hasattr(component, "name")
    }
    assert "suspend_session" in tool_names
    assert plugin.suspend_session.available(session.state) is False
    publish_schemas(components, session.state)
    assert session.state.tool_schemas["suspend_session"]["available"] is False


def test_nonfork_registration_gates_suspend_tool(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.delenv(plugin.FORK_ENABLE_ENV, raising=False)
    session = registration_session("root-session")

    plugin.register_features(
        builder,
        session=session,
        config={"event_architecture": {"enabled": True}},
    )

    components = builder.features[1].components
    tool_names = {
        component.name for component in components if hasattr(component, "name")
    }
    assert "suspend_session" in tool_names
    assert plugin.suspend_session.available(session.state) is False
    publish_schemas(components, session.state)
    assert session.state.tool_schemas["suspend_session"]["available"] is False


def test_suspend_tool_becomes_available_after_fork_state_is_restored(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.delenv(plugin.FORK_ENABLE_ENV, raising=False)
    session = registration_session("child-session")

    plugin.register_features(
        builder,
        session=session,
        config={"event_architecture": {"enabled": True}},
    )

    components = builder.features[1].components
    assert plugin.suspend_session in components
    assert plugin.suspend_session.available(session.state) is False

    authorize_ephemeral_fork(plugin, session.state)
    publish_schemas(components, session.state)

    assert plugin.suspend_session.available(session.state) is True
    assert session.state.tool_schemas["suspend_session"]["available"] is True


def test_fork_session_inherits_launch_enablement_without_mutating_env(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "parent-session")
    session = registration_session(
        "child-session",
        kind="fork",
        parent_session_id="parent-session",
    )

    plugin.register_features(builder, session=session, config={})

    assert [feature.name for feature in builder.features] == ["event_architecture_settings", "event_architecture"]
    assert plugin.os.environ[plugin.FORK_ENABLE_ENV] == "parent-session"


def test_fork_session_rejects_other_lineage_marker(plugin, monkeypatch):
    builder = Builder()
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "different-parent")
    session = registration_session(
        "child-session",
        kind="fork",
        parent_session_id="parent-session",
    )

    plugin.register_features(builder, session=session, config={})

    assert [feature.name for feature in builder.features] == ["event_architecture_settings"]


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

    assert [feature.name for feature in builder.features] == ["event_architecture_settings"]


def test_command_watch_driver_pushes_events_and_delivers_interrupt(plugin, tmp_path):
    state = make_state("session-live", cwd=str(tmp_path))
    plugin._event_durable(state).subscriptions = {
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
    driver = plugin.CommandWatchDriver()
    assert driver.durability == "process_bound"
    assert driver.request_types == (plugin.SyncCommandWatches,)
    check = plugin.CommandWatchCheck()

    pushed_events = []
    lock = threading.Lock()

    def _emit(ev):
        with lock:
            pushed_events.append(ev)

    driver.start(_emit)
    try:
        check.handle(TurnSettled(), state, lambda _ev: None)
        outbox = state.io.take_outbox()
        assert len(outbox) == 1
        driver.submit(outbox[0].op_id, outbox[0].request)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not state.pending_interrupts:
            with lock:
                batch = list(pushed_events)
                pushed_events.clear()
            for ev in batch:
                check.handle(ev, state, lambda _ev: None)
            if not state.pending_interrupts:
                time.sleep(0.02)

        assert state.pending_interrupts
        assert "live-ready" in state.pending_interrupts[0]
        assert "READY" in state.pending_interrupts[0]
        assert driver.quiesce(policy="terminate") == []
    finally:
        driver.close()


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
