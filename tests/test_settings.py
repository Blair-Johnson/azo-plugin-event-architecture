"""Settings use PluginStorage + PluginStoreDriver + SettingsDispatcher (strict Sans-I/O)."""
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utils.core.events import IoCompleted, IoFailed, RenderRequested
from agent_utils.core.io import SubmitOp
from agent_utils.core.pipeline import Feature, PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.core.state import State
from agent_zoo.core.feature_state import session_state
from agent_zoo.durability import PluginStorage, PluginStoreDriver, StoreUpdate
from agent_zoo.events import ReplyFrontendRequest, SettingsRequest
from agent_zoo.features.settings import Setting, SettingsDispatcher
from agent_zoo.testing.sans_io_guard import assert_pipeline_sans_io, audit_module_sans_io
import event_architecture as plugin


def document():
    return {
        "event_architecture": {"enabled": False, "confirmation_message": "private prompt"},
        "unknown": {"prompt": "do not change", "ratio": 0.125},
    }


def dump(raw):
    return json.dumps(raw)


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_STATE_ROOT", str(tmp_path))
    monkeypatch.delenv(plugin.CONFIG_PATH_ENV, raising=False)
    monkeypatch.delenv(plugin.FORK_ENABLE_ENV, raising=False)
    storage = PluginStorage(plugin.PLUGIN_NAME, config_filename=plugin.CONFIG_REL_PATH)
    path = storage.config.path
    assert path == tmp_path / "plugin-configs" / plugin.PLUGIN_NAME / "config" / plugin.CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump(document()))
    path.chmod(0o640)
    return path


def build_dispatcher(config=None, session=None):
    session = session or Session(state=State())
    dispatcher = SettingsDispatcher(config=config or {})
    builder = PipelineBuilder()
    builder.add(Feature("test_settings_dispatcher", components=[dispatcher], order=[SettingsDispatcher, Setting]))
    plugin.register_features(builder, config=config or {})
    pipeline = builder.build()
    pipeline.init_state(session.state)
    settings = [component for component in pipeline.components if isinstance(component, Setting)]
    assert len(settings) == 1
    assert dispatcher.registry.error is None
    return dispatcher, pipeline, session, builder


def _run_settings_request(dispatcher, state, payload, *, state_root=None):
    """Step a SettingsRequest through dispatcher + real PluginStoreDriver."""
    driver = PluginStoreDriver(state_root=state_root)
    driver_events = []
    driver.start(driver_events.append)
    replies = []

    def _drain_outbox():
        for eff in state.io.take_outbox():
            if not isinstance(eff, SubmitOp):
                continue
            if isinstance(eff.request, ReplyFrontendRequest):
                replies.append(eff.request.payload)
            elif isinstance(eff.request, StoreUpdate):
                driver.submit(eff.op_id, eff.request)

    try:
        dispatcher.handle(SettingsRequest(raw_payload=payload), state, lambda _ev: None)
        _drain_outbox()
        driver.drain(timeout_s=5.0)
        while driver_events:
            ev = driver_events.pop(0)
            if isinstance(ev, (IoCompleted, IoFailed)):
                dispatcher.handle(ev, state, lambda _ev: None)
                _drain_outbox()
                driver.drain(timeout_s=5.0)
    finally:
        driver.close()
    assert len(replies) == 1
    return replies[0]


def list_rows(dispatcher, state):
    return dispatcher.list_settings(state)


def edit(dispatcher, state, value, *, state_root=None):
    dispatcher._ensure_registry(state)
    payload = {
        "type": "plugin_settings",
        "request_id": "test",
        "action": "set",
        "binding": dispatcher.registry.binding,
        "key": "event_architecture.enabled",
        "value": value,
    }
    return _run_settings_request(dispatcher, state, payload, state_root=state_root)


def test_disabled_registration_round_trip(settings_file, tmp_path):
    original = load(settings_file)
    dispatcher, pipeline, session, builder = build_dispatcher()
    assert audit_module_sans_io(Path(plugin.__file__)) == []
    assert_pipeline_sans_io(pipeline, session=session, events=(RenderRequested(),))

    row, = list_rows(dispatcher, session.state)
    assert row["value"] is False
    assert row["kind"] == "bool"
    assert row["apply"] == "reload" and row["writable"]
    assert row["error"] is None
    assert not builder.has("event_architecture")
    assert "private prompt" not in json.dumps(list_rows(dispatcher, session.state))

    result = edit(dispatcher, session.state, True, state_root=tmp_path)
    assert result["ok"]
    expected = original.copy()
    expected["event_architecture"] = dict(original["event_architecture"], enabled=True)
    assert load(settings_file) == expected
    assert stat.S_IMODE(settings_file.stat().st_mode) == 0o640
    # The existing builder is not live-mutated by a reload-scoped setting.
    assert not builder.has("event_architecture")
    fresh = PipelineBuilder()
    plugin.register_features(fresh, config={})
    assert fresh.has("event_architecture")
    assert list_rows(dispatcher, session.state)[0]["value"] is True
    # Only the config file (and advisory lock file) exist; no leaked temp files.
    names = {p.name for p in settings_file.parent.iterdir()}
    assert settings_file.name in names
    assert not any(n.startswith(".") and n != f".{settings_file.name}" for n in names)


@pytest.mark.parametrize("broken", ["[", "[]", "null", "", '{"event_architecture": []}'])
def test_bad_file_is_never_overwritten(settings_file, tmp_path, broken):
    settings_file.write_text(broken)
    before = settings_file.read_bytes()
    dispatcher, _pipeline, session, _ = build_dispatcher()
    assert list_rows(dispatcher, session.state)[0]["error"] is not None
    result = edit(dispatcher, session.state, True, state_root=tmp_path)
    assert not result["ok"]
    assert settings_file.read_bytes() == before


def test_failed_replace_preserves_file_and_rolls_back(settings_file, tmp_path, monkeypatch):
    import agent_utils.storage.durable_fs as durable_fs

    dispatcher, _pipeline, session, _ = build_dispatcher()
    before = settings_file.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("private path must not leak")

    monkeypatch.setattr(durable_fs.os, "replace", fail)
    result = edit(dispatcher, session.state, True, state_root=tmp_path)
    assert not result["ok"]
    assert "private path" not in json.dumps(result)
    assert settings_file.read_bytes() == before
    # Rolled back in-memory SettingsBox value!
    assert list_rows(dispatcher, session.state)[0]["value"] is False
    # No temp files left behind.
    assert not any(p.name.startswith(f".{settings_file.name}.") for p in settings_file.parent.iterdir())


def test_invalid_type_does_not_write(settings_file, tmp_path):
    dispatcher, _pipeline, session, _ = build_dispatcher()
    before = settings_file.read_bytes()
    assert not edit(dispatcher, session.state, 1, state_root=tmp_path)["ok"]
    assert settings_file.read_bytes() == before


def test_unreadable_file_is_safe(settings_file, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("private path")

    monkeypatch.setattr(Path, "read_text", fail)
    dispatcher, _pipeline, session, _ = build_dispatcher()
    row, = list_rows(dispatcher, session.state)
    assert row["error"] is not None
    assert "private path" not in json.dumps(row)
    result = edit(dispatcher, session.state, True, state_root=tmp_path)
    assert not result["ok"]
    assert "private path" not in json.dumps(result)


@pytest.mark.parametrize("override", [False, True])
def test_registration_override_does_not_replace_or_lock_saved_default(settings_file, tmp_path, override):
    saved = document()
    saved["event_architecture"]["enabled"] = not override
    settings_file.write_text(dump(saved))
    config = {"event_architecture": {"enabled": override}}
    dispatcher, _pipeline, session, builder = build_dispatcher(config)
    row, = list_rows(dispatcher, session.state)
    assert row["value"] is not override
    assert row["writable"] and row["kind"] == "bool"
    assert builder.has("event_architecture") is override
    assert edit(dispatcher, session.state, override, state_root=tmp_path)["ok"]
    assert edit(dispatcher, session.state, not override, state_root=tmp_path)["ok"]
    assert list_rows(dispatcher, session.state)[0]["value"] is not override
    assert load(settings_file)["event_architecture"]["enabled"] is not override
    assert config == {"event_architecture": {"enabled": override}}
    assert builder.has("event_architecture") is override
    fresh, _fresh_pipe, fresh_sess, fresh_builder = build_dispatcher(config)
    assert fresh_builder.has("event_architecture") is override
    assert list_rows(fresh, fresh_sess.state)[0]["value"] is not override


def test_missing_file_preserves_absent_defaults(settings_file, tmp_path):
    settings_file.unlink()
    dispatcher, _pipeline, session, _ = build_dispatcher()
    assert not settings_file.exists()
    assert list_rows(dispatcher, session.state)[0]["value"] is False
    assert edit(dispatcher, session.state, True, state_root=tmp_path)["ok"]
    assert load(settings_file) == {"event_architecture": {"enabled": True}}


@pytest.mark.parametrize("value", ["inherit", "enabled", "disabled"])
def test_old_choices_are_rejected(settings_file, tmp_path, value):
    dispatcher, _pipeline, session, _ = build_dispatcher()
    before = settings_file.read_bytes()
    assert not edit(dispatcher, session.state, value, state_root=tmp_path)["ok"]
    assert settings_file.read_bytes() == before


def test_fork_inheritance_is_not_the_saved_default(settings_file, tmp_path, monkeypatch):
    saved = document()
    saved["event_architecture"].pop("enabled")
    settings_file.write_text(dump(saved))
    before = settings_file.read_bytes()
    state = State()
    smeta = session_state(state)
    smeta.kind = "fork"
    smeta.parent_session_id = "parent"
    session = Session(state=state)
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "parent")
    assert plugin._plugin_enabled({}, session)
    dispatcher, _pipeline, session, builder = build_dispatcher(session=session)
    assert builder.has("event_architecture")
    row, = list_rows(dispatcher, session.state)
    assert row["value"] is False and row["kind"] == "bool"
    assert not row["choices"]
    assert settings_file.read_bytes() == before
    assert edit(dispatcher, session.state, False, state_root=tmp_path)["ok"]
    assert load(settings_file)["event_architecture"]["enabled"] is False
    assert list_rows(dispatcher, session.state)[0]["value"] is False
    # Saving defaults does not remove active fork tools or mutate os.environ.
    assert builder.has("event_architecture")
    assert plugin.os.environ[plugin.FORK_ENABLE_ENV] == "parent"
    # An explicit saved default is honored on a subsequent registration.
    _, _, _, fresh = build_dispatcher(session=Session(state=state))
    assert not fresh.has("event_architecture")
