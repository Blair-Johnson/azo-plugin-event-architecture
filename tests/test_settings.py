"""Settings use temporary files and the real feature/pipeline registry."""
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utils import Feature, PipelineBuilder, State, Tool
from agent_zoo.runtime.settings import Setting, SettingError, SettingsContext, SettingsRegistry
import event_architecture as plugin


def document():
    return {"event_architecture": {"enabled": False, "confirmation_message": "private prompt"},
            "unknown": {"prompt": "do not change", "ratio": 0.125}}


def dump(raw):
    return json.dumps(raw)


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text(dump(document()))
    path.chmod(0o640)
    monkeypatch.setenv("AZO_EVENT_ARCHITECTURE_CONFIG", str(path))
    monkeypatch.delenv(plugin.FORK_ENABLE_ENV, raising=False)
    return path


def registry(config=None, session=None):
    session = session or SimpleNamespace(state=State())
    builder = PipelineBuilder(session)
    plugin.register_features(builder, session=session, config=config or {})
    if builder.has("event_architecture"):
        builder.add(Feature("test_runtime_order", order=[Tool, plugin.CommandWatchCheck]))
    # Build the normal registration, including its settings while disabled.
    components = builder.build()
    settings = [component for component in components if isinstance(component, Setting)]
    result = SettingsRegistry(settings)
    assert result.error is None
    assert len(settings) == 1
    return result, SettingsContext(session), builder


def edit(registry, ctx, value):
    return registry.handle({"type": "plugin_settings", "request_id": "test", "action": "set",
                           "binding": registry.binding,
                           "key": "event_architecture.enabled", "value": value}, ctx)


def test_disabled_registration_round_trip(settings_file):
    original = load(settings_file)
    reg, ctx, builder = registry()
    row, = reg.catalog(ctx)
    assert row["value"] is False
    assert row["kind"] == "bool"
    assert row["apply"] == "reload" and row["writable"]
    assert row["error"] is None
    assert not builder.has('event_architecture')
    assert "private prompt" not in json.dumps(reg.catalog(ctx))
    assert edit(reg, ctx, True)["ok"]
    expected = original.copy()
    expected["event_architecture"]["enabled"] = True
    assert load(settings_file) == expected
    assert stat.S_IMODE(settings_file.stat().st_mode) == 0o640
    # The existing builder is not live-mutated by a reload-scoped setting.
    assert not builder.has('event_architecture')
    session = SimpleNamespace(state=State())
    fresh = PipelineBuilder(session)
    plugin.register_features(fresh, session=session, config={})
    assert fresh.has('event_architecture')
    assert reg.catalog(ctx)[0]["value"] is True
    assert list(settings_file.parent.iterdir()) == [settings_file]


@pytest.mark.parametrize("broken", ["[", "[]", "null", "", "{\"event_architecture\": []}"])
def test_bad_file_is_never_overwritten(settings_file, broken):
    reg, ctx, _ = registry()
    settings_file.write_text(broken)
    before = settings_file.read_bytes()
    assert reg.catalog(ctx)[0]["error"] is not None
    assert not edit(reg, ctx, True)["ok"]
    with pytest.raises(SettingError):
        plugin._save_enabled(True)
    assert settings_file.read_bytes() == before


def test_failed_replace_preserves_file_and_cleans_temp(settings_file, monkeypatch):
    reg, ctx, _ = registry()
    before = settings_file.read_bytes()
    def fail(*args):
        raise OSError("private path must not leak")
    monkeypatch.setattr(plugin.os, "replace", fail)
    result = edit(reg, ctx, True)
    assert not result["ok"]
    assert "private path" not in json.dumps(result)
    assert settings_file.read_bytes() == before
    assert list(settings_file.parent.iterdir()) == [settings_file]


def test_invalid_type_does_not_write(settings_file):
    reg, ctx, _ = registry()
    before = settings_file.read_bytes()
    assert not edit(reg, ctx, 1)["ok"]
    assert settings_file.read_bytes() == before


def test_unreadable_file_is_safe(settings_file, monkeypatch):
    reg, ctx, _ = registry()
    def fail(*args, **kwargs):
        raise PermissionError("private path")
    monkeypatch.setattr(Path, "read_text", fail)
    row, = reg.catalog(ctx)
    assert row["error"] is not None
    assert "private path" not in json.dumps(row)
    with pytest.raises(SettingError):
        plugin._save_enabled(True)


@pytest.mark.parametrize("override", [False, True])
def test_registration_override_does_not_replace_or_lock_saved_default(settings_file, override):
    saved = document()
    saved["event_architecture"]["enabled"] = not override
    settings_file.write_text(dump(saved))
    config = {"event_architecture": {"enabled": override}}
    reg, ctx, builder = registry(config)
    row, = reg.catalog(ctx)
    assert row["value"] is not override
    assert row["writable"] and row["kind"] == "bool"
    assert builder.has("event_architecture") is override
    assert edit(reg, ctx, override)["ok"]
    assert edit(reg, ctx, not override)["ok"]
    assert reg.catalog(ctx)[0]["value"] is not override
    assert load(settings_file)["event_architecture"]["enabled"] is not override
    assert config == {"event_architecture": {"enabled": override}}
    assert builder.has("event_architecture") is override
    fresh, fresh_ctx, fresh_builder = registry(config)
    assert fresh_builder.has("event_architecture") is override
    assert fresh.catalog(fresh_ctx)[0]["value"] is not override


def test_missing_file_preserves_absent_defaults(settings_file):
    settings_file.unlink()
    reg, ctx, _ = registry()
    assert not settings_file.exists()
    assert reg.catalog(ctx)[0]["value"] is False
    assert edit(reg, ctx, True)["ok"]
    assert load(settings_file) == {"event_architecture": {"enabled": True}}


@pytest.mark.parametrize("value", ["inherit", "enabled", "disabled"])
def test_old_choices_are_rejected(settings_file, value):
    reg, ctx, _ = registry()
    before = settings_file.read_bytes()
    assert not edit(reg, ctx, value)["ok"]
    with pytest.raises(SettingError):
        plugin._save_enabled(value)
    assert settings_file.read_bytes() == before


def test_fork_inheritance_is_not_the_saved_default(settings_file, monkeypatch):
    saved = document()
    saved["event_architecture"].pop("enabled")
    settings_file.write_text(dump(saved))
    before = settings_file.read_bytes()
    state = State()
    state._session_kind = "fork"
    state._parent_session_id = "parent"
    session = SimpleNamespace(state=state)
    monkeypatch.setenv(plugin.FORK_ENABLE_ENV, "parent")
    assert plugin._plugin_enabled({}, session)
    reg, ctx, builder = registry(session=session)
    assert builder.has("event_architecture")
    row, = reg.catalog(ctx)
    assert row["value"] is False and row["kind"] == "bool"
    assert not row["choices"]
    assert settings_file.read_bytes() == before
    assert edit(reg, ctx, False)["ok"]
    assert load(settings_file)["event_architecture"]["enabled"] is False
    assert reg.catalog(ctx)[0]["value"] is False
    # Saving defaults does not remove active fork tools or their inheritance marker.
    assert builder.has("event_architecture")
    assert plugin.os.environ[plugin.FORK_ENABLE_ENV] == "parent"
    # An explicit saved default is honored on a subsequent registration.
    _, _, fresh = registry(session=session)
    assert not fresh.has("event_architecture")
