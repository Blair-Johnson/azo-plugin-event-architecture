"""Settings use temporary files and the real feature/pipeline registry."""
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utils import PipelineBuilder, State
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


def registry(config=None):
    session = SimpleNamespace(state=State())
    builder = PipelineBuilder(session)
    plugin.register_features(builder, session=session, config=config or {})
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
                           "key": "event_architecture.enablement", "value": value}, ctx)


def test_disabled_registration_round_trip(settings_file):
    original = load(settings_file)
    reg, ctx, builder = registry()
    row, = reg.catalog(ctx)
    assert row["value"] == "disabled"
    assert row["kind"] == "choice"
    assert row["apply"] == "reload" and row["writable"]
    assert row["error"] is None
    assert not builder.has('event_architecture')
    assert "private prompt" not in json.dumps(reg.catalog(ctx))
    assert edit(reg, ctx, "enabled")["ok"]
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
    assert reg.catalog(ctx)[0]["value"] == "enabled"
    assert list(settings_file.parent.iterdir()) == [settings_file]


@pytest.mark.parametrize("broken", ["[", "[]", "null", "", "{\"event_architecture\": []}"])
def test_bad_file_is_never_overwritten(settings_file, broken):
    reg, ctx, _ = registry()
    settings_file.write_text(broken)
    before = settings_file.read_bytes()
    assert reg.catalog(ctx)[0]["error"] is not None
    assert not edit(reg, ctx, "enabled")["ok"]
    with pytest.raises(SettingError):
        plugin._save_enabled("enabled")
    assert settings_file.read_bytes() == before


def test_failed_replace_preserves_file_and_cleans_temp(settings_file, monkeypatch):
    reg, ctx, _ = registry()
    before = settings_file.read_bytes()
    def fail(*args):
        raise OSError("private path must not leak")
    monkeypatch.setattr(plugin.os, "replace", fail)
    result = edit(reg, ctx, "enabled")
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
        plugin._save_enabled("enabled")


def test_registration_override_is_readonly(settings_file):
    reg, ctx, _ = registry({"event_architecture": {"enabled": False}})
    row, = reg.catalog(ctx)
    assert row["value"] == "disabled"
    assert not row["writable"]
    before = settings_file.read_bytes()
    assert edit(reg, ctx, "enabled")["error"]["code"] == "read_only"
    assert settings_file.read_bytes() == before


def test_missing_file_preserves_absent_defaults(settings_file):
    settings_file.unlink()
    reg, ctx, _ = registry()
    assert not settings_file.exists()
    assert edit(reg, ctx, "enabled")["ok"]
    assert load(settings_file) == {"event_architecture": {"enabled": True}}


def test_inherit_removes_only_enabled(settings_file):
    reg, ctx, _ = registry()
    expected = load(settings_file)
    expected["event_architecture"].pop("enabled")
    assert edit(reg, ctx, "inherit")["ok"]
    assert reg.catalog(ctx)[0]["value"] == "inherit"
    assert load(settings_file) == expected
