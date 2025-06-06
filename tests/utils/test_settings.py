import os
import json
import pytest
from utils.settings import SettingsManager
from PyQt6.QtCore import QCoreApplication
import tempfile
import threading

@pytest.fixture
def app():
    return QCoreApplication([])

def test_load_existing_config(tmp_path, app):
    config = {"foo": "bar"}
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    sm = SettingsManager(str(config_path))
    assert sm.settings == config

def test_load_nonexistent_config(tmp_path, app):
    config_path = tmp_path / "does_not_exist.json"
    sm = SettingsManager(str(config_path))
    assert sm.settings == {}

def test_reload_emits_signal(tmp_path, qtbot, app):
    config = {"foo": "bar"}
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    sm = SettingsManager(str(config_path))
    with qtbot.waitSignal(sm.settings_reloaded, timeout=1000):
        sm.reload()

def test_load_malformed_json(tmp_path, app):
    config_path = tmp_path / "bad.json"
    with open(config_path, "w") as f:
        f.write("{bad json}")
    with pytest.raises(json.JSONDecodeError):
        sm = SettingsManager(str(config_path))

def test_reload_after_modification(tmp_path, app):
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump({"foo": "bar"}, f)
    sm = SettingsManager(str(config_path))
    with open(config_path, "w") as f:
        json.dump({"foo": "baz"}, f)
    sm.reload()
    assert sm.settings["foo"] == "baz"

def test_signal_emitted_on_missing_file(tmp_path, qtbot, app):
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump({"foo": "bar"}, f)
    sm = SettingsManager(str(config_path))
    os.remove(config_path)
    with qtbot.waitSignal(sm.settings_reloaded, timeout=1000):
        sm.reload()
    assert sm.settings == {}

def test_load_empty_file(tmp_path, app):
    config_path = tmp_path / "empty.json"
    with open(config_path, "w") as f:
        f.write("")
    with pytest.raises(json.JSONDecodeError):
        SettingsManager(str(config_path))

def test_load_partial_file(tmp_path, app):
    config_path = tmp_path / "partial.json"
    with open(config_path, "w") as f:
        f.write('{"foo":')
    with pytest.raises(json.JSONDecodeError):
        SettingsManager(str(config_path))

def test_settings_thread_safety(tmp_path, app):
    config_path = tmp_path / "threaded.json"
    with open(config_path, "w") as f:
        json.dump({"foo": 0}, f)
    sm = SettingsManager(str(config_path))
    def update_config():
        for i in range(50):
            with open(config_path, "w") as f:
                json.dump({"foo": i}, f)
            sm.reload()
    threads = [threading.Thread(target=update_config) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # La dernière valeur doit être cohérente
        sm = SettingsManager(str(config_path)) 