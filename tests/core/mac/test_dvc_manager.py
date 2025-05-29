import pytest
from unittest.mock import patch, MagicMock
from bitcoin_scalper.core.dvc_manager import DVCManager

@pytest.fixture
def dvc():
    return DVCManager(repo_path="/tmp")

@patch("bitcoin_scalper.core.dvc_manager.subprocess.run")
def test_init_success(mock_run, dvc):
    mock_run.return_value = MagicMock(returncode=0)
    assert dvc.init() is True

@patch("bitcoin_scalper.core.dvc_manager.subprocess.run")
def test_init_fail(mock_run, dvc):
    mock_run.return_value = MagicMock(returncode=1)
    assert dvc.init() is False

@patch("bitcoin_scalper.core.dvc_manager.subprocess.run")
def test_add(mock_run, dvc):
    mock_run.return_value = MagicMock(returncode=0)
    assert dvc.add("data/file.csv") is True

@patch("bitcoin_scalper.core.dvc_manager.subprocess.run")
def test_commit(mock_run, dvc):
    mock_run.return_value = MagicMock(returncode=0)
    assert dvc.commit() is True
    assert dvc.commit("data/file.csv") is True

@patch("bitcoin_scalper.core.dvc_manager.subprocess.run")
def test_push_pull(mock_run, dvc):
    mock_run.return_value = MagicMock(returncode=0)
    assert dvc.push() is True
    assert dvc.pull() is True

@patch("bitcoin_scalper.core.dvc_manager.subprocess.run")
def test_repro(mock_run, dvc):
    mock_run.return_value = MagicMock(returncode=0)
    assert dvc.repro() is True
    assert dvc.repro("target") is True

@patch("bitcoin_scalper.core.dvc_manager.subprocess.run")
def test_status_diff(mock_run, dvc):
    mock_run.return_value = MagicMock(stdout="ok", returncode=0)
    assert dvc.status() == "ok"
    assert dvc.diff() == "ok"

@patch("bitcoin_scalper.core.dvc_manager.subprocess.run")
def test_gc(mock_run, dvc):
    mock_run.return_value = MagicMock(returncode=0)
    assert dvc.gc() is True
    assert dvc.gc(workspace=True) is True

@patch("bitcoin_scalper.core.dvc_manager.subprocess.run")
def test_remote(mock_run, dvc):
    mock_run.return_value = MagicMock(returncode=0)
    assert dvc.remote("add", "origin", "s3://bucket") is True
    assert dvc.remote("remove", "origin") is True 