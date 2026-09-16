from pathlib import Path

from skrecsys.datasets import clear_data_home, get_data_home


def test_explicit_data_home_is_created(tmp_path):
    path = tmp_path / "a" / "b"
    assert get_data_home(path) == path
    assert path.is_dir()


def test_env_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("SKRECSYS_DATA", str(tmp_path / "env"))
    assert get_data_home() == tmp_path / "env"


def test_default_under_home(tmp_path, monkeypatch):
    monkeypatch.delenv("SKRECSYS_DATA", raising=False)
    # ``~`` is resolved from HOME on POSIX and from USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    assert get_data_home() == Path(tmp_path) / "skrecsys_data"


def test_clear_data_home(tmp_path):
    path = get_data_home(tmp_path / "home")
    (path / "file").write_text("x")
    clear_data_home(path)
    assert not path.exists()
