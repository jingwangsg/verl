import os
import re
import socket

import pytest

from debug.snapshot import Snapshot


class Dummy:
    def __init__(self, x, y=1):
        self.x = x
        self.y = y


@pytest.fixture
def enable_snapshot(monkeypatch):
    monkeypatch.setenv("SNAPSHOT", "1")
    yield
    monkeypatch.delenv("SNAPSHOT", raising=False)


def test_snapshot_object_generates_context_suffix_file(tmp_path, enable_snapshot):
    snap = Snapshot("obj", output_dir=tmp_path, once=False)
    obj = {"a": 1}

    result = snap.snapshot(obj)

    files = list(tmp_path.glob("obj_*.pkl"))
    assert result == obj
    assert len(files) == 1
    expected_suffix = snap._get_context_suffix()  # type: ignore[attr-defined]
    assert files[0].name == f"obj_{expected_suffix}.pkl"
    assert snap.restore() == obj


def test_snapshot_once_reuses_existing(tmp_path, enable_snapshot):
    snap = Snapshot("single", output_dir=tmp_path, once=True)

    first = snap.snapshot({"v": 1})
    second = snap.snapshot({"v": 2})

    assert first == {"v": 1}
    assert second == {"v": 2}
    assert tmp_path.joinpath("single.pkl").exists()
    assert len(list(tmp_path.glob("single*.pkl"))) == 1
    assert snap.restore() == {"v": 1}


def test_snapshot_class_and_args(tmp_path, enable_snapshot):
    snap = Snapshot("klass", output_dir=tmp_path, once=False)

    snap.snapshot_class(Dummy)
    snap.snapshot_args(3, y=4)

    restored = snap.restore()
    files = list(tmp_path.glob("klass_*.pkl"))

    assert isinstance(restored, Dummy)
    assert (restored.x, restored.y) == (3, 4)
    assert len(files) == 1
    expected_suffix = snap._get_context_suffix()  # type: ignore[attr-defined]
    assert files[0].name == f"klass_{expected_suffix}.pkl"
    snap_again = Snapshot("klass", output_dir=tmp_path, once=False)
    assert snap_again.restore_cls() is Dummy
    assert snap_again.restore_args() == ((3,), {"y": 4})


def test_snapshot_once_class_updates_args(tmp_path, enable_snapshot):
    snap = Snapshot("once_class", output_dir=tmp_path, once=True)
    snap.snapshot_class(Dummy)
    snap.snapshot_args(5, y=6)

    snap2 = Snapshot("once_class", output_dir=tmp_path, once=True)
    restored = snap2.restore()

    assert isinstance(restored, Dummy)
    assert (restored.x, restored.y) == (5, 6)


def test_snapshot_torch_module(tmp_path, enable_snapshot):
    torch = pytest.importorskip("torch")

    model = torch.nn.Linear(2, 2)
    snap = Snapshot("torchmod", output_dir=tmp_path, once=False)

    saved_model = snap.snapshot(model)
    restored = snap.restore()
    files = list(tmp_path.glob("torchmod_*.pkl"))

    assert isinstance(saved_model, torch.nn.Module)
    assert isinstance(restored, torch.nn.Module)
    assert len(files) == 1
    expected_suffix = snap._get_context_suffix()  # type: ignore[attr-defined]
    assert files[0].name == f"torchmod_{expected_suffix}.pkl"
    assert torch.allclose(model.weight, restored.weight)
    assert torch.allclose(model.bias, restored.bias)


def test_restore_class_args_on_object_snapshot(tmp_path, enable_snapshot):
    snap = Snapshot("objonly", output_dir=tmp_path, once=False)
    snap.snapshot({"v": 1})

    snap2 = Snapshot("objonly", output_dir=tmp_path, once=False)
    assert snap2.restore_cls() is None
    assert snap2.restore_args() == (None, None)


def test_restore_legacy_base_path(tmp_path):
    # simulate old snapshots saved without suffix even when once=False
    snap = Snapshot("legacy", output_dir=tmp_path, once=False)
    payload = {"kind": "object", "object": {"v": 1}}
    snap.base_dir.mkdir(parents=True, exist_ok=True)
    with open(snap.base_path, "wb") as f:
        import dill

        dill.dump(payload, f)

    snap2 = Snapshot("legacy", output_dir=tmp_path, once=False)
    restored = snap2.restore()

    assert restored == {"v": 1}


def test_snapshot_name_with_pkl_suffix(tmp_path, enable_snapshot):
    snap = Snapshot("withsuffix.pkl", output_dir=tmp_path, once=False)
    obj = {"v": 1}
    snap.snapshot(obj)

    expected_suffix = snap._get_context_suffix()  # type: ignore[attr-defined]
    expected_file = tmp_path / f"withsuffix_{expected_suffix}.pkl"
    assert expected_file.exists()
    snap2 = Snapshot("withsuffix.pkl", output_dir=tmp_path, once=False)
    assert snap2.restore() == obj


def test_list_snapshots_with_glob(tmp_path, enable_snapshot):
    snap_dir = tmp_path / "agent_loop" / "03c34b35"
    snap1 = Snapshot("agent_loop/03c34b35/a", output_dir=tmp_path, once=False)
    snap2 = Snapshot("agent_loop/03c34b35/b", output_dir=tmp_path, once=False)
    snap1.snapshot({"a": 1})
    snap2.snapshot({"b": 2})
    (snap_dir / ".hidden.pkl").write_text("hide")

    found = Snapshot.list_snapshot("agent_loop/03c34b35/*", output_dir=tmp_path)
    names = sorted(p.name for p in found)

    assert len(names) == 2
    assert ".hidden.pkl" not in names
    assert all(name.endswith(".pkl") for name in names)
    assert any(name.startswith("a_") for name in names)
    assert any(name.startswith("b_") for name in names)


def test_snapshot_disabled_skips_io(tmp_path, monkeypatch):
    # create one when enabled
    monkeypatch.setenv("SNAPSHOT", "1")
    snap_on = Snapshot("disabled", output_dir=tmp_path, once=False)
    obj = {"k": 1}
    snap_on.snapshot(obj)
    monkeypatch.setenv("SNAPSHOT", "0")

    snap_off = Snapshot("disabled", output_dir=tmp_path, once=False)
    result = snap_off.snapshot({"k": 2})  # should not overwrite
    restored = snap_off.restore()

    assert result == {"k": 2}
    assert restored == obj  # restore still works when disabled
    files = list(tmp_path.glob("disabled*.pkl"))
    assert len(files) == 1


def test_snapshot_nested_path_creates_dirs(tmp_path, enable_snapshot):
    snap = Snapshot("a/b/c", output_dir=tmp_path, once=False)
    obj = {"nested": True}

    result = snap.snapshot(obj)
    expected_dir = tmp_path / "a" / "b"
    expected_suffix = snap._get_context_suffix()  # type: ignore[attr-defined]
    expected_file = expected_dir / f"c_{expected_suffix}.pkl"

    assert result == obj
    assert expected_dir.is_dir()
    assert expected_file.exists()
    assert snap.restore() == obj


def test_snapshot_subsys_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAPSHOT", "1")
    monkeypatch.setenv("SNAPSHOT_SUBSYS", "alpha,beta")

    snap_allowed = Snapshot("allowed", output_dir=tmp_path, subsys="alpha")
    snap_blocked = Snapshot("blocked", output_dir=tmp_path, subsys="gamma")

    snap_allowed.snapshot({"v": 1})
    snap_blocked.snapshot({"v": 2})

    allowed_files = list(tmp_path.glob("allowed*.pkl"))
    blocked_files = list(tmp_path.glob("blocked*.pkl"))

    assert len(allowed_files) == 1
    assert len(blocked_files) == 0
    assert snap_allowed.restore() == {"v": 1}
