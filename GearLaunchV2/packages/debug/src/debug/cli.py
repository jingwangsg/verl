import argparse
from pathlib import Path

from debug.snapshot import Snapshot


def _parse_snapshot_path(raw):
    path = Path(raw)
    base_dir = path.parent
    name = path.name
    return base_dir, name


def restore_cli():
    parser = argparse.ArgumentParser(description="Restore a snapshot and drop into ipdb.")
    parser.add_argument("snapshot", help="Path like dump_dir/dump_name[.pkl]")
    args = parser.parse_args()

    base_dir, name = _parse_snapshot_path(args.snapshot)
    snap = Snapshot(name, output_dir=base_dir)
    obj = snap.restore()
    print(f"Restored snapshot '{name}' from {base_dir}")
    _debug(obj)


def restore_cls_cli():
    parser = argparse.ArgumentParser(description="Restore only class from snapshot.")
    parser.add_argument("snapshot", help="Path like dump_dir/dump_name[.pkl]")
    args = parser.parse_args()

    base_dir, name = _parse_snapshot_path(args.snapshot)
    snap = Snapshot(name, output_dir=base_dir)
    cls = snap.restore_class()
    print(f"Restored class from '{name}' in {base_dir}: {cls}")
    _debug(cls)


def restore_args_cli():
    parser = argparse.ArgumentParser(description="Restore only args/kwargs from snapshot.")
    parser.add_argument("snapshot", help="Path like dump_dir/dump_name[.pkl]")
    args = parser.parse_args()

    base_dir, name = _parse_snapshot_path(args.snapshot)
    snap = Snapshot(name, output_dir=base_dir)
    args_val, kwargs_val = snap.restore_args()
    print(f"Restored args from '{name}' in {base_dir}: args={args_val}, kwargs={kwargs_val}")
    _debug({"args": args_val, "kwargs": kwargs_val})


def _debug(obj):
    try:
        import ipdb

        ipdb.set_trace()
    except Exception as exc:  # pragma: no cover
        print(f"ipdb not available ({exc}); printing object instead:")
        print(obj)


if __name__ == "__main__":  # pragma: no cover
    restore_cli()
