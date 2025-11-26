import copy
import fcntl
import inspect
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import dill

try:
    import torch
    from torch.nn import Module as TorchModule
except Exception:
    torch = None
    TorchModule = ()

try:
    import ray
except Exception:
    ray = None
try:
    from ray.runtime_context import RuntimeContext
except Exception:
    RuntimeContext = None


class Snapshot:
    def __init__(self, name, output_dir=None, once=None, subsys=None):
        env_output_dir = os.environ.get("SNAPSHOT_DIR")
        env_once = os.environ.get("SNAPSHOT_ONCE")
        env_enabled = os.environ.get("SNAPSHOT")
        env_subsys = os.environ.get("SNAPSHOT_SUBSYS")
        self.name = name
        self.output_dir = Path(output_dir or env_output_dir or "/tmp/snapshots")
        self.name_path = Path(self.name)
        if self.name_path.suffix == ".pkl":
            self.name_path = self.name_path.with_suffix("")
        self.base_dir = self.output_dir / self.name_path.parent
        self.base_filename = self.name_path.name
        self.subsys = subsys
        self.subsys_filter = (
            {part.strip() for part in env_subsys.split(",") if part.strip()}
            if env_subsys
            else None
        )
        self.enabled = self._as_bool(env_enabled)
        if self.enabled:
            os.makedirs(self.base_dir, exist_ok=True)
        self.once = self._as_bool(once if once is not None else env_once)
        self.base_path = self.base_dir / f"{self.base_filename}.pkl"
        self._klass = None
        self._args = ()
        self._kwargs = {}
        self._mode = None
        self._last_path = None
        self._lock_path = self.base_path.with_name(f".{self.base_path.name}.lock")
        self._suffix = None

    def _as_bool(self, value):
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "y", "on"}

    def _is_torch_module(self, obj):
        return torch is not None and isinstance(obj, TorchModule)

    def _ensure_mode(self, mode):
        if self._mode is not None and self._mode != mode:
            raise ValueError("snapshot() and snapshot_class/snapshot_args cannot be mixed")
        self._mode = mode

    def _save(self, payload, return_value=None, caller=None):
        if not self._is_dump_allowed():
            print(f"Snapshot disabled or filtered; skipping save for {self.name}")
            return return_value
        caller = caller or self._caller_info()
        target_path = self.base_path if self.once else self.base_dir / f"{self.base_filename}_{self._make_suffix()}.pkl"
        with self._file_lock():
            if self.once and self.base_path.exists():
                if payload.get("kind") == "class":
                    updated = self._maybe_update_class_payload(payload)
                    if updated:
                        print(f"Snapshot updated (once=True) at {self.base_path} with class args (caller: {caller})")
                    else:
                        print(f"Snapshot already exists, reusing {self.base_path} (caller: {caller})")
                    return return_value
                print(f"Snapshot already exists, reusing {self.base_path} (caller: {caller})")
                return return_value
            self._atomic_dump(target_path, payload)
            self._last_path = target_path
            print(f"Snapshot saved to {target_path} (caller: {caller})")
            return return_value

    def _save_class_payload(self, return_value=None, caller=None):
        if self._klass is None:
            return None
        payload = {
            "kind": "class",
            "class": self._klass,
            "args": self._args,
            "kwargs": self._kwargs,
        }
        return self._save(payload, return_value if return_value is not None else self._klass, caller=caller)

    def snapshot_class(self, klass):
        self._ensure_mode("class")
        self._klass = klass
        caller = self._caller_info()
        if not self.enabled:
            print(f"Snapshot disabled; skipping save for {self.name} (caller: {caller})")
            return klass
        return self._save_class_payload(return_value=klass, caller=caller)

    def snapshot_args(self, *args, **kwargs):
        self._ensure_mode("class")
        self._args = args
        self._kwargs = kwargs
        # Only persist once we know the class.
        caller = self._caller_info()
        if not self.enabled:
            print(f"Snapshot disabled; skipping save for {self.name} (caller: {caller})")
            return args, kwargs
        return self._save_class_payload(return_value=(args, kwargs), caller=caller) or (args, kwargs)

    def snapshot(self, obj):
        self._ensure_mode("object")
        caller = self._caller_info()
        if not self.enabled:
            print(f"Snapshot disabled; skipping save for {self.name} (caller: {caller})")
            return obj
        if self._is_torch_module(obj):
            obj = copy.deepcopy(obj)
            obj.cpu()  # move a copy to CPU so GPU state is not serialized
        payload = {"kind": "object", "object": obj}
        return self._save(payload, obj, caller=caller)

    def restore(self):
        data = self._load_latest()
        if data is None:
            return None

        if isinstance(data, dict) and data.get("kind") == "class":
            klass = data["class"]
            args = data.get("args", ())
            kwargs = data.get("kwargs", {})
            return klass(*args, **kwargs)

        if isinstance(data, dict) and "object" in data:
            return data["object"]

        return data

    def restore_cls(self):
        data = self._load_latest()
        if isinstance(data, dict) and data.get("kind") == "class":
            return data.get("class")
        return None

    def restore_args(self):
        data = self._load_latest()
        if isinstance(data, dict) and data.get("kind") == "class":
            return data.get("args", ()), data.get("kwargs", {})
        return None, None

    @classmethod
    def list_snapshots(cls, pattern="*.pkl", output_dir=None, include_hidden=False):
        env_output_dir = os.environ.get("SNAPSHOT_DIR")
        base_dir = Path(output_dir or env_output_dir or "/tmp/snapshots")
        pattern_path = Path(pattern)

        if pattern_path.is_absolute():
            search_path = pattern_path
        else:
            search_path = base_dir / pattern_path

        parent = search_path.parent
        glob_pat = search_path.name
        results = []
        for p in parent.glob(glob_pat):
            if not p.is_file():
                continue
            if not include_hidden and p.name.startswith("."):
                continue
            if p.suffix != ".pkl":
                continue
            results.append(p)
        return sorted(results, key=lambda p: p.name)

    @classmethod
    def list_snapshot(cls, pattern="*.pkl", output_dir=None, include_hidden=False):
        return cls.list_snapshots(pattern=pattern, output_dir=output_dir, include_hidden=include_hidden)

    def _resolve_restore_path(self):
        if self.once:
            return self.base_path
        if self._last_path and self._last_path.exists():
            return self._last_path

        candidates = []
        if self.base_path.exists():
            candidates.append(self.base_path)
        candidates.extend(self.base_dir.glob(f"{self.base_filename}_*.pkl"))
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
        return candidates[-1]

    @contextmanager
    def _file_lock(self):
        lock_file = open(self._lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    def _atomic_dump(self, path, payload):
        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
        with open(tmp_path, "wb") as f:
            dill.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def _load_path(self, path):
        with open(path, "rb") as f:
            return dill.load(f)

    def _load_latest(self):
        path = self._resolve_restore_path()
        if path is None or not path.exists():
            return None
        return self._load_path(path)

    def _maybe_update_class_payload(self, payload):
        try:
            existing = self._load_path(self.base_path)
        except Exception:
            existing = None
        if not isinstance(existing, dict) or existing.get("kind") != "class":
            return False
        new_payload = dict(existing)
        changed = False
        for key in ("class", "args", "kwargs"):
            if key in payload and payload[key]:
                if new_payload.get(key) != payload[key]:
                    new_payload[key] = payload[key]
                    changed = True
        if changed:
            self._atomic_dump(self.base_path, new_payload)
            self._last_path = self.base_path
        return changed

    def _make_suffix(self):
        return self._get_context_suffix()

    def _get_context_suffix(self):
        if self._suffix:
            return self._suffix

        suffix = None
        if ray is not None:
            try:
                is_init = getattr(ray, "is_initialized", lambda: False)()
                if is_init:
                    ctx = ray.get_runtime_context()
                    if isinstance(ctx, RuntimeContext) or ctx is not None:
                        actor_id = self._format_id(getattr(ctx, "get_actor_id", lambda: None)())
                        task_id = self._format_id(getattr(ctx, "get_task_id", lambda: None)())
                        if actor_id:
                            suffix = f"actor_{actor_id}"
                        elif task_id:
                            suffix = f"task_{task_id}"
            except Exception:
                suffix = None

        if suffix is None:
            suffix = f"{socket.gethostname()}_{os.getpid()}"

        self._suffix = self._sanitize_suffix(str(suffix))
        return self._suffix

    def _sanitize_suffix(self, value):
        return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)

    @staticmethod
    def _format_id(value):
        if not value:
            return None
        if hasattr(value, "hex"):
            try:
                return value.hex()
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def _caller_info():
        frame = inspect.currentframe()
        if not frame:
            return "unknown"
        caller = frame.f_back.f_back if frame.f_back and frame.f_back.f_back else None
        if not caller:
            return "unknown"
        try:
            return f"{caller.f_code.co_filename}:{caller.f_lineno} ({caller.f_code.co_name})"
        except Exception:
            return "unknown"

    def _is_dump_allowed(self):
        if not self.enabled:
            return False
        if self.subsys_filter is None:
            return True
        if not self.subsys:
            return False
        return self.subsys in self.subsys_filter
