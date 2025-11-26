"""Helper utilities for bad node detection."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
import subprocess
from typing import Dict, List, Set, Tuple

import ray
import ray.exceptions
from ray.train._internal.utils import get_address_and_port
from ray.train.torch.train_loop_utils import get_device
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_actors, list_tasks
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


INTERNAL_ACTOR_CLASSES = {
    "_StatsActor",
    "JobSupervisor",
    "ServeController",
    "ProxyActor",
}

INTERNAL_NAMESPACES = {
    "serve",
}


def is_ray_internal_actor(actor: Dict) -> bool:
    """Return True when the actor entry belongs to Ray internal services."""

    actor_name = actor.get("name", "")
    actor_class_name = actor.get("class_name", "")
    actor_namespace = actor.get("ray_namespace", "")

    if actor_name.startswith("_ray_internal_"):
        return True
    if actor_class_name in INTERNAL_ACTOR_CLASSES:
        return True
    if actor_namespace in INTERNAL_NAMESPACES:
        return True
    return False


class SimpleModel(nn.Module):
    """Tiny feedforward network used to validate GPU communication."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class NodeSpec:
    node_id: str
    ip: str
    hostname_hint: str
    gpu_count: int


class WorkloadInspector:
    """Collect and report Ray nodes that already execute workloads."""

    def __init__(self):
        self.busy_nodes: Set[str] = set()
        self.details: Dict[str, Dict] = {}

    def refresh(self) -> Set[str]:
        self.busy_nodes, self.details = self._collect()
        return self.busy_nodes

    def describe(self, node_id: str, node_ip: str):
        details = self.details.get(node_id)
        if not details:
            return

        actors = details.get("actors", [])
        tasks = details.get("tasks", [])

        print(f"  Skipping {node_ip} - has existing workloads")
        print(f"    Total: {len(actors)} actors, {len(tasks)} tasks")
        self._print_records("ACTORS", actors, self._format_actor)
        self._print_records("TASKS", tasks, self._format_task)

    def _collect(self) -> Tuple[Set[str], Dict[str, Dict[str, List[Dict]]]]:
        nodes_with_workloads: Set[str] = set()
        workload_details: Dict[str, Dict[str, List[Dict]]] = defaultdict(
            lambda: {"actors": [], "tasks": []}
        )

        try:
            actors = list_actors(filters=[("state", "=", "ALIVE")], detail=True)
            for actor in actors:
                if is_ray_internal_actor(actor):
                    continue
                node_id = actor.get("node_id")
                if not node_id:
                    continue
                nodes_with_workloads.add(node_id)
                workload_details[node_id]["actors"].append(
                    {
                        "actor_id": actor.get("actor_id", "unknown"),
                        "name": actor.get("name", "unnamed"),
                        "class_name": actor.get("class_name", "unknown"),
                        "state": actor.get("state", "unknown"),
                        "job_id": actor.get("job_id", "unknown"),
                        "pid": actor.get("pid", "N/A"),
                        "required_resources": actor.get("required_resources", {}),
                        "placement_group_id": actor.get("placement_group_id"),
                        "num_restarts": actor.get("num_restarts", 0),
                    }
                )
        except Exception as exc:
            print(f"Warning: Failed to get actor list: {exc}")

        try:
            tasks = list_tasks(filters=[("state", "=", "RUNNING")], detail=True)
            for task in tasks:
                node_id = task.get("node_id")
                if not node_id:
                    continue
                nodes_with_workloads.add(node_id)
                workload_details[node_id]["tasks"].append(
                    {
                        "task_id": task.get("task_id", "unknown"),
                        "name": task.get("name", "unnamed"),
                        "func_or_class_name": task.get("func_or_class_name", "unknown"),
                        "state": task.get("state", "unknown"),
                        "job_id": task.get("job_id", "unknown"),
                        "type": task.get("type", "unknown"),
                        "worker_pid": task.get("worker_pid", "N/A"),
                        "required_resources": task.get("required_resources", {}),
                        "placement_group_id": task.get("placement_group_id"),
                    }
                )
        except Exception as exc:
            print(f"Warning: Failed to get task list: {exc}")

        return nodes_with_workloads, workload_details

    def _print_records(self, label: str, records: List[Dict], formatter):
        if not records:
            return
        print(f"    {label} ({len(records)}):")
        for record in records[:5]:
            for line in formatter(record):
                print(f"      {line}")
        if len(records) > 5:
            print(f"      + {len(records) - 5} more {label.lower()}")

    @staticmethod
    def _short(value: str) -> str:
        if not value or value == "unknown":
            return "unknown"
        return f"{value[:8]}..."

    def _format_actor(self, actor: Dict) -> List[str]:
        lines = [
            f"- {actor.get('name', 'unnamed')} [class: {actor.get('class_name', 'unknown')}]",
            (
                "  ID: "
                f"{self._short(actor.get('actor_id'))}, Job: {self._short(actor.get('job_id'))},"
                f" PID: {actor.get('pid', 'N/A')}"
            ),
        ]
        resources = actor.get("required_resources") or {}
        if resources:
            res_str = ", ".join(f"{k}: {v}" for k, v in resources.items())
            lines.append(f"  Resources: {{{res_str}}}")
        pg_id = actor.get("placement_group_id")
        pg_str = self._short(pg_id) if pg_id else "None"
        lines.append(
            f"  Placement Group: {pg_str} | Restarts: {actor.get('num_restarts', 0)}"
        )
        return lines

    def _format_task(self, task: Dict) -> List[str]:
        lines = [
            (
                f"- {task.get('name', 'unnamed')} "
                f"[func: {task.get('func_or_class_name', 'unknown')}, type: {task.get('type', 'unknown')}]"
            ),
            (
                "  ID: "
                f"{self._short(task.get('task_id'))}, Job: {self._short(task.get('job_id'))},"
                f" Worker PID: {task.get('worker_pid', 'N/A')}"
            ),
        ]
        resources = task.get("required_resources") or {}
        if resources:
            res_str = ", ".join(f"{k}: {v}" for k, v in resources.items())
            lines.append(f"  Resources: {{{res_str}}}")
        pg_id = task.get("placement_group_id")
        pg_str = self._short(pg_id) if pg_id else "None"
        lines.append(f"  Placement Group: {pg_str}")
        return lines


@ray.remote
class TrainingWorker:
    """Distributed worker that runs the synthetic DDP workload."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.hostname = os.uname().nodename
        self.ip = ray.util.get_node_ip_address()
        self.sanity_report = self._run_basic_sanity_checks()

    def _run_basic_sanity_checks(self) -> Dict[str, str]:
        try:  # Local imports prevent driver dependency issues
            import numpy
            import transformers
        except Exception as exc:
            raise RuntimeError(f"Failed to import basic libs: {exc}") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        versions = {
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "transformers": transformers.__version__,
        }

        print(f"torch version: {versions['torch']}")
        print(f"numpy version: {versions['numpy']}")
        print(f"transformers version: {versions['transformers']}")
        return versions

    def get_info(self):
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "ip": self.ip,
            "sanity_check": self.sanity_report,
        }

    def get_address(self):
        return get_address_and_port()

    def setup_env_vars(self, addr, port, world_rank, world_size):
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(world_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["NCCL_DEBUG"] = "DEBUG"

    def train_ddp(self, epochs: int, timeout: float) -> Dict:
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(seconds=timeout),
            )
            gloo_group = dist.new_group(backend="gloo")

            x = torch.tensor(0, device=torch.device("cuda:0"))
            dist.all_reduce(x, group=gloo_group)

            device = get_device()
            model = SimpleModel().to(device)
            model = DDP(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            data = torch.randn(100, 10).to(device)
            targets = torch.randn(100, 1).to(device)
            dataset = TensorDataset(data, targets)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

            for epoch in range(epochs):
                sampler.set_epoch(epoch)
                total_loss = 0.0
                for batch_data, batch_targets in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)
                print(f"Worker {self.worker_id} Epoch {epoch}: Loss = {avg_loss:.4f}")

            dist.destroy_process_group()

            return {
                "status": "success",
                "worker_id": self.worker_id,
                "hostname": self.hostname,
                "final_loss": avg_loss,
                "epochs_completed": epochs,
            }

        except Exception as exc:  # pragma: no cover - heavily environment dependent
            print(f"Worker {self.worker_id} training failed: {exc}")
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as cleanup_exc:
                print(f"Failed to destroy process group: {cleanup_exc}")

            return {
                "status": "failed",
                "worker_id": self.worker_id,
                "hostname": self.hostname,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    def shutdown(self):
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
            time.sleep(0.1)
            return {"status": "success", "worker_id": self.worker_id}
        except Exception as exc:
            return {"status": "failed", "worker_id": self.worker_id, "error": str(exc)}


class WorkerPool:
    """Encapsulate worker lifecycle management for the detector."""

    def __init__(self, detector: "BadNodeDetector"):
        self.detector = detector
        self.timeout = detector.timeout
        self.workers: List = []
        self.worker_info: Dict[int, Dict] = {}
        self.physical_nodes: Dict[str, List[int]] = {}
        self.hostname_cache: Dict[str, str] = {}
        self._next_worker_id = 0

    def launch(self, node_specs: List[NodeSpec]) -> int:
        self.workers = []
        self.worker_info = {}
        self.physical_nodes = {}

        if not node_specs:
            print("No eligible nodes for worker creation.")
            return 0

        future_map = {}

        for spec in node_specs:
            if spec.node_id in self.detector.failed_node_ids:
                continue

            node_workers = []
            node_futures = []
            launch_failed = False

            for _ in range(spec.gpu_count):
                try:
                    worker = TrainingWorker.options(
                        num_gpus=1,
                        num_cpus=1,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=spec.node_id, soft=False
                        ),
                    ).remote(worker_id=self._next_worker_id)
                    node_workers.append(worker)
                    node_futures.append(worker.get_info.remote())
                    self._next_worker_id += 1
                except Exception as exc:
                    launch_failed = True
                    self.detector._mark_node_bad(
                        spec.node_id,
                        f"Failed to launch worker on {spec.hostname_hint or spec.ip}: {exc}",
                    )
                    break

            if launch_failed or not node_workers:
                for worker in node_workers:
                    self._kill_worker(worker)
                continue

            for worker, info_future in zip(node_workers, node_futures):
                future_map[info_future] = (worker, spec)

        if not future_map:
            print("No workers could be launched on selected nodes.")
            return 0

        pending_futures = set(future_map.keys())
        deadline = time.time() + self.timeout if self.timeout else None
        valid_workers: List = []
        worker_info: Dict[int, Dict] = {}
        physical_nodes: Dict[str, List[int]] = defaultdict(list)

        while pending_futures:
            timeout_left = None if deadline is None else max(0, deadline - time.time())
            if timeout_left == 0:
                break

            ready_futures, _ = ray.wait(
                list(pending_futures),
                num_returns=len(pending_futures),
                timeout=timeout_left,
            )

            if not ready_futures:
                break

            for future in ready_futures:
                pending_futures.discard(future)
                worker, spec = future_map[future]

                if spec.node_id in self.detector.failed_node_ids:
                    self._kill_worker(worker)
                    continue

                try:
                    info = ray.get(future)
                except Exception as exc:
                    self.detector._mark_node_bad(spec.node_id, f"Worker info failed: {exc}")
                    self._kill_worker(worker)
                    continue

                worker_id = len(valid_workers)
                valid_workers.append(worker)
                worker_info[worker_id] = info
                hostname = info["hostname"]

                self.detector.node_metadata.setdefault(spec.node_id, {})
                self.detector.node_metadata[spec.node_id].update(
                    {"hostname": hostname, "ip": info["ip"]}
                )
                physical_nodes[hostname].append(worker_id)

        if pending_futures:
            print(f"Timeout waiting for {len(pending_futures)} worker info responses")
            for future in list(pending_futures):
                worker, spec = future_map[future]
                self.detector._mark_node_bad(spec.node_id, "Worker info request timed out")
                self._kill_worker(worker)
                pending_futures.discard(future)

        self.workers = valid_workers
        self.worker_info = worker_info
        self.physical_nodes = dict(physical_nodes)

        if self.workers:
            print(
                f"\nSuccessfully initialized {len(self.workers)} workers "
                f"across {len(self.physical_nodes)} physical nodes:"
            )
            for hostname, worker_ids in sorted(self.physical_nodes.items()):
                worker_ip = self.worker_info.get(worker_ids[0], {}).get("ip", "unknown IP")
                print(f"  {hostname} ({worker_ip}): {len(worker_ids)} workers")
        else:
            print("Initialization sanity checks failed for all nodes.")

        return len(self.workers)

    def workers_for(self, hostname: str) -> List:
        worker_ids = self.physical_nodes.get(hostname, [])
        return [self.workers[i] for i in worker_ids]

    def cleanup(self, save_hostname_mapping: bool = False, gentle_timeout: float = 30.0):
        if save_hostname_mapping:
            self.hostname_cache = {}
            for hostname, worker_ids in self.physical_nodes.items():
                if worker_ids and worker_ids[0] in self.worker_info:
                    self.hostname_cache[hostname] = self.worker_info[worker_ids[0]]["ip"]

        if not self.workers:
            return

        print(
            f"Attempting gentle shutdown of {len(self.workers)} workers (timeout: {gentle_timeout}s)..."
        )

        shutdown_futures = []
        for worker in self.workers:
            try:
                shutdown_futures.append(worker.shutdown.remote())
            except Exception as exc:
                print(f"  ⚠ Failed to initiate gentle shutdown for a worker: {exc}")

        if shutdown_futures:
            try:
                results = ray.get(shutdown_futures, timeout=gentle_timeout)
                successful = sum(1 for r in results if r.get("status") == "success")
                print(f"  ✓ Gentle shutdown completed: {successful}/{len(results)} workers")
            except ray.exceptions.GetTimeoutError:
                print(f"  ⚠ Gentle shutdown timed out after {gentle_timeout}s, forcing kill...")
            except Exception as exc:
                print(f"  ⚠ Gentle shutdown failed: {exc}, forcing kill...")

        print("  Force killing all workers...")
        for worker in self.workers:
            self._kill_worker(worker)

        print("  Waiting for Ray state to update...")
        time.sleep(3)
        self.workers.clear()
        self.worker_info.clear()
        self.physical_nodes.clear()
        print("Workers cleaned up.")

    @staticmethod
    def _kill_worker(worker):
        try:
            ray.kill(worker)
        except Exception:
            pass


@ray.remote
def kill_processes_on_single_node(hostname: str, ip: str, node_id: str) -> Dict[str, str]:
    """Kill sleep processes on a single bad node (runs directly on that node)."""

    current_hostname = os.uname().nodename
    ray.util.get_node_ip_address()

    if current_hostname != hostname:
        return {
            "hostname": hostname,
            "ip": ip,
            "status": "error",
            "message": f"Scheduled on wrong node: {current_hostname} instead of {hostname}",
        }

    kill_command = (
        "ps aux | grep -E 'sleep|block' | grep -v grep | awk '{print $2}' | xargs kill -9"
    )

    try:
        subprocess.run(
            kill_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            "hostname": hostname,
            "ip": ip,
            "status": "success",
            "message": "Kill command executed (node may disconnect)",
        }
    except subprocess.TimeoutExpired:
        return {
            "hostname": hostname,
            "ip": ip,
            "status": "success",
            "message": "Kill command initiated (node disconnecting)",
        }
    except Exception as exc:
        return {
            "hostname": hostname,
            "ip": ip,
            "status": "error",
            "message": f"Kill command failed: {exc}",
        }


def kill_processes_on_bad_nodes(
    bad_nodes_list: List[List[str]],
    drain_callback,
) -> Dict[str, List[Dict[str, str]]]:
    """Run kill commands on provided nodes, draining on failure via callback."""

    filtered_bad_nodes: List[Tuple[str, str]] = []
    skipped_current_nodes: List[Tuple[str, str]] = []
    graceful_shutdowns: List[Tuple[str, str]] = []
    forced_drains: Dict[str, Dict[str, str]] = {}
    failed_to_schedule: List[Tuple[str, str]] = []

    def _build_kill_summary(processed_count: int) -> Dict[str, List[Dict[str, str]]]:
        return {
            "nodes_requested": len(bad_nodes_list),
            "nodes_scheduled": len(filtered_bad_nodes),
            "nodes_processed": processed_count,
            "graceful_kills": [
                {"hostname": host, "ip": ip} for host, ip in graceful_shutdowns
            ],
            "forced_drains": [
                {"hostname": host, "ip": info["ip"], "reason": info["reason"]}
                for host, info in forced_drains.items()
            ],
            "failed_to_schedule": [
                {"hostname": host, "ip": ip} for host, ip in failed_to_schedule
            ],
            "skipped_current_node": [
                {"hostname": host, "ip": ip} for host, ip in skipped_current_nodes
            ],
        }

    if not bad_nodes_list:
        print("No bad nodes to kill processes on.")
        return _build_kill_summary(0)

    print(f"\n🔥 Starting kill operations on {len(bad_nodes_list)} bad nodes...")
    print("⚠️  WARNING: Nodes will likely disconnect after kill command executes")

    try:
        result = subprocess.run("hostname -I", shell=True, capture_output=True, text=True)
        current_node_ips = set(result.stdout.strip().split())
        print(f"Current node IPs: {current_node_ips}")
    except Exception as exc:
        print(f"⚠️  Warning: Could not get current node IPs: {exc}")
        current_node_ips = set()

    for hostname, ip in bad_nodes_list:
        if ip in current_node_ips:
            skipped_current_nodes.append((hostname, ip))
        else:
            filtered_bad_nodes.append((hostname, ip))

    if skipped_current_nodes:
        print("\n⚠️  WARNING: Skipping kill on current node to prevent self-termination:")
        for hostname, ip in skipped_current_nodes:
            print(f"    - {hostname} ({ip}) - This is the current node!")

    if not filtered_bad_nodes:
        print("No nodes to kill after filtering out current node.")
        return _build_kill_summary(0)

    all_nodes = {node["NodeManagerAddress"]: node for node in ray.nodes()}
    alive_nodes = {
        node_ip: node_info["NodeID"]
        for node_ip, node_info in all_nodes.items()
        if node_info.get("Alive")
    }

    def _node_id_for_ip(ip_addr: str) -> str:
        node_entry = all_nodes.get(ip_addr)
        if node_entry:
            return node_entry.get("NodeID")
        return None

    def _record_drain(hostname: str, ip_addr: str, node_id: str, reason: str):
        if not node_id:
            return
        success = drain_callback(node_id, reason)
        if success:
            forced_drains[hostname] = {"ip": ip_addr, "reason": reason}

    kill_tasks = []
    future_map = {}

    for hostname, ip in filtered_bad_nodes:
        if ip in alive_nodes:
            node_id = alive_nodes[ip]
            print(f"  Scheduling kill task on {hostname} ({ip})")
            try:
                task = kill_processes_on_single_node.options(
                    num_cpus=0.1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                ).remote(hostname, ip, node_id)
                kill_tasks.append(task)
                future_map[task] = (hostname, ip)
            except Exception:
                failed_to_schedule.append((hostname, ip))
                _record_drain(hostname, ip, node_id, f"Failed to schedule kill task on {hostname} ({ip})")
        else:
            failed_to_schedule.append((hostname, ip))
            missing_node_id = _node_id_for_ip(ip)
            if missing_node_id:
                _record_drain(hostname, ip, missing_node_id, f"{hostname} ({ip}) missing from alive nodes during kill")

    total_killed = 0

    if not kill_tasks:
        print("No kill tasks could be scheduled.")

        if forced_drains:
            print("\n🛑 Nodes drained via GCS (kill not scheduled):")
            for host, info in forced_drains.items():
                print(f"   - {host} ({info['ip']}): {info['reason']}")
        return _build_kill_summary(total_killed)

    print(f"\nExecuting {len(kill_tasks)} kill operations...")
    print("Nodes will disconnect after kill command - this is expected\n")

    success_count = 0
    disconnected_count = 0

    kill_timeout = 10
    pending_tasks = set(kill_tasks)
    deadline = time.time() + kill_timeout

    while pending_tasks:
        timeout_left = max(0, deadline - time.time())
        if timeout_left == 0:
            break

        ready_tasks, _ = ray.wait(
            list(pending_tasks),
            num_returns=len(pending_tasks),
            timeout=timeout_left,
        )

        if not ready_tasks:
            break

        for task in ready_tasks:
            pending_tasks.discard(task)
            hostname, ip = future_map.get(task, ("unknown", "unknown"))

            try:
                result = ray.get(task, timeout=1)
            except Exception as exc:
                result = {"status": "error", "message": str(exc)}

            if result.get("status") == "success":
                success_count += 1
                graceful_shutdowns.append((hostname, ip))
            else:
                print(f"   ⚠️  Kill task failed on {hostname} ({ip}): {result.get('message')}")
                _record_drain(hostname, ip, alive_nodes.get(ip) or _node_id_for_ip(ip), result.get("message", "Kill task failed"))

    if pending_tasks:
        print(f"   ⚠️  Kill timeout for {len(pending_tasks)} nodes - draining them")
        for task in pending_tasks:
            hostname, ip = future_map.get(task, ("unknown", "unknown"))
            _record_drain(hostname, ip, alive_nodes.get(ip) or _node_id_for_ip(ip), "Kill task timed out")

    total_killed = success_count

    if graceful_shutdowns:
        print("\n✅ Kill command executed on:")
        for hostname, ip in graceful_shutdowns:
            print(f"   - {hostname} ({ip})")

    if forced_drains:
        print("\n🛑 Nodes forcibly drained via GCS:")
        for host, info in forced_drains.items():
            print(f"   - {host} ({info['ip']}): {info['reason']}")

    return _build_kill_summary(total_killed)
