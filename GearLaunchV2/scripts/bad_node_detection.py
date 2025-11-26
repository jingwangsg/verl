#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from typing import Dict, List, Set, Tuple

import ray
import ray.exceptions
from bad_node_helpers import (
    NodeSpec,
    WorkloadInspector,
    WorkerPool,
    kill_processes_on_bad_nodes,
)


def _get_gcs_client():
    """Create (or reuse) a Ray GCS client for draining nodes."""
    global _GCS_CLIENT_SINGLETON

    if _GCS_CLIENT_SINGLETON is not None:
        return _GCS_CLIENT_SINGLETON

    if GcsClient is None:
        print("⚠️  GCS client is unavailable (Ray version may be too old). Cannot drain nodes.")
        return None

    try:
        gcs_address = ray.get_runtime_context().gcs_address
        _GCS_CLIENT_SINGLETON = GcsClient(address=gcs_address)
        print(f"Initialized GCS client for draining nodes (GCS address: {gcs_address})")
    except Exception as e:  # pragma: no cover - depends on runtime environment
        print(f"⚠️  Failed to initialize GCS client: {e}")
        _GCS_CLIENT_SINGLETON = None
    return _GCS_CLIENT_SINGLETON


def drain_ray_node(node_id: str, reason_message: str, deadline_seconds: int = 300) -> bool:
    """
    Drain a Ray node via the GCS so Ray stops scheduling tasks there.

    Args:
        node_id: Hex node ID to drain.
        reason_message: Human-readable reason for draining.
        deadline_seconds: Informational deadline for shutdown.
    """
    if not node_id:
        print(f"⚠️  Cannot drain node without node_id. Reason: {reason_message}")
        return False

    if autoscaler_pb2 is None:
        print(f"⚠️  Drain API not available. Skipping drain for node {node_id}")
        return False

    gcs_client = _get_gcs_client()
    if gcs_client is None:
        print(f"⚠️  GCS client unavailable. Unable to drain node {node_id}")
        return False

    deadline_ms = int(time.time() * 1000) + deadline_seconds * 1000
    reason = autoscaler_pb2.DrainNodeReason.DRAIN_NODE_REASON_PREEMPTION

    try:
        accepted, rejection_reason = gcs_client.drain_node(
            node_id,
            reason,
            reason_message,
            deadline_ms,
        )
        node_short = node_id[:8]
        if accepted:
            print(f"  🌀 Drain request accepted for node {node_short} ({reason_message})")
            return True
        else:
            print(
                f"⚠️  Drain request rejected for node {node_short}: "
                f"{rejection_reason or 'Unknown reason'}"
            )
    except Exception as e:  # pragma: no cover - depends on runtime environment
        print(f"⚠️  Failed to drain node {node_id}: {e}")
    return False

try:
    from ray._raylet import GcsClient
    from ray.core.generated import autoscaler_pb2
except ImportError:  # pragma: no cover - dependent on Ray version
    GcsClient = None
    autoscaler_pb2 = None


_GCS_CLIENT_SINGLETON = None


class BadNodeDetector:
    def __init__(self, epochs: int = 2, timeout: float = 20.0):
        self.epochs = epochs
        self.timeout = timeout
        self.worker_pool = WorkerPool(self)
        self.workloads = WorkloadInspector()
        self.good_nodes = set()
        self.bad_nodes = set()
        self.pair_results = {}
        self.total_physical_nodes = 0
        self.node_metadata: Dict[str, Dict[str, str]] = {}  # Ray node metadata cache
        self.failed_node_ids: Set[str] = set()  # NodeIDs flagged as bad during init
        self.initialization_failures: Dict[str, Dict[str, str]] = {}  # hostname -> info

    @staticmethod
    def _safe_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

    def _gpu_capacity(self, node: Dict) -> int:
        """Best-effort GPU count for a Ray node."""
        resources = node.get("Resources") or node.get("TotalResources") or {}
        for key in ("GPU", "gpu"):
            if key in resources:
                return self._safe_int(resources[key])

        for key, value in resources.items():
            if key.lower() == "nvidia.com/gpu":
                return self._safe_int(value)

        return 0

    def _resolve_ip(self, hostname: str) -> str:
        """Best effort hostname → IP lookup using cached metadata."""
        worker_ids = self.worker_pool.physical_nodes.get(hostname, [])
        if worker_ids and worker_ids[0] in self.worker_pool.worker_info:
            return self.worker_pool.worker_info[worker_ids[0]]["ip"]

        hostname_map = getattr(self.worker_pool, "hostname_cache", {})
        if hostname in hostname_map:
            return hostname_map[hostname]

        for metadata in self.node_metadata.values():
            if metadata.get("hostname") == hostname or metadata.get("hostname_hint") == hostname:
                return metadata.get("ip", "unknown IP")

        return "unknown IP"

    def _create_workers_on_nodes(self, node_specs: List[NodeSpec]) -> int:
        """Launch workers on provided nodes and run sanity checks."""
        return self.worker_pool.launch(node_specs)

    def _mark_node_bad(self, node_id: str, reason: str):
        """Record a node as bad during initialization to skip further work on it."""
        if not node_id or node_id in self.failed_node_ids:
            return

        meta = self.node_metadata.get(node_id, {})
        hostname = (
            meta.get("hostname")
            or meta.get("hostname_hint")
            or meta.get("ip")
            or f"node-{node_id[:6]}"
        )
        ip = meta.get("ip", "unknown")

        self.failed_node_ids.add(node_id)
        self.initialization_failures[hostname] = {
            "ip": ip,
            "reason": reason,
            "node_id": node_id,
        }
        self.bad_nodes.add(hostname)
        self.worker_pool.physical_nodes.pop(hostname, None)
        print(f"  ⚠️ Marking node {hostname} ({ip}) as BAD: {reason}")

    def initialize_workers(self):
        """Create Ray workers in proportion to each node's GPU capacity."""

        busy_nodes = self.workloads.refresh()
        if busy_nodes:
            print(f"Found {len(busy_nodes)} nodes with existing workloads - skipping these nodes")

        alive_nodes = [node for node in ray.nodes() if node["Alive"]]
        print(f"Found {len(alive_nodes)} alive nodes in cluster")

        node_specs: List[NodeSpec] = []
        skipped_for_gpu: List[str] = []

        for node in alive_nodes:
            node_id = node["NodeID"]
            node_ip = node["NodeManagerAddress"]
            hostname_hint = (
                node.get("NodeManagerHostname")
                or node.get("NodeName")
                or node["NodeManagerAddress"]
            )

            gpu_count = self._gpu_capacity(node)
            self.node_metadata[node_id] = {
                "hostname": hostname_hint,
                "hostname_hint": hostname_hint,
                "ip": node_ip,
            }

            if node_id in busy_nodes:
                self.workloads.describe(node_id, node_ip)
                continue

            if gpu_count < 1:
                skipped_for_gpu.append(hostname_hint or node_ip)
                continue

            node_specs.append(
                NodeSpec(node_id=node_id, ip=node_ip, hostname_hint=hostname_hint, gpu_count=gpu_count)
            )

        if skipped_for_gpu:
            print(
                f"Skipping {len(skipped_for_gpu)} nodes without reported GPUs: "
                f"{', '.join(skipped_for_gpu)}"
            )

        print(f"Will use {len(node_specs)} nodes for testing")
        for spec in node_specs:
            print(f"  {spec.hostname_hint} ({spec.ip}): {spec.gpu_count} GPUs detected")

        self.total_physical_nodes = len(node_specs)
        created = self._create_workers_on_nodes(node_specs)

        return created

    def _setup_and_run_ddp_training(self, workers: List) -> Tuple[bool, List[Dict]]:
        """
        Setup distributed environment and run DDP training
        
        Args:
            workers: List of worker actors
        
        Returns:
            Tuple of (success, results)
        """
        world_size = len(workers)
        
        try:
            # Get master address (use first worker)
            head_node_addr, port = ray.get(workers[0].get_address.remote())
            print(f"  Master address: {head_node_addr}:{port}")
            
            # Setup environment for all workers
            setup_futures = [
                worker.setup_env_vars.remote(
                    addr=head_node_addr, 
                    port=port, 
                    world_rank=rank, 
                    world_size=world_size
                )
                for rank, worker in enumerate(workers)
            ]
            ray.get(setup_futures, timeout=10)
            
            # Start DDP training on all workers
            train_futures = [
                worker.train_ddp.remote(self.epochs, self.timeout)
                for worker in workers
            ]
            
            # Wait for training completion
            results = []
            for future in train_futures:
                try:
                    result = ray.get(future, timeout=self.timeout + 10)
                    results.append(result)
                except ray.exceptions.RayActorError as e:
                    print(f"Worker failed: {str(e)}")
                    results.append({"status": "failed", "error": str(e)})
            
            success = all(r["status"] == "success" for r in results)
            return success, results
            
        except Exception as e:
            print(f"❌ DDP training setup failed: {str(e)}")
            return False, []

    def run_node_pair_training(self, hostname1: str, hostname2: str) -> Dict:
        """Run DDP training between two physical nodes (8 GPUs total)"""
        print(f"\n🔄 Starting node pair training: {hostname1} ↔ {hostname2}")

        # Get workers for both nodes
        workers1 = self.worker_pool.workers_for(hostname1)
        workers2 = self.worker_pool.workers_for(hostname2)

        all_workers = workers1 + workers2
        world_size = len(all_workers)

        print(f"  Node 1 ({hostname1}): {len(workers1)} workers")
        print(f"  Node 2 ({hostname2}): {len(workers2)} workers")
        print(f"  Total world size: {world_size}")

        try:
            # Run DDP training
            success, results = self._setup_and_run_ddp_training(all_workers)

            result = {
                "success": success,
                "hostname1": hostname1,
                "hostname2": hostname2,
                "results": results,
                "world_size": world_size,
            }

            if success:
                print(f"✅ Node pair ({hostname1},{hostname2}) completed successfully")
            else:
                print(f"❌ Node pair ({hostname1},{hostname2}) failed")
                for r in results:
                    if r["status"] != "success":
                        print(
                            f"    Worker {r['worker_id']}: {r['status']} - {r.get('error', 'Unknown error')}"
                        )

            return result

        except ray.exceptions.RayTaskError as e:
            print(f"❌ Node pair ({hostname1},{hostname2}) failed with Ray task error: {str(e)}")
            return {
                "success": False,
                "hostname1": hostname1,
                "hostname2": hostname2,
                "error": f"RayTaskError: {str(e)}",
                "error_type": "ray_task_error",
            }
        except ray.exceptions.RayActorError as e:
            print(f"❌ Node pair ({hostname1},{hostname2}) failed with Ray actor error: {str(e)}")
            return {
                "success": False,
                "hostname1": hostname1,
                "hostname2": hostname2,
                "error": f"RayActorError: {str(e)}",
                "error_type": "ray_actor_error",
            }
        except Exception as e:
            print(f"❌ Node pair ({hostname1},{hostname2}) failed with exception: {str(e)}")
            return {
                "success": False,
                "hostname1": hostname1,
                "hostname2": hostname2,
                "error": str(e),
                "error_type": "general_exception",
            }

    def _cleanup_workers(self, save_hostname_mapping: bool = False, gentle_timeout: float = 30.0):
        """Kill all workers and clear state with gentle shutdown."""
        self.worker_pool.cleanup(
            save_hostname_mapping=save_hostname_mapping, gentle_timeout=gentle_timeout
        )

    def _run_parallel_node_pair_tests(
        self,
        node_pairs: List[Tuple[str, str]],
        max_workers: int = 8,
    ) -> Tuple[Set[str], Set[str], Dict[Tuple[str, str], Dict]]:
        """
        Run parallel tests on node pairs and collect results using driver-side threads.
        """
        if not node_pairs:
            return set(), set(), {}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        successful_nodes: Set[str] = set()
        failed_nodes: Set[str] = set()
        pair_results: Dict[Tuple[str, str], Dict] = {}

        max_workers = max(1, min(max_workers, len(node_pairs)))
        print(f"\nScheduling {len(node_pairs)} node pairs (max concurrency: {max_workers})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self.run_node_pair_training, host1, host2): (host1, host2)
                for host1, host2 in node_pairs
            }

            for future in as_completed(future_map):
                hostname1, hostname2 = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"❌ Node pair ({hostname1},{hostname2}) raised exception: {exc}")
                    result = {
                        "success": False,
                        "hostname1": hostname1,
                        "hostname2": hostname2,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }

                pair_results[(hostname1, hostname2)] = result

                if result.get("success"):
                    successful_nodes.add(hostname1)
                    successful_nodes.add(hostname2)
                else:
                    failed_nodes.add(hostname1)
                    failed_nodes.add(hostname2)

        return successful_nodes, failed_nodes, pair_results

    def run_first_round(self) -> Tuple[Set[str], Set[str]]:
        """First round: Test non-overlapping physical node pairs in parallel"""
        print("\n" + "=" * 80)
        print("ROUND 1: Testing non-overlapping physical node pairs in parallel")
        print("=" * 80)

        # Get list of physical node hostnames
        hostnames = list(self.worker_pool.physical_nodes.keys())

        # Generate non-overlapping node pairs
        node_pairs = [(hostnames[i], hostnames[i + 1]) for i in range(0, len(hostnames) - 1, 2)]

        print(f"Physical nodes: {len(hostnames)}")
        print(f"Node pairs to test: {node_pairs}")
        print(f"Total pairs: {len(node_pairs)}")

        if not node_pairs:
            print("Not enough physical nodes for pairwise testing")
            return set(), set()

        # Run all node pairs in parallel
        print(f"\nStarting parallel testing of {len(node_pairs)} node pairs...")

        successful_nodes, failed_nodes, new_pair_results = self._run_parallel_node_pair_tests(node_pairs)
        self.pair_results.update(new_pair_results)

        # Handle odd number of nodes
        if len(hostnames) % 2 == 1:
            odd_hostname = hostnames[-1]
            print(f"\nOdd node {odd_hostname} will be tested in round 2")
            failed_nodes.add(odd_hostname)

        print(f"\n{'='*60}")
        print("Round 1 Summary:")
        print(f"  Successful nodes: {sorted(successful_nodes)}")
        print(f"  Failed nodes: {sorted(failed_nodes)}")

        # Kill all workers after round 1 to ensure clean state for round 2
        print("\nCleaning up Round 1 workers...")
        self._cleanup_workers(save_hostname_mapping=True)

        return successful_nodes, failed_nodes

    def create_workers_for_nodes(self, hostnames: Set[str]):
        """Create new workers for specified nodes (used in round 2)"""
        print(f"Creating fresh workers for {len(hostnames)} nodes...")

        if not getattr(self.worker_pool, "hostname_cache", None):
            print("Error: No hostname->IP mapping available!")
            return

        busy_nodes = self.workloads.refresh()
        if busy_nodes:
            print(f"Found {len(busy_nodes)} nodes with existing workloads - will skip these nodes")

        alive_nodes = {
            node["NodeManagerAddress"]: node for node in ray.nodes() if node["Alive"]
        }
        node_specs: List[NodeSpec] = []
        skipped_hostnames: List[str] = []

        for hostname in hostnames:
            if hostname not in self.worker_pool.hostname_cache:
                print(f"  Warning: No IP mapping found for hostname {hostname}")
                continue

            node_ip = self.worker_pool.hostname_cache[hostname]
            node = alive_nodes.get(node_ip)
            if not node:
                print(f"  Warning: Node {hostname} ({node_ip}) not found in alive nodes")
                continue

            node_id = node["NodeID"]
            if node_id in busy_nodes:
                self.workloads.describe(node_id, node_ip)
                skipped_hostnames.append(hostname)
                continue

            gpu_count = self._gpu_capacity(node)
            if gpu_count < 1:
                print(f"  Warning: Node {hostname} ({node_ip}) reports 0 GPUs - skipping")
                continue

            node_specs.append(
                NodeSpec(node_id=node_id, ip=node_ip, hostname_hint=hostname, gpu_count=gpu_count)
            )
            self.node_metadata[node_id] = {
                "hostname": hostname,
                "hostname_hint": hostname,
                "ip": node_ip,
            }

        if skipped_hostnames:
            print(
                f"Skipped {len(skipped_hostnames)} nodes with existing workloads: "
                f"{', '.join(skipped_hostnames)}"
            )

        self._create_workers_on_nodes(node_specs)

    def run_second_round(self, failed_hostnames: Set[str], successful_hostnames: Set[str]):
        """Second round: Test failed nodes with good nodes to isolate bad ones"""
        if not failed_hostnames or not successful_hostnames:
            print("Skipping Round 2: Need both failed and successful nodes")
            return

        print("\n" + "=" * 80)
        print("ROUND 2: Testing failed nodes with good nodes")
        print("=" * 80)

        # Create fresh workers for all nodes that will participate in round 2
        all_round2_nodes = failed_hostnames | successful_hostnames
        self.create_workers_for_nodes(all_round2_nodes)

        # Filter to only include nodes with successfully created workers
        # (some may have been skipped due to existing workloads)
        remaining_failed_hostnames = [
            h for h in failed_hostnames if h in self.worker_pool.physical_nodes
        ]
        available_good_hostnames = [
            h for h in successful_hostnames if h in self.worker_pool.physical_nodes
        ]

        if not available_good_hostnames:
            print("No good nodes available for Round 2 testing")
            self.bad_nodes.update(remaining_failed_hostnames)
            return

        print(
            f"Testing {len(remaining_failed_hostnames)} failed nodes"
            f"with {len(available_good_hostnames)} good nodes"
        )

        # Determine pairing strategy based on the number of failed vs good nodes
        if len(remaining_failed_hostnames) <= len(available_good_hostnames):
            # Case 1: More good nodes than bad nodes - one-to-one pairing
            print("Strategy: One-to-one pairing (each failed node gets a unique good node)")

            # Build node pairs for testing
            node_pairs = [
                (remaining_failed_hostnames[i], available_good_hostnames[i])
                for i in range(len(remaining_failed_hostnames))
            ]

            # Run parallel tests
            _, _, new_pair_results = self._run_parallel_node_pair_tests(node_pairs)
            self.pair_results.update(new_pair_results)

            # Process results to classify failed nodes as good or bad
            for failed_hostname, good_hostname in node_pairs:
                pair_key = (failed_hostname, good_hostname)
                if pair_key in new_pair_results:
                    result = new_pair_results[pair_key]
                    if result["success"]:
                        print(f"✅ {failed_hostname} succeeded with {good_hostname} - marking as GOOD")
                        self.good_nodes.add(failed_hostname)
                    else:
                        print(f"❌ {failed_hostname} failed with {good_hostname} - marking as BAD")
                        self.bad_nodes.add(failed_hostname)
                else:
                    # If no result, mark as bad
                    self.bad_nodes.add(failed_hostname)

        else:
            # Case 2: More bad nodes than good nodes - batch processing
            batch_size = len(available_good_hostnames)
            num_batches = (len(remaining_failed_hostnames) + batch_size - 1) // batch_size
            print(f"Strategy: Batch processing ({num_batches} batches, batch size = {batch_size})")

            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(remaining_failed_hostnames))
                batch_failed_nodes = remaining_failed_hostnames[start_idx:end_idx]

                print(
                    f"\n📦 Processing batch {batch_num + 1}/{num_batches}"
                    f"({len(batch_failed_nodes)} failed nodes)..."
                )

                # Build node pairs for this batch
                batch_node_pairs = [
                    (batch_failed_nodes[i], available_good_hostnames[i % len(available_good_hostnames)])
                    for i in range(len(batch_failed_nodes))
                ]

                # Run parallel tests for this batch
                _, _, batch_pair_results = self._run_parallel_node_pair_tests(batch_node_pairs)
                self.pair_results.update(batch_pair_results)

                # Process results to classify failed nodes as good or bad
                for failed_hostname, good_hostname in batch_node_pairs:
                    pair_key = (failed_hostname, good_hostname)
                    if pair_key in batch_pair_results:
                        result = batch_pair_results[pair_key]
                        if result["success"]:
                            print(f"✅ {failed_hostname} succeeded with {good_hostname} - marking as GOOD")
                            self.good_nodes.add(failed_hostname)
                        else:
                            print(f"❌ {failed_hostname} failed with {good_hostname} - marking as BAD")
                            self.bad_nodes.add(failed_hostname)
                    else:
                        # If no result, mark as bad
                        self.bad_nodes.add(failed_hostname)

        # Add the remaining successful nodes to good nodes
        self.good_nodes.update(successful_hostnames)

    def analyze_results(self):
        """Analyze and report final results"""
        print("\n" + "=" * 80)
        print("FINAL ANALYSIS: Bad Node Detection Results")
        print("=" * 80)

        print("\n📊 Node Classification:")
        print(f"  Total physical nodes tested: {self.total_physical_nodes}")
        print(f"  Good nodes: {len(self.good_nodes)}")
        print(f"  Bad nodes: {len(self.bad_nodes)}")

        if self.good_nodes:
            print("\n✅ GOOD NODES (can communicate normally):")
            good_list = []
            for hostname in sorted(self.good_nodes):
                ip = self._resolve_ip(hostname)
                good_list.append(f"{hostname} ({ip})")
            print(f"  {', '.join(good_list)}")

        if self.bad_nodes:
            print("\n❌ BAD NODES (communication issues detected):")
            bad_list = []
            for hostname in sorted(self.bad_nodes):
                ip = self._resolve_ip(hostname)
                if hostname in self.initialization_failures:
                    ip = self.initialization_failures[hostname].get("ip", ip)
                bad_list.append(f"{hostname} ({ip})")
            print(f"  {', '.join(bad_list)}")

        # Prepare results for JSON output
        good_nodes_list = []
        bad_nodes_list = []

        for hostname in sorted(self.good_nodes):
            good_nodes_list.append([hostname, self._resolve_ip(hostname)])

        for hostname in sorted(self.bad_nodes):
            if hostname in self.initialization_failures:
                ip = self.initialization_failures[hostname].get("ip", self._resolve_ip(hostname))
            else:
                ip = self._resolve_ip(hostname)
            bad_nodes_list.append([hostname, ip])

        results = {
            "good_nodes": good_nodes_list,
            "bad_nodes": bad_nodes_list,
            "total_nodes": self.total_physical_nodes,
            "summary": {"good_count": len(self.good_nodes), "bad_count": len(self.bad_nodes)},
        }

        return results

    def run_single_node_test(self, hostname: str) -> bool:
        """Test GPU-to-GPU communication within a single node"""
        print(f"\n🔄 Starting single-node GPU communication test: {hostname}")

        # Get all workers for this node
        workers = self.worker_pool.workers_for(hostname)
        world_size = len(workers)

        print(f"  Node: {hostname}")
        print(f"  GPUs to test: {world_size}")

        if world_size < 2:
            print(f"  ⚠️ Need at least 2 GPUs, found {world_size}")
            return False

        try:
            # Run DDP training
            success, results = self._setup_and_run_ddp_training(workers)

            if success:
                print(f"✅ Single-node test on {hostname} completed successfully")
            else:
                print(f"❌ Single-node test on {hostname} failed")
                for r in results:
                    if r["status"] != "success":
                        print(f"    Worker {r['worker_id']}: {r.get('error', 'Unknown error')}")

            return success

        except Exception as e:
            print(f"❌ Single-node test on {hostname} failed with exception: {str(e)}")
            return False

    def run(self):
        """Main detection workflow"""
        print("\n" + "=" * 80)
        print("Starting Bad Node Detection")
        print(f"Configuration: {self.epochs} epochs, {self.timeout}s timeout")
        print("=" * 80)

        # Initialize workers
        self.initialize_workers()

        node_count = len(self.worker_pool.physical_nodes)
        initial_hostnames = set(self.worker_pool.physical_nodes.keys())

        if node_count == 1:
            # Single node mode: Test GPU-to-GPU communication within the node
            print("\n🔍 Single node detected - testing intra-node GPU communication")
            hostname = next(iter(initial_hostnames))

            success = self.run_single_node_test(hostname)

            if success:
                print(f"\n✅ Node {hostname} passed intra-node GPU communication test")
                self.good_nodes.add(hostname)
            else:
                print(f"\n❌ Node {hostname} failed intra-node GPU communication test")
                self.bad_nodes.add(hostname)

        elif node_count == 0:
            print("❌ Error: No physical nodes available for testing")
            self.bad_nodes.update(initial_hostnames)
        else:
            # Multi-node mode: Test inter-node communication
            print("\n🔍 Multiple nodes detected - testing inter-node communication")
            # Run first round
            successful_nodes, failed_nodes = self.run_first_round()

            # Run second round if needed
            if failed_nodes and successful_nodes:
                self.run_second_round(failed_nodes, successful_nodes)
            elif not successful_nodes:
                # All nodes failed in round 1
                print("All nodes failed in Round 1")
                self.bad_nodes.update(initial_hostnames)
            else:
                # All nodes succeeded in round 1
                self.good_nodes = successful_nodes

        # Final analysis
        return self.analyze_results()
def main():
    parser = argparse.ArgumentParser(description="Detect bad nodes through pairwise DDP training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs per test")
    parser.add_argument(
        "--timeout", type=float, default=20.0, help="Timeout in seconds for each pairwise test"
    )
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address")
    parser.add_argument(
        "--kill", action="store_true", help="Kill sleep processes on detected bad nodes via SSH"
    )
    args = parser.parse_args()

    if os.environ.get("WORKFLOW_ID"):
        workflow_id = os.environ.get("WORKFLOW_ID")
    else:
        workflow_id = None

    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    # Auto-detect number of alive nodes in the cluster
    alive_nodes = [node for node in ray.nodes() if node["Alive"]]
    num_physical_nodes = len(alive_nodes)

    print(f"Ray initialized. Detected {num_physical_nodes} alive physical nodes in cluster.")

    # Run bad node detection with error handling
    detector = BadNodeDetector(epochs=args.epochs, timeout=args.timeout)

    results = None
    try:
        results = detector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user. Saving partial results...")
        results = detector.analyze_results()
    except Exception as e:
        print(f"\n\n❌ Script failed with error: {str(e)}")
        print("Saving partial results...")
        import traceback

        traceback.print_exc()
        try:
            results = detector.analyze_results()
        except Exception as e:
            # Fallback minimal results
            results = {
                "good_nodes": (
                    sorted(detector.good_nodes) if hasattr(detector, "good_nodes") else []
                ),
                "bad_nodes": sorted(detector.bad_nodes) if hasattr(detector, "bad_nodes") else [],
                "total_nodes": detector.total_physical_nodes,
                "summary": {
                    "good_count": (
                        len(detector.good_nodes) if hasattr(detector, "good_nodes") else 0
                    ),
                    "bad_count": len(detector.bad_nodes) if hasattr(detector, "bad_nodes") else 0,
                },
            }
    finally:
        if results:
            print("\n🎯 Final Summary:")
            print(f"   Good nodes: {results['summary']['good_count']}")
            print(f"   Bad nodes: {results['summary']['bad_count']}")
            print(f"   Total nodes: {results['total_nodes']}")

            # Query workflow info
            # Direct command without user
            if workflow_id:
                print("Reporting the bad nodes to the workflow...")

                query_cmd = f"/osmo/usr/bin/osmo workflow query -t json {workflow_id}"
                query_result = subprocess.run(query_cmd, shell=True, capture_output=True, text=True)

                try:
                    query_result = json.loads(query_result.stdout)
                except json.JSONDecodeError:
                    print(f"STDOUT: {query_result.stdout}")
                    print(f"STDERR: {query_result.stderr}")
                    raise ValueError("Failed to parse workflow query result")

                tasks = query_result["groups"][0]["tasks"]
                bad_node_ips = set([node[1] for node in results["bad_nodes"]])

                bad_node_names = [
                    task["node_name"] for task in tasks if task["pod_ip"] in bad_node_ips
                ]
                print(f"Bad node names: {bad_node_names}")

                if os.path.exists("/mnt/aws-lfs-01/"):
                    # in gb200
                    nfs_root = "/mnt/aws-lfs-01/"
                elif (
                    os.path.exists("/mnt/amlfs-01/")
                    and os.path.exists("/mnt/amlfs-02/")
                    and os.path.exists("/mnt/amlfs-03/")
                ):
                    # in h100
                    nfs_root = "/mnt/amlfs-01/"
                else:
                    raise ValueError("Unknown filesystem")

                # Merge bad node names with existing bad node list
                shared_bad_node_path = os.path.join(nfs_root, "shared", "reported_bad_nodes.txt")
                if os.path.exists(shared_bad_node_path):
                    with open(shared_bad_node_path, "r") as f:
                        existing_bad_node_list = f.read().splitlines()
                else:
                    existing_bad_node_list = []
                bad_node_set = set(existing_bad_node_list) | set(bad_node_names)
                bad_node_list = sorted(list(bad_node_set))
                with open(shared_bad_node_path, "w") as f:
                    content = "\n".join(bad_node_list)
                    f.write(content)

            # Kill processes on bad nodes if --kill flag is provided
            # Do this BEFORE shutting down Ray
            if args.kill and results.get("bad_nodes"):
                kill_summary = kill_processes_on_bad_nodes(results["bad_nodes"], drain_ray_node)
                if kill_summary:
                    results["kill_summary"] = kill_summary
                    print("\n🧹 Kill summary:")
                    print(json.dumps(kill_summary, indent=2))

        # Shutdown Ray after all operations are complete
        ray.shutdown()


if __name__ == "__main__":
    main()
