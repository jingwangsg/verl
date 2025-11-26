#!/usr/bin/env python3
"""
GPU Worker Daemon Manager
Manages GPU training workers using Ray actors - launches workers to fill available GPUs
when no pending actors/tasks, kills them when pending actors/tasks are detected.
"""

import time
import logging
import sys
import ray
from typing import List, Set, Optional, Any
from datetime import datetime, timedelta
from ray.util.state import list_actors, list_tasks, list_placement_groups

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gpu_worker_daemon.log')
    ]
)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class GPUWorker:
    """A Ray actor that runs on a single GPU with real GPU utilization."""
    
    def __init__(self, worker_id: str):
        """Initialize the GPU worker."""
        self.worker_id = worker_id
        self.is_running = True
        self.iteration = 0
        logger.info(f"GPU Worker {worker_id} initialized")
        
        # Initialize PyTorch and allocate GPU memory
        try:
            import torch
            import torch.nn as nn
            
            # Get GPU device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Worker {worker_id} using GPU: {gpu_name}")
                
                # Pre-allocate large matrices to reach ~70GB memory usage
                # Using large square matrices for efficient matrix multiplication
                self.matrices = []
                target_memory_gb = 70
                matrix_size = 16384  # 16K x 16K matrix = ~1GB in float32
                
                current_memory_gb = 0
                while current_memory_gb < target_memory_gb:
                    try:
                        # Allocate matrix (each element is 4 bytes for float32)
                        matrix = torch.randn(matrix_size, matrix_size, device=self.device, dtype=torch.float32)
                        self.matrices.append(matrix)
                        current_memory_gb = torch.cuda.memory_allocated(self.device) / 1e9
                        logger.info(f"Worker {worker_id} allocated matrix {len(self.matrices)}, total memory: {current_memory_gb:.1f}GB")
                        
                        if len(self.matrices) >= 70:  # Safety limit
                            break
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"Worker {worker_id} reached GPU memory limit at {current_memory_gb:.1f}GB")
                        break
                
                # Log final GPU memory usage
                allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
                reserved_gb = torch.cuda.memory_reserved(self.device) / 1e9
                logger.info(f"Worker {worker_id} final GPU memory: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {len(self.matrices)} matrices")
            else:
                logger.warning(f"Worker {worker_id}: No GPU available, running in CPU mode")
                self.device = None
                self.matrices = []
        except ImportError:
            logger.error(f"Worker {worker_id}: PyTorch not installed")
            self.device = None
            self.matrices = []
        except Exception as e:
            logger.error(f"Worker {worker_id} GPU init error: {e}")
            self.device = None
            self.matrices = []
    
    def run_training_loop(self):
        """
        Training loop that runs forever with continuous matrix multiplication for high GPU utilization.
        """
        logger.info(f"Worker {self.worker_id} starting infinite matrix multiplication loop")
        
        while self.is_running:
            if self.device is not None and len(self.matrices) > 0:
                try:
                    import torch
                    
                    # Continuous matrix multiplication to maintain high GPU utilization
                    # Use the pre-allocated matrices to avoid memory allocation overhead
                    num_matrices = len(self.matrices)
                    
                    # Perform matrix multiplication between different pairs
                    for i in range(10):  # 10 operations per iteration for sustained load
                        idx1 = i % num_matrices
                        idx2 = (i + 1) % num_matrices
                        
                        # Matrix multiplication (this is the GPU-intensive operation)
                        result = torch.matmul(self.matrices[idx1], self.matrices[idx2])
                        
                        # Optionally do more operations to increase utilization
                        result = torch.nn.functional.relu(result)
                        result = result.T @ result  # Transpose and multiply
                        
                        # Update one of the matrices to keep data changing
                        self.matrices[idx1] = self.matrices[idx1] * 0.999 + result * 0.001
                    
                    self.iteration += 1
                    
                    if self.iteration % 10 == 0:  # Log every 10 iterations
                        mem_allocated = torch.cuda.memory_allocated(self.device) / 1e9
                        logger.info(f"Worker {self.worker_id} iter {self.iteration}: "
                                  f"GPU mem={mem_allocated:.2f}GB, "
                                  f"matrices={num_matrices}")
                    
                    # No sleep - maintain maximum GPU utilization
                    # Only yield occasionally to prevent blocking
                    if self.iteration % 100 == 0:
                        time.sleep(0.001)  # Minimal sleep just to yield
                    
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} computation error: {e}")
                    time.sleep(1)
            else:
                # Fallback if no GPU available
                time.sleep(10)
                self.iteration += 1
                if self.iteration % 6 == 0:
                    logger.debug(f"Worker {self.worker_id} (CPU mode) iteration {self.iteration}")
        
        logger.info(f"Worker {self.worker_id} stopped after {self.iteration} iterations")
        return "stopped"
    
    def stop(self):
        """Stop the training loop and clean up GPU memory gracefully."""
        logger.info(f"GPU Worker {self.worker_id} stopping and cleaning up GPU memory...")
        self.is_running = False
        
        # Clean up GPU memory
        try:
            if hasattr(self, 'matrices') and len(self.matrices) > 0:
                logger.info(f"Worker {self.worker_id} releasing {len(self.matrices)} matrices from GPU memory")
                
                # Clear all matrices
                for i, matrix in enumerate(self.matrices):
                    del matrix
                self.matrices.clear()
                
                # Force GPU memory cleanup
                import torch
                if torch.cuda.is_available() and hasattr(self, 'device'):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Log final memory state
                    allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
                    logger.info(f"Worker {self.worker_id} final GPU memory after cleanup: {allocated_gb:.2f}GB")
                
        except Exception as e:
            logger.error(f"Worker {self.worker_id} error during GPU cleanup: {e}")
        
        logger.info(f"GPU Worker {self.worker_id} stopped and memory cleaned")
        return "stopped"
    
    def health_check(self) -> str:
        """Quick health check."""
        return f"alive:{self.worker_id}"


class GPUWorkerDaemon:
    """Manages GPU training workers based on Ray cluster resource availability."""
    
    def __init__(self,
                 ray_address: str = "auto",
                 check_interval: int = 10):
        """
        Initialize the daemon manager.

        Args:
            ray_address: Ray cluster address ("auto" for auto-discovery)
            check_interval: Interval in seconds to check for pending actors/tasks
        """
        self.ray_address = ray_address
        self.check_interval = check_interval

        # Track worker actors
        self.worker_actors: Set[ray.ObjectRef] = set()
        self.is_running = True
        self.worker_counter = 0
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)
        
        # Get current job ID to filter out our own actors/tasks
        self.current_job_id = ray.get_runtime_context().get_job_id()
        
        logger.info(f"GPU Worker Daemon initialized with job_id: {self.current_job_id}")
    
    def get_available_gpus(self) -> float:
        """Get the number of available GPUs in the cluster."""
        try:
            resources = ray.available_resources()
            return resources.get("GPU", 0.0)
        except Exception as e:
            logger.error(f"Error getting available resources: {e}")
            return 0.0
    
    def get_total_gpus(self) -> float:
        """Get the total number of GPUs in the cluster."""
        try:
            resources = ray.cluster_resources()
            return resources.get("GPU", 0.0)
        except Exception as e:
            logger.error(f"Error getting cluster resources: {e}")
            return 0.0

    def _log_pending_items(self, items: list, title: str, id_field: str,
                          show_pg_id: bool = False, max_items: int = 10) -> None:
        """Helper method to log pending items (tasks/actors) with consistent formatting.

        Args:
            items: List of items to log
            title: Display title (e.g., "Tasks without PG")
            id_field: Name of the ID field ("task_id" or "actor_id")
            show_pg_id: Whether to display placement group ID
            max_items: Maximum number of items to display in detail
        """
        if not items:
            return

        logger.info(f"\n{title} ({len(items)}):")
        for i, item in enumerate(items[:max_items], 1):
            logger.info(f"  {i}. {id_field.replace('_', ' ').title()}: {item[id_field]}")
            logger.info(f"     Name: {item['name']}")
            logger.info(f"     State: {item['state']}")
            if show_pg_id:
                logger.info(f"     PG ID: {item.get('pg_id', 'N/A')}")

        if len(items) > max_items:
            logger.info(f"  ... and {len(items) - max_items} more")

    def analyze_pending_workload(self) -> dict:
        """
        Analyze pending workload and distinguish between PG and non-PG tasks/actors.

        Returns:
            Dictionary with detailed analysis:
            - tasks_without_pg: List of tasks not using placement groups
            - tasks_with_pending_pg: List of tasks using PENDING placement groups
            - tasks_with_created_pg: List of tasks using CREATED placement groups
            - actors_without_pg: Similar for actors
            - actors_with_pending_pg: Similar for actors
            - actors_with_created_pg: Similar for actors
            - pg_stats: Statistics about placement groups
        """
        result = {
            "tasks_without_pg": [],
            "tasks_with_pending_pg": [],
            "tasks_with_created_pg": [],
            "actors_without_pg": [],
            "actors_with_pending_pg": [],
            "actors_with_created_pg": [],
            "pg_stats": {
                "pending": 0,
                "created": 0,
                "total": 0
            }
        }

        try:
            # Get all placement groups and their states
            all_pgs = list_placement_groups(detail=True)
            pg_info = {}
            for pg in all_pgs:
                pg_info[pg.placement_group_id] = {
                    "state": pg.state,
                    "name": pg.name if hasattr(pg, 'name') else None
                }
                result["pg_stats"]["total"] += 1
                if pg.state == "PENDING":
                    result["pg_stats"]["pending"] += 1
                elif pg.state == "CREATED":
                    result["pg_stats"]["created"] += 1

            # Analyze pending tasks (focus on resource-waiting states)
            resource_pending_states = [
                "PENDING_NODE_ASSIGNMENT",  # Most important - waiting for node/resource
                "PENDING_OBJ_STORE_MEM_AVAIL"  # Waiting for object store memory
            ]

            for state in resource_pending_states:
                tasks = list_tasks(
                    filters=[
                        ("job_id", "!=", self.current_job_id),
                        ("state", "=", state)
                    ],
                    detail=True
                )

                for task in tasks:
                    task_info = {
                        "task_id": task.task_id,
                        "name": task.name if hasattr(task, 'name') else "unknown",
                        "state": state
                    }

                    if hasattr(task, 'placement_group_id') and task.placement_group_id:
                        # Task uses a placement group
                        pg_state = pg_info.get(task.placement_group_id, {}).get("state", "UNKNOWN")
                        task_info["pg_id"] = task.placement_group_id
                        task_info["pg_state"] = pg_state

                        if pg_state == "PENDING":
                            result["tasks_with_pending_pg"].append(task_info)
                        elif pg_state == "CREATED":
                            result["tasks_with_created_pg"].append(task_info)
                        else:
                            # Unknown PG state, treat as pending to be safe
                            result["tasks_with_pending_pg"].append(task_info)
                    else:
                        # Task does not use placement group
                        result["tasks_without_pg"].append(task_info)

            # Analyze pending actors
            actor_pending_states = ["PENDING_CREATION", "DEPENDENCIES_UNREADY"]

            for state in actor_pending_states:
                actors = list_actors(
                    filters=[("state", "=", state)],
                    detail=True
                )

                # Filter by job_id manually (list_actors might not support != filter)
                actors = [a for a in actors if a.job_id != self.current_job_id]

                for actor in actors:
                    actor_info = {
                        "actor_id": actor.actor_id,
                        "name": actor.name if hasattr(actor, 'name') else "unknown",
                        "state": state
                    }

                    if hasattr(actor, 'placement_group_id') and actor.placement_group_id:
                        # Actor uses a placement group
                        pg_state = pg_info.get(actor.placement_group_id, {}).get("state", "UNKNOWN")
                        actor_info["pg_id"] = actor.placement_group_id
                        actor_info["pg_state"] = pg_state

                        if pg_state == "PENDING":
                            result["actors_with_pending_pg"].append(actor_info)
                        elif pg_state == "CREATED":
                            result["actors_with_created_pg"].append(actor_info)
                        else:
                            result["actors_with_pending_pg"].append(actor_info)
                    else:
                        # Actor does not use placement group
                        result["actors_without_pg"].append(actor_info)

        except Exception as e:
            logger.error(f"Error analyzing pending workload: {e}", exc_info=True)

        return result

    def launch_workers_for_available_gpus(self):
        """Launch workers to fill all available GPU resources."""
        # Debug: Check type at start of method
        logger.debug(f"At start of launch_workers: worker_actors type = {type(self.worker_actors)}, content = {self.worker_actors}")

        # Ensure worker_actors is a set (defensive programming)
        if not isinstance(self.worker_actors, set):
            logger.warning(f"worker_actors was {type(self.worker_actors)}, converting to set")
            self.worker_actors = set(self.worker_actors) if self.worker_actors else set()

        available_gpus = self.get_available_gpus()

        if available_gpus < 1:
            logger.debug(f"No available GPUs (available: {available_gpus})")
            return

        # Calculate how many workers we can launch (1 worker per GPU)
        num_workers_to_launch = int(available_gpus)

        if num_workers_to_launch > 0:
            logger.info(f"Launching {num_workers_to_launch} workers to use {available_gpus} available GPUs")

            launched_count = 0
            for _ in range(num_workers_to_launch):
                try:
                    self.worker_counter += 1
                    worker_id = f"worker_{self.worker_counter}"

                    # Create worker actor with 1 GPU
                    worker = GPUWorker.options(
                        name=f"gpu_{worker_id}",
                        num_gpus=1,
                        max_restarts=0,  # Don't restart to release resources quickly
                        max_task_retries=0
                    ).remote(worker_id)

                    self.worker_actors.add(worker)

                    # Start the infinite training loop (non-blocking)
                    worker.run_training_loop.remote()

                    launched_count += 1

                except Exception as e:
                    logger.error(f"Error launching worker: {e}")
                    break

            logger.info(f"✅ Successfully launched {launched_count} workers")
    
    def kill_all_workers(self):
        """Kill all running workers with graceful GPU memory cleanup."""
        if not self.worker_actors:
            return
            
        logger.info(f"Gracefully stopping {len(self.worker_actors)} worker actors and cleaning GPU memory...")
        
        # First, send stop signal to all workers (non-blocking)
        stop_futures = []
        for worker in self.worker_actors:
            try:
                stop_futures.append(worker.stop.remote())
            except:
                pass
        
        # Give them more time to stop gracefully and clean up GPU memory
        if stop_futures:
            try:
                logger.info("Waiting for workers to clean up GPU memory...")
                ready, remaining = ray.wait(stop_futures, num_returns=len(stop_futures), timeout=10)
                logger.info(f"{len(ready)}/{len(stop_futures)} workers completed graceful shutdown")
                
                # Get results to see cleanup logs
                if ready:
                    results = ray.get(ready)
                    for result in results:
                        logger.debug(f"Worker result: {result}")
                        
            except Exception as e:
                logger.warning(f"Error during graceful shutdown: {e}")
        
        # Force kill any remaining workers
        logger.info("Force killing any remaining worker actors...")
        for worker in self.worker_actors:
            try:
                ray.kill(worker)
            except:
                pass
        
        self.worker_actors.clear()
        logger.info("All workers stopped and GPU memory cleaned")
        
        # Debug: Check type after clear
        logger.debug(f"After kill_all: worker_actors type = {type(self.worker_actors)}")

    def run(self):
        """Main daemon loop - Single loop with PG-aware resource management."""
        logger.info("Starting GPU Worker Daemon...")
        logger.info(f"Ray address: {self.ray_address}")
        logger.info(f"Check interval: {self.check_interval}s")

        total_gpus = self.get_total_gpus()
        logger.info(f"Total GPUs in cluster: {total_gpus}")

        if total_gpus == 0:
            logger.error("No GPUs detected in cluster, exiting...")
            return

        last_detailed_print_time = None

        try:
            while self.is_running:
                # Analyze pending workload with PG awareness
                analysis = self.analyze_pending_workload()

                # Determine if we need to kill workers
                # Kill workers if there are pending jobs that need resources:
                # - Non-PG tasks/actors (direct resource requests)
                # - Tasks/actors using PENDING PGs (PG is requesting resources)
                should_kill_workers = (
                    len(analysis["tasks_without_pg"]) > 0 or
                    len(analysis["actors_without_pg"]) > 0 or
                    len(analysis["tasks_with_pending_pg"]) > 0 or
                    len(analysis["actors_with_pending_pg"]) > 0
                )

                # Check if there's any pending work at all (including CREATED PG)
                has_any_pending = should_kill_workers or (
                    len(analysis["tasks_with_created_pg"]) > 0 or
                    len(analysis["actors_with_created_pg"]) > 0
                )

                # Check if we should print detailed information (every 300 seconds = 5 minutes)
                current_time = datetime.now()
                should_print_details = (
                    last_detailed_print_time is None or
                    (current_time - last_detailed_print_time).total_seconds() >= 300
                )

                if has_any_pending and should_print_details:
                    # Print detailed pending job information
                    logger.info("=" * 60)
                    logger.info("PENDING JOBS DETAILS:")
                    logger.info("=" * 60)

                    # Summary
                    logger.info("Summary:")
                    logger.info(f"  Tasks without PG: {len(analysis['tasks_without_pg'])}")
                    logger.info(f"  Tasks with PENDING PG: {len(analysis['tasks_with_pending_pg'])}")
                    logger.info(f"  Tasks with CREATED PG: {len(analysis['tasks_with_created_pg'])}")
                    logger.info(f"  Actors without PG: {len(analysis['actors_without_pg'])}")
                    logger.info(f"  Actors with PENDING PG: {len(analysis['actors_with_pending_pg'])}")
                    logger.info(f"  Actors with CREATED PG: {len(analysis['actors_with_created_pg'])}")
                    logger.info(f"  Placement Groups - PENDING: {analysis['pg_stats']['pending']}, "
                              f"CREATED: {analysis['pg_stats']['created']}")

                    # Detailed task and actor information
                    self._log_pending_items(analysis["tasks_without_pg"], "Tasks without PG", "task_id")
                    self._log_pending_items(analysis["tasks_with_pending_pg"], "Tasks with PENDING PG", "task_id", show_pg_id=True)
                    self._log_pending_items(analysis["tasks_with_created_pg"], "Tasks with CREATED PG", "task_id", show_pg_id=True)
                    self._log_pending_items(analysis["actors_without_pg"], "Actors without PG", "actor_id")
                    self._log_pending_items(analysis["actors_with_pending_pg"], "Actors with PENDING PG", "actor_id", show_pg_id=True)
                    self._log_pending_items(analysis["actors_with_created_pg"], "Actors with CREATED PG", "actor_id", show_pg_id=True)

                    # Resource information
                    available_gpus = self.get_available_gpus()
                    reserved_gpus = total_gpus - available_gpus
                    logger.info(f"Resources:")
                    logger.info(f"  Total GPUs: {total_gpus}")
                    logger.info(f"  Available GPUs: {available_gpus}")
                    logger.info(f"  Reserved/Used GPUs: {reserved_gpus}")
                    logger.info("=" * 60)
                    last_detailed_print_time = current_time

                # Execute action based on analysis
                if should_kill_workers:
                    # Kill workers to release resources for pending jobs
                    if self.worker_actors:
                        reason_parts = []
                        if analysis["tasks_without_pg"]:
                            reason_parts.append(f"{len(analysis['tasks_without_pg'])} non-PG tasks")
                        if analysis["actors_without_pg"]:
                            reason_parts.append(f"{len(analysis['actors_without_pg'])} non-PG actors")
                        if analysis["tasks_with_pending_pg"]:
                            reason_parts.append(f"{len(analysis['tasks_with_pending_pg'])} tasks with PENDING PG")
                        if analysis["actors_with_pending_pg"]:
                            reason_parts.append(f"{len(analysis['actors_with_pending_pg'])} actors with PENDING PG")

                        reason = " + ".join(reason_parts)
                        logger.info(f"🔴 KILLING WORKERS - Reason: {reason}")
                        self.kill_all_workers()
                    else:
                        logger.debug("Would kill workers but none running")

                elif not has_any_pending:
                    # No pending jobs - check if we should launch/refill workers
                    available_gpus = self.get_available_gpus()
                    current_workers = len(self.worker_actors)

                    if available_gpus >= 1:
                        logger.info(f"🟢 No pending jobs, launching workers to fill {available_gpus} available GPUs")
                        self.launch_workers_for_available_gpus()
                    else:
                        logger.debug(f"No pending jobs and no available GPUs (workers: {current_workers}, available: {available_gpus})")

                    # Print status every 5 minutes when no pending jobs
                    if should_print_details:
                        logger.info("=" * 60)
                        logger.info("STATUS: No pending jobs detected")
                        logger.info(f"  Workers running: {len(self.worker_actors)}")
                        logger.info(f"  Available GPUs: {int(self.get_available_gpus())}/{int(total_gpus)}")
                        logger.info("  Daemon is keeping workers active to hold cluster resources")
                        logger.info("=" * 60)

                else:
                    # Only CREATED PG tasks/actors pending - keep workers
                    if should_print_details:
                        logger.info("🟢 KEEPING WORKERS - Only CREATED PG tasks/actors pending")
                        logger.info("These workloads are using pre-reserved PG resources")
                        logger.info("Killing workers would NOT help them (resources already reserved)")
                    else:
                        logger.debug(f"Only CREATED PG pending - keeping workers")

                # Sleep before next check
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            self.shutdown()
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
            self.shutdown()
            raise
    
    def shutdown(self):
        """Gracefully shutdown the daemon."""
        logger.info("Shutting down GPU worker daemon...")
        self.is_running = False
        
        # Kill all running workers
        self.kill_all_workers()
        
        # Shutdown Ray if we initialized it
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("GPU worker daemon shutdown complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Worker Daemon Manager')
    parser.add_argument('--ray-address', default='auto',
                        help='Ray cluster address (default: auto)')
    parser.add_argument('--check-interval', type=int, default=5,
                        help='Interval in seconds to check for pending actors/tasks')

    args = parser.parse_args()

    daemon = GPUWorkerDaemon(
        ray_address=args.ray_address,
        check_interval=args.check_interval
    )
    
    try:
        daemon.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()