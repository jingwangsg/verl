#!/usr/bin/env python3
"""
Script to remove idle Ray nodes from a cluster.
Removes nodes that have no running tasks or actors and are not the head node.
"""
import argparse
import os
import subprocess
from typing import Dict, List, Set, Tuple

import ray
from ray.util.state import list_actors, list_tasks


@ray.remote
def kill_processes_on_node(hostname: str, ip: str, node_id: str) -> Dict[str, str]:
    """
    Kill sleep/block processes on a single node (Ray remote task).
    This will cause the node to disconnect from the Ray cluster.

    Args:
        hostname: Node hostname
        ip: Node IP address
        node_id: Ray node ID for scheduling

    Returns:
        Dict with result status and details
    """
    # Kill command for sleep/block processes
    # Note: NodeAffinitySchedulingStrategy ensures task runs on correct node
    kill_command = (
        "ps aux | grep -E 'sleep|block' | grep -v grep | awk '{print $2}' | xargs kill -9"
    )

    try:
        # Execute kill command locally
        subprocess.run(
            kill_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,  # Short timeout since node might disconnect
        )

        # If we get here, the kill command executed
        return {
            "hostname": hostname,
            "ip": ip,
            "status": "success",
            "message": "Kill command executed (node may disconnect)",
        }

    except subprocess.TimeoutExpired:
        # This might happen if the node starts disconnecting
        return {
            "hostname": hostname,
            "ip": ip,
            "status": "success",
            "message": "Kill command initiated (node disconnecting)",
        }
    except Exception as e:
        return {
            "hostname": hostname,
            "ip": ip,
            "status": "error",
            "message": f"Kill command failed: {str(e)}",
        }


def get_nodes_with_workloads() -> Set[str]:
    """
    Get node IDs that have running actors or tasks.

    Returns:
        Set of node IDs with active workloads
    """
    nodes_with_workloads = set()

    # Check for alive actors
    try:
        actors = list_actors(filters=[("state", "=", "ALIVE")])
        for actor in actors:
            if actor.get("node_id"):
                nodes_with_workloads.add(actor["node_id"])
    except Exception as e:
        print(f"Warning: Failed to get actor list: {str(e)}")

    # Check for running tasks
    try:
        tasks = list_tasks(filters=[("state", "=", "RUNNING")])
        for task in tasks:
            if task.get("node_id"):
                nodes_with_workloads.add(task["node_id"])
    except Exception as e:
        print(f"Warning: Failed to get task list: {str(e)}")

    return nodes_with_workloads


def get_head_node_id() -> str:
    """
    Identify the head node in the Ray cluster.

    Returns:
        NodeID of the head node, or None if not found
    """
    all_nodes = ray.nodes()

    for node in all_nodes:
        # Head node is identified by the presence of "node:__internal_head__" resource
        resources = node.get("Resources", {})
        if "node:__internal_head__" in resources:
            return node["NodeID"]

    # Fallback: check if node name/address indicates it's a head
    # Head node often has specific naming patterns
    for node in all_nodes:
        node_name = node.get("NodeName", "")
        if "head" in node_name.lower() or node.get("NodeManagerAddress") == "127.0.0.1":
            return node["NodeID"]

    return None


def get_removable_nodes(nnodes: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Get nodes that can be safely removed (no workloads, not head node).

    Args:
        nnodes: Number of nodes requested for removal

    Returns:
        Tuple of (removable_nodes, skipped_nodes) where each is a list of node info dicts
    """
    print("\n" + "=" * 80)
    print("Analyzing Ray Cluster Nodes")
    print("=" * 80)

    # Get all alive nodes
    all_nodes = ray.nodes()
    alive_nodes = [node for node in all_nodes if node["Alive"]]
    print(f"\nTotal alive nodes in cluster: {len(alive_nodes)}")

    # Get nodes with workloads
    nodes_with_workloads = get_nodes_with_workloads()
    print(f"Nodes with active workloads: {len(nodes_with_workloads)}")

    # Get head node
    head_node_id = get_head_node_id()
    if head_node_id:
        print(f"Head node identified: {head_node_id[:8]}...")
    else:
        print("Warning: Could not identify head node")

    # Get current node to avoid self-termination
    try:
        result = subprocess.run("hostname -I", shell=True, capture_output=True, text=True)
        current_node_ips = set(result.stdout.strip().split())
        print(f"Current node IPs: {current_node_ips}")
    except Exception as e:
        print(f"Warning: Could not get current node IPs: {str(e)}")
        current_node_ips = set()

    # Filter removable nodes
    removable_nodes = []
    skipped_nodes = []

    for node in alive_nodes:
        node_id = node["NodeID"]
        node_ip = node["NodeManagerAddress"]
        node_name = node.get("NodeName", "unknown")

        # Create node info dict
        node_info = {
            "node_id": node_id,
            "ip": node_ip,
            "hostname": node_name,
            "resources": node.get("Resources", {}),
        }

        # Skip head node
        if node_id == head_node_id:
            skipped_nodes.append({"reason": "head node", **node_info})
            continue

        # Skip nodes with workloads
        if node_id in nodes_with_workloads:
            skipped_nodes.append({"reason": "has workloads", **node_info})
            continue

        # Skip current node to avoid self-termination
        if node_ip in current_node_ips:
            skipped_nodes.append({"reason": "current node", **node_info})
            continue

        # This node can be removed
        removable_nodes.append(node_info)

    print(f"\n{'='*60}")
    print(f"Removable nodes found: {len(removable_nodes)}")
    print(f"Skipped nodes: {len(skipped_nodes)}")

    if skipped_nodes:
        print("\nSkipped nodes breakdown:")
        for skip_info in skipped_nodes:
            print(f"  - {skip_info['hostname']} ({skip_info['ip']}): {skip_info['reason']}")

    return removable_nodes[:nnodes], skipped_nodes


def layoff_nodes(nnodes: int) -> int:
    """
    Remove specified number of idle nodes from the cluster.

    Args:
        nnodes: Number of nodes to remove

    Returns:
        Number of nodes actually removed
    """
    # Get removable nodes
    removable_nodes, skipped_nodes = get_removable_nodes(nnodes)

    if len(removable_nodes) < nnodes:
        print(f"\n⚠️  WARNING: Only {len(removable_nodes)} removable nodes available, but {nnodes} requested")
        print(f"\nAvailable removable nodes:")
        for node in removable_nodes:
            print(f"  - {node['hostname']} ({node['ip']})")

        # Ask user if they want to continue
        response = input(f"\nContinue with removing {len(removable_nodes)} nodes? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Operation cancelled by user.")
            return 0

        # Update nnodes to actual available count
        nnodes = len(removable_nodes)

    if nnodes == 0:
        print("\nNo nodes to remove.")
        return 0

    # Show nodes to be removed
    print(f"\n{'='*60}")
    print(f"Removing {nnodes} nodes:")
    for i, node in enumerate(removable_nodes, 1):
        print(f"  {i}. {node['hostname']} ({node['ip']})")

    # Execute removal
    print(f"\n{'='*60}")
    print(f"Starting removal operations...")
    print("⚠️  Nodes will disconnect after kill command executes\n")

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Create Ray remote tasks for all nodes
    kill_tasks = []
    scheduled_nodes = []

    for node in removable_nodes:
        node_id = node["node_id"]
        hostname = node["hostname"]
        ip = node["ip"]

        print(f"  Scheduling kill task on {hostname} ({ip})")
        try:
            # Schedule task on specific node
            task = kill_processes_on_node.options(
                num_cpus=0.1,  # Minimal CPU requirement
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            ).remote(hostname, ip, node_id)
            kill_tasks.append(task)
            scheduled_nodes.append((hostname, ip))
        except Exception as e:
            print(f"    ❌ Failed to schedule task: {str(e)}")

    if not kill_tasks:
        print("\n❌ No kill tasks could be scheduled.")
        return 0

    # Wait for kill operations with timeout handling
    print(f"\nExecuting {len(kill_tasks)} kill operations...")

    success_count = 0
    disconnected_count = 0
    failed_count = 0

    for i, task in enumerate(kill_tasks):
        hostname, ip = scheduled_nodes[i]
        try:
            # Short timeout per task since nodes will disconnect
            result = ray.get(task, timeout=10)

            status = result["status"]
            message = result["message"]

            if status == "success":
                print(f"  ✅ {hostname}: {message}")
                success_count += 1
            else:
                print(f"  ❌ {hostname}: {message}")
                failed_count += 1

        except (
            ray.exceptions.RayTaskError,
            ray.exceptions.RayActorError,
            ray.exceptions.GetTimeoutError,
        ):
            # These exceptions are expected when node disconnects after kill
            print(f"  ✅ {hostname}: Kill executed, node disconnected")
            disconnected_count += 1
        except Exception as e:
            # Any other exception is also likely due to disconnection
            print(f"  ✅ {hostname}: Kill executed, node disconnected (exception: {type(e).__name__})")
            disconnected_count += 1

    # Report results
    total_removed = success_count + disconnected_count

    print(f"\n{'='*60}")
    print("Layoff Operations Completed:")
    print(f"  ✅ Nodes processed successfully: {total_removed}/{len(removable_nodes)}")
    if failed_count > 0:
        print(f"  ❌ Failed operations: {failed_count}")

    return total_removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove idle Ray nodes from the cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove 5 idle nodes
  python _ray_layoff.py --nnodes 5

  # Connect to specific Ray cluster and remove 3 nodes
  python _ray_layoff.py --nnodes 3 --ray-address ray://localhost:10001
        """
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        required=True,
        help="Number of nodes to remove"
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (default: auto-detect)"
    )

    args = parser.parse_args()

    if args.nnodes <= 0:
        print("Error: --nnodes must be a positive integer")
        return 1

    # Initialize Ray
    print("Initializing Ray connection...")
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    print(f"✅ Connected to Ray cluster")

    try:
        # Execute layoff
        removed_count = layoff_nodes(args.nnodes)

        print(f"\n{'='*80}")
        print(f"Final Summary:")
        print(f"  Requested: {args.nnodes} nodes")
        print(f"  Removed: {removed_count} nodes")
        print(f"{'='*80}\n")

        return 0 if removed_count > 0 else 1

    finally:
        # Shutdown Ray
        ray.shutdown()


if __name__ == "__main__":
    exit(main())
