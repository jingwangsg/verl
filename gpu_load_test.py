#!/usr/bin/env python3
"""
Multi-GPU Load Test Script
Occupies ~50% GPU memory and maintains 100% GPU utilization on all available GPUs.
Press Ctrl+C to stop.
"""

import torch
import torch.multiprocessing as mp
import time
import argparse


def get_gpu_memory_gb(device_id=0):
    """Get total GPU memory in GB."""
    return torch.cuda.get_device_properties(device_id).total_memory / (1024**3)


def occupy_gpu_memory(device_id=0, memory_fraction=0.5):
    """
    Allocate tensors to occupy a fraction of GPU memory.

    Args:
        device_id: GPU device ID
        memory_fraction: Fraction of GPU memory to occupy (0.0 to 1.0)

    Returns:
        List of allocated tensors
    """
    device = torch.device(f'cuda:{device_id}')
    total_memory_gb = get_gpu_memory_gb(device_id)
    target_memory_gb = total_memory_gb * memory_fraction

    print(f"GPU {device_id}: Total memory = {total_memory_gb:.2f} GB")
    print(f"GPU {device_id}: Target memory usage = {target_memory_gb:.2f} GB ({memory_fraction*100:.0f}%)")

    # Allocate memory in chunks
    tensors = []
    allocated_gb = 0
    chunk_size_gb = 0.5  # Allocate in 500MB chunks for faster allocation

    while allocated_gb < target_memory_gb * 0.95:  # Leave some margin
        try:
            # Calculate elements needed for this chunk (float32 = 4 bytes)
            num_elements = int(chunk_size_gb * 1024**3 / 4)
            tensor = torch.randn(num_elements, device=device, dtype=torch.float32)
            tensors.append(tensor)
            allocated_gb += chunk_size_gb
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU {device_id}: Reached memory limit at {allocated_gb:.2f} GB")
                break
            raise

    print(f"GPU {device_id}: Successfully allocated {allocated_gb:.2f} GB")
    return tensors


def worker_process(device_id, memory_fraction):
    """
    Worker process for a single GPU.

    Args:
        device_id: GPU device ID to use
        memory_fraction: Fraction of GPU memory to occupy
    """
    try:
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')

        print(f"\n[GPU {device_id}] Starting...")

        # Occupy GPU memory
        tensors = occupy_gpu_memory(device_id=device_id, memory_fraction=memory_fraction)

        # Create working tensors for computation
        size = 8192  # Large matrix size to keep GPU busy
        A = torch.randn(size, size, device=device, dtype=torch.float32)
        B = torch.randn(size, size, device=device, dtype=torch.float32)

        print(f"[GPU {device_id}] Starting computation loop...")

        iteration = 0
        start_time = time.time()

        while True:
            # Perform intensive matrix operations
            C = torch.matmul(A, B)
            D = torch.matmul(B, A)

            # Additional operations to maximize utilization
            E = C + D
            F = torch.relu(E)
            G = torch.softmax(F, dim=0)

            # In-place operations to create continuous work
            A.add_(G * 0.001)
            B.add_(C * 0.001)

            # Synchronize to ensure operations complete
            torch.cuda.synchronize(device)

            iteration += 1

            # Print status every 100 iterations
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                iter_per_sec = iteration / elapsed
                print(f"[GPU {device_id}] Iteration {iteration} | {iter_per_sec:.2f} iter/s")

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n[GPU {device_id}] Stopping...")
        print(f"[GPU {device_id}] Total iterations: {iteration}")
        print(f"[GPU {device_id}] Total time: {elapsed:.2f}s")
        print(f"[GPU {device_id}] Average iterations/sec: {iteration/elapsed:.2f}")
    except Exception as e:
        print(f"[GPU {device_id}] Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Load Test - Occupy memory and maintain utilization')
    parser.add_argument('--devices', type=str, default='all',
                        help='GPU device IDs to use, comma-separated (e.g., "0,1,2") or "all" (default: all)')
    parser.add_argument('--memory', type=float, default=0.5,
                        help='Fraction of GPU memory to occupy per GPU (default: 0.5 for 50%%)')
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires GPUs.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")

    # Parse device list
    if args.devices == 'all':
        device_ids = list(range(num_gpus))
    else:
        device_ids = [int(d.strip()) for d in args.devices.split(',')]
        # Validate device IDs
        for d in device_ids:
            if d >= num_gpus:
                print(f"Error: GPU {d} not found. Available GPUs: 0-{num_gpus-1}")
                return

    if not (0.0 < args.memory <= 1.0):
        print("Error: Memory fraction must be between 0.0 and 1.0")
        return

    print("="*80)
    print("Multi-GPU Load Test Script")
    print("="*80)
    print(f"Target devices: GPU {device_ids}")
    print(f"Target memory usage per GPU: {args.memory*100:.0f}%")
    print(f"Total GPUs to use: {len(device_ids)}")
    print("="*80)
    print("\nPress Ctrl+C to stop all processes\n")

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Create a process for each GPU
    processes = []
    for device_id in device_ids:
        p = mp.Process(target=worker_process, args=(device_id, args.memory))
        p.start()
        processes.append(p)
        time.sleep(0.5)  # Stagger process starts slightly

    try:
        # Wait for all processes
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\nStopping all GPU processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("All processes stopped. Exiting.")


if __name__ == "__main__":
    main()
