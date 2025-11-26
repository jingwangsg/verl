#!/usr/bin/env python3
import os
import ray
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import argparse
import time
import asyncio
import subprocess
from datetime import timedelta
from typing import Dict, List, Tuple, Set
from ray.train._internal.utils import get_address_and_port
from ray.train.torch.train_loop_utils import get_device
from collections import defaultdict
import ray.exceptions
import json

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

@ray.remote
class TrainingWorker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.hostname = os.uname().nodename
    
    def get_info(self):
        """Get worker information including hostname and IP"""
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "ip": ray.util.get_node_ip_address()
        }
    
    def get_address(self):
        """Get the address and port for the master node"""
        return get_address_and_port()
    
    def setup_env_vars(self, addr, port, world_rank, world_size, debug_level="TRACE"):
        """Setup distributed training environment variables with enhanced debug output"""
        os.environ['MASTER_ADDR'] = addr
        os.environ['MASTER_PORT'] = str(port)
        os.environ['RANK'] = str(world_rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        # Enhanced NCCL debugging
        print(f"Worker {self.worker_id} (rank {world_rank}): Setting up enhanced debug environment")
        os.environ["NCCL_DEBUG"] = debug_level  # TRACE, INFO, WARN
        os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"  # ALL, INIT, COLL, P2P, SHM, NET, GRAPH, TUNING
        
        # Additional NCCL debugging flags
        # os.environ["NCCL_DEBUG_FILE"] = f"/tmp/nccl_debug_worker_{self.worker_id}_rank_{world_rank}.log"
        os.environ["NCCL_CHECK_DISABLE"] = "0"
        os.environ["NCCL_WARN_ALIGN"] = "1"
        
        # Fabric Interface debugging
        # os.environ["FI_PROVIDER"] = "efa"
        os.environ["FI_LOG_LEVEL"] = "debug"
        # os.environ["FI_LOG_PROV"] = "efa"
        os.environ["FI_EFA_ENABLE_SHM_TRANSFER"] = "0"  # Disable shared memory for debugging
        
        # Additional EFA debugging
        # os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
        # os.environ["FI_EFA_USE_DEVICE_RDMA"] = "0"
        os.environ["RDMAV_FORK_SAFE"] = "1"
        
        # CUDA debugging
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        
        print(f"Worker {self.worker_id}: Debug environment configured")
        print(f"  MASTER_ADDR: {addr}")
        print(f"  MASTER_PORT: {port}")
        print(f"  RANK: {world_rank}")
        print(f"  WORLD_SIZE: {world_size}")
        print(f"  NCCL_DEBUG: {debug_level}")
        print(f"  Debug log file: /tmp/nccl_debug_worker_{self.worker_id}_rank_{world_rank}.log")
    
    def train_ddp(self, epochs: int, timeout: float) -> Dict:
        """Run DDP training with enhanced debugging output"""
        try:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            
            print(f"Worker {self.worker_id} (rank {rank}): Starting DDP training")
            print(f"  Hostname: {self.hostname}")
            print(f"  World size: {world_size}")
            print(f"  Timeout: {timeout}s")
            
            # Initialize distributed training
            print(f"Worker {self.worker_id}: Initializing process group...")
            start_time = time.time()
            
            dist.init_process_group(
                backend='nccl',
                init_method=f"env://",
                timeout=timedelta(seconds=timeout),
                rank=rank,
                world_size=world_size
            )
            dist.new_group(backend="gloo")
            
            init_time = time.time() - start_time
            print(f"Worker {self.worker_id}: Process group initialized in {init_time:.2f}s")
            
            # Get device and print CUDA info
            device = get_device()
            cuda_device = None
            if torch.cuda.is_available():
                cuda_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(cuda_device)
                print(f"Worker {self.worker_id}: Using CUDA device {cuda_device}: {device_name}")
                print(f"  Device memory: {torch.cuda.get_device_properties(cuda_device).total_memory / 1e9:.1f} GB")
            
            # Create model and wrap with DDP
            print(f"Worker {self.worker_id}: Creating and wrapping model with DDP...")
            model = SimpleModel().to(device)
            model = DDP(model, device_ids=[cuda_device] if device.type == 'cuda' else None)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Create dummy data with different seeds per rank
            torch.manual_seed(42 + rank)  # Different seed per rank
            data = torch.randn(100, 10).to(device)
            targets = torch.randn(100, 1).to(device)
            dataset = TensorDataset(data, targets)
            sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size)
            dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)
            
            print(f"Worker {self.worker_id}: Starting training loop for {epochs} epochs...")
            
            # Training loop with detailed timing
            for epoch in range(epochs):
                epoch_start = time.time()
                sampler.set_epoch(epoch)
                total_loss = 0.0
                batch_count = 0
                
                print(f"Worker {self.worker_id}: Epoch {epoch} starting...")
                
                for batch_idx, (batch_data, batch_targets) in enumerate(dataloader):
                    batch_start = time.time()
                    
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    
                    # DDP gradient synchronization happens in loss.backward()
                    # optimizer.step() only does parameter updates
                    optimizer.step()
                    sync_time = 0.0  # Real sync happens in backward(), not here
                    
                    total_loss += loss.item()
                    batch_count += 1
                    batch_time = time.time() - batch_start
                    
                    if batch_idx % 5 == 0:  # Log every 5 batches
                        print(f"Worker {self.worker_id}: Epoch {epoch}, Batch {batch_idx}: "
                              f"loss={loss.item():.4f}, batch_time={batch_time:.3f}s, sync_time={sync_time:.3f}s")
                
                avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
                epoch_time = time.time() - epoch_start
                
                print(f"Worker {self.worker_id}: Epoch {epoch} completed in {epoch_time:.2f}s, avg_loss={avg_loss:.4f}")
                
                # Test all-reduce communication (ALL ranks must participate)
                print(f"Worker {self.worker_id}: Testing all-reduce communication...")
                test_tensor = torch.ones(1).to(device) * rank
                dist.all_reduce(test_tensor)
                expected_sum = sum(range(world_size))
                
                # Add synchronization barrier
                dist.barrier()
                
                if rank == 0:
                    if abs(test_tensor.item() - expected_sum) < 1e-6:
                        print(f"Worker {self.worker_id}: All-reduce test PASSED (sum={test_tensor.item()})")
                    else:
                        print(f"Worker {self.worker_id}: All-reduce test FAILED (got {test_tensor.item()}, expected {expected_sum})")
                else:
                    # Other ranks also verify but don't print (avoid spam)
                    if abs(test_tensor.item() - expected_sum) >= 1e-6:
                        print(f"Worker {self.worker_id}: All-reduce test FAILED (got {test_tensor.item()}, expected {expected_sum})")
            
            print(f"Worker {self.worker_id}: Training completed successfully")
            dist.destroy_process_group()
            
            return {
                "status": "success",
                "worker_id": self.worker_id,
                "hostname": self.hostname,
                "rank": rank,
                "final_loss": avg_loss,
                "epochs_completed": epochs,
                "init_time": init_time,
                "total_training_time": time.time() - start_time
            }
            
        except Exception as e:
            print(f"Worker {self.worker_id} training failed: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            
            # Print additional debug info on failure
            if "NCCL" in str(e):
                print(f"Worker {self.worker_id}: NCCL error detected, checking debug log...")
                debug_log_file = f"/tmp/nccl_debug_worker_{self.worker_id}_rank_{os.environ.get('RANK', 'unknown')}.log"
                if os.path.exists(debug_log_file):
                    try:
                        with open(debug_log_file, 'r') as f:
                            log_content = f.read()
                            # Print last few lines of debug log
                            lines = log_content.strip().split('\n')
                            print(f"Worker {self.worker_id}: Last 10 lines of NCCL debug log:")
                            for line in lines[-10:]:
                                print(f"  {line}")
                    except Exception as log_e:
                        print(f"Worker {self.worker_id}: Failed to read debug log: {log_e}")
            
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except:
                pass
            
            return {
                "status": "failed",
                "worker_id": self.worker_id,
                "hostname": self.hostname,
                "rank": os.environ.get('RANK', 'unknown'),
                "error": str(e),
                "error_type": type(e).__name__
            }

class HostnameDDPTrainer:
    def __init__(self, hostnames: List[str], workers_per_node: int = 4, epochs: int = 2, timeout: float = 60.0, debug_level: str = "TRACE"):
        self.hostnames = hostnames
        self.workers_per_node = workers_per_node
        self.epochs = epochs
        self.timeout = timeout
        self.debug_level = debug_level
        self.workers = []
        self.worker_info = {}
        self.hostname_to_workers = {}
        
    def initialize_workers(self):
        """Create Ray workers on specified hostnames"""
        print(f"Initializing workers on {len(self.hostnames)} hostnames...")
        print(f"Target hostnames: {self.hostnames}")
        
        # Get all alive nodes and create hostname to node mapping
        alive_nodes = [node for node in ray.nodes() if node["Alive"]]
        hostname_to_node = {}
        
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            # Try to resolve hostname from IP or use IP directly
            try:
                import socket
                hostname = socket.gethostbyaddr(node_ip)[0]
            except:
                hostname = node_ip
            
            hostname_to_node[hostname] = node
            hostname_to_node[node_ip] = node  # Also map by IP
        
        print(f"Available nodes: {list(hostname_to_node.keys())}")
        
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        
        worker_id = 0
        successfully_created = 0
        
        for target_hostname in self.hostnames:
            print(f"\nCreating {self.workers_per_node} workers on {target_hostname}...")
            
            # Find matching node - improved matching logic
            matching_node = None
            for available_hostname, node in hostname_to_node.items():
                # Exact match
                if target_hostname == available_hostname:
                    matching_node = node
                    break
                # Check if target is substring of available (short hostname in FQDN)
                elif target_hostname in available_hostname:
                    matching_node = node
                    break
                # Check if available is substring of target (FQDN in short hostname)
                elif available_hostname in target_hostname:
                    matching_node = node
                    break
                # Extract short hostname from FQDN and compare
                elif '.' in available_hostname:
                    short_hostname = available_hostname.split('.')[0]
                    if target_hostname == short_hostname:
                        matching_node = node
                        break
            
            if not matching_node:
                print(f"  ❌ No matching node found for hostname {target_hostname}")
                # Add placeholder workers
                for i in range(self.workers_per_node):
                    self.workers.append(None)
                    worker_id += 1
                continue
            
            node_id = matching_node["NodeID"]
            node_ip = matching_node["NodeManagerAddress"]
            print(f"  Found matching node: {node_ip} (ID: {node_id})")
            
            hostname_workers = []
            for i in range(self.workers_per_node):
                try:
                    worker = TrainingWorker.options(
                        num_gpus=1,
                        num_cpus=1,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                    ).remote(worker_id=worker_id)
                    self.workers.append(worker)
                    hostname_workers.append(worker_id)
                    successfully_created += 1
                    print(f"    ✅ Created worker {worker_id}")
                    worker_id += 1
                except Exception as e:
                    print(f"    ❌ Failed to create worker {worker_id}: {str(e)}")
                    self.workers.append(None)
                    worker_id += 1
            
            if hostname_workers:
                self.hostname_to_workers[target_hostname] = hostname_workers
        
        # Filter out failed workers and collect info
        valid_workers = []
        valid_worker_info = {}
        
        print(f"\nCollecting worker information...")
        for i, worker in enumerate(self.workers):
            if worker is not None:
                try:
                    info = ray.get(worker.get_info.remote(), timeout=self.timeout)
                    new_id = len(valid_workers)
                    valid_workers.append(worker)
                    valid_worker_info[new_id] = info
                    print(f"  Worker {i} -> {new_id}: {info['hostname']} ({info['ip']})")
                except Exception as e:
                    print(f"  Worker {i}: Failed to get info - {str(e)}")
        
        self.workers = valid_workers
        self.worker_info = valid_worker_info
        
        print(f"\nSuccessfully initialized {len(self.workers)} workers")
        return len(self.workers)
    
    def run_ddp_training(self) -> Dict:
        """Run DDP training across all workers with enhanced debugging"""
        print(f"\n{'='*80}")
        print(f"Starting DDP Training with Enhanced Debugging")
        print(f"{'='*80}")
        
        if len(self.workers) < 2:
            return {
                "status": "failed",
                "error": "Need at least 2 workers for DDP training",
                "worker_count": len(self.workers)
            }
        
        world_size = len(self.workers)
        print(f"Training configuration:")
        print(f"  Total workers: {world_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Timeout: {self.timeout}s")
        print(f"  Debug level: {self.debug_level}")
        
        # Print worker distribution by hostname
        worker_by_hostname = defaultdict(list)
        for worker_id, info in self.worker_info.items():
            worker_by_hostname[info['hostname']].append(worker_id)
        
        print(f"\nWorker distribution:")
        for hostname, worker_ids in worker_by_hostname.items():
            print(f"  {hostname}: {len(worker_ids)} workers (IDs: {worker_ids})")
        
        try:
            # Get master address
            head_node_addr, port = ray.get(self.workers[0].get_address.remote())
            print(f"\nMaster address: {head_node_addr}:{port}")
            
            # Setup environment for all workers
            print(f"\nSetting up distributed environment...")
            setup_futures = []
            for rank, worker in enumerate(self.workers):
                setup_futures.append(
                    worker.setup_env_vars.remote(
                        addr=head_node_addr,
                        port=port,
                        world_rank=rank,
                        world_size=world_size,
                        debug_level=self.debug_level
                    )
                )
            
            ray.get(setup_futures, timeout=30)
            print(f"Environment setup completed for all workers")
            
            # Start DDP training
            print(f"\n🚀 Starting DDP training...")
            train_start_time = time.time()
            
            train_futures = []
            for worker in self.workers:
                train_futures.append(worker.train_ddp.remote(self.epochs, self.timeout))
            
            # Wait for training completion
            results = ray.get(train_futures, timeout=self.timeout + 60)
            total_time = time.time() - train_start_time
            
            # Analyze results
            successful_workers = [r for r in results if r["status"] == "success"]
            failed_workers = [r for r in results if r["status"] == "failed"]
            
            success = len(failed_workers) == 0
            
            print(f"\n{'='*80}")
            print(f"Training Results Summary")
            print(f"{'='*80}")
            print(f"Total training time: {total_time:.2f}s")
            print(f"Successful workers: {len(successful_workers)}/{world_size}")
            print(f"Failed workers: {len(failed_workers)}/{world_size}")
            
            if successful_workers:
                print(f"\n✅ Successful workers:")
                for r in successful_workers:
                    print(f"  Worker {r['worker_id']} (rank {r['rank']}) on {r['hostname']}: "
                          f"loss={r['final_loss']:.4f}, init_time={r.get('init_time', 0):.2f}s")
            
            if failed_workers:
                print(f"\n❌ Failed workers:")
                for r in failed_workers:
                    print(f"  Worker {r['worker_id']} (rank {r['rank']}) on {r['hostname']}: "
                          f"{r['error_type']} - {r['error']}")
            
            # Save detailed results
            detailed_results = {
                "training_success": success,
                "total_workers": world_size,
                "successful_count": len(successful_workers),
                "failed_count": len(failed_workers),
                "total_time": total_time,
                "hostnames": self.hostnames,
                "debug_level": self.debug_level,
                "worker_results": results,
                "worker_distribution": dict(worker_by_hostname)
            }
            
            # Save to JSON file
            timestamp = int(time.time())
            results_file = f"ddp_training_results_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(detailed_results, f, indent=2)
            
            print(f"\n💾 Detailed results saved to {results_file}")
            
            return detailed_results
            
        except Exception as e:
            print(f"\n❌ Training failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "worker_count": world_size
            }

def main():
    parser = argparse.ArgumentParser(description='Run DDP training on specific hostnames with enhanced debugging')
    parser.add_argument('--hostnames', type=str, nargs='+', required=True, 
                       help='List of hostnames to run DDP training on')
    parser.add_argument('--hostnames-file', type=str, 
                       help='File containing list of hostnames (one per line)')
    parser.add_argument('--workers-per-node', type=int, default=4, 
                       help='Number of workers per node (default: 4)')
    parser.add_argument('--epochs', type=int, default=2, 
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--timeout', type=float, default=60.0, 
                       help='Timeout in seconds for training (default: 60)')
    parser.add_argument('--debug-level', type=str, default="TRACE", 
                       choices=["TRACE", "INFO", "WARN"],
                       help='NCCL debug level (default: TRACE)')
    parser.add_argument('--ray-address', type=str, default=None, 
                       help='Ray cluster address')
    
    args = parser.parse_args()
    
    # Get hostnames from file or command line
    hostnames = []
    if args.hostnames_file:
        try:
            with open(args.hostnames_file, 'r') as f:
                hostnames = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading hostnames file: {e}")
            return
    else:
        hostnames = args.hostnames
    
    if not hostnames:
        print("Error: No hostnames provided")
        return
    
    print(f"Target hostnames: {hostnames}")
    
    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()
    
    print(f"Ray initialized successfully")
    
    try:
        # Create trainer and run
        trainer = HostnameDDPTrainer(
            hostnames=hostnames,
            workers_per_node=args.workers_per_node,
            epochs=args.epochs,
            timeout=args.timeout,
            debug_level=args.debug_level
        )
        
        # Initialize workers
        worker_count = trainer.initialize_workers()
        if worker_count == 0:
            print("❌ No workers could be initialized")
            return
        
        # Run training
        results = trainer.run_ddp_training()
        
        # Final summary
        if results.get("training_success"):
            print(f"\n🎉 DDP training completed successfully!")
        else:
            print(f"\n💥 DDP training failed")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()