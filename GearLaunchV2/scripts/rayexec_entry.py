import argparse
import os
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.train.torch.train_loop_utils import get_device
import subprocess
import time


@ray.remote
class _RayWorker:
    def get_address(self):
        from ray.train._internal.utils import get_address_and_port

        return get_address_and_port()

    def setup_env_vars(self, addr, port, node_rank, nnodes):
        # HERE: Set all important environment variables in this function
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)
        os.environ["NODE_RANK"] = str(node_rank)
        os.environ["NNODES"] = str(nnodes)

    def run_remote_cmd(self, cmd, workdir):
        try:
            subprocess.run("&&".join([f"cd {workdir}", cmd]), shell=True)
        except Exception as e:
            if any(
                [
                    "ECC" in str(e) or "CUDA driver initialization failed" in str(e),
                    "c10d/NCCLUtils.hpp" in str(e),
                ]
            ):
                # Force stop Ray if we encounter GPU errors
                print(e)
                subprocess.run(["ray", "stop", "--force"], check=False)
            else:
                raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, default="auto")
    parser.add_argument("-n", "--nnodes", type=int, default=1)
    parser.add_argument("-g", "--ngpus", type=int, default=8)
    parser.add_argument("--ncpus", type=int, default=8)
    parser.add_argument("-w", "--workdir", type=str, default=os.getcwd())
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--cmd", nargs="+")
    args = parser.parse_args()

    cmd = " ".join(args.cmd)
    ray.init(address=args.address, ignore_reinit_error=True)

    for i in range(args.max_retries):
        try:
            print(f"Trying {i+1}/{args.max_retries} to run command: {cmd}")
            bundle = {"CPU": args.ncpus, "GPU": args.ngpus}
            print(f"Bundle: {bundle}")
            placement_group = ray.util.placement_group([bundle] * args.nnodes, strategy="SPREAD")

            # Wait for placement group to be ready
            ray.get(placement_group.ready())

            workers = [
                _RayWorker.options(
                    num_cpus=bundle.get("CPU", 0),
                    num_gpus=bundle.get("GPU", 0),
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=i,
                    ),
                ).remote()
                for i, bundle in enumerate(placement_group.bundle_specs)
            ]
            head_node_addr, port = ray.get(workers[0].get_address.remote())
            print(f"Head node address: {head_node_addr}, port: {port}")
            ray.get(
                [
                    worker.setup_env_vars.remote(
                        head_node_addr, args.master_port, node_rank, args.nnodes
                    )
                    for node_rank, worker in enumerate(workers)
                ]
            )

            ray.get([worker.run_remote_cmd.remote(cmd, args.workdir) for worker in workers])

            exit(0)
        except Exception as e:
            print(f"Error on retry {i+1}/{args.max_retries}: {e}")
            print("Removing placement group and killing workers")

            try:
                if "workers" in locals():
                    for worker in workers:
                        try:
                            ray.kill(worker)
                        except:
                            pass
                    del workers

                if "placement_group" in locals():
                    try:
                        ray.util.remove_placement_group(placement_group)
                        del placement_group
                    except:
                        pass

            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")

            time.sleep(30)
            if i == args.max_retries:
                raise e
