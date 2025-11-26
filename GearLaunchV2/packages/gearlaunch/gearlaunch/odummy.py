import os
from typing import Optional
import numpy as np
import argparse

from .constants import RESOURCE_LIMIT
from .utils import run_cmd
from .orun import orun
import time
from tqdm import tqdm


import threading

file_path = __file__


def odummy(pool: Optional[str] = None, interval: int = 10, once=False,):
    """
    Launch a background job on each node in the cluster.
    """

    while True:
        infos = (
            run_cmd(f"osmo resource list -p {pool} | grep RESERVED | awk '{{print $8}}'")
            .stdout.strip()
            .splitlines()
        )
        num_gpus_used = np.array([int(info.split("/")[0]) for info in infos])
        num_gpus_total = np.array([int(info.split("/")[1]) for info in infos])

        # take up idle nodes
        num_idle_nodes = np.sum(num_gpus_used == 0)
        idle_ratio = num_idle_nodes / len(num_gpus_used)
        gpu_count_per_node = RESOURCE_LIMIT[pool]["gpu"]
        num_gpus_on_idle_nodes = gpu_count_per_node * num_idle_nodes

        if 512 <= num_gpus_on_idle_nodes:
            num_nodes_per_job = 128 // gpu_count_per_node
        elif 256 <= num_gpus_on_idle_nodes < 512:
            num_nodes_per_job = 64 // gpu_count_per_node
        elif 128 <= num_gpus_on_idle_nodes < 256:
            num_nodes_per_job = 32 // gpu_count_per_node
        elif 64 <= num_gpus_on_idle_nodes < 128:
            num_nodes_per_job = 16 // gpu_count_per_node
        else:
            num_nodes_per_job = 8 // gpu_count_per_node

        print(f"[{pool}] Idle Ratio: {idle_ratio} | Num Nodes Per Job: {num_nodes_per_job}")

        # ! deprecated, too many failed queue timeout
        # fill up non-idle nodes if necessary
        # here we have to set 3 variables: num_idle_nodes, num_nodes_per_job, gpu_count_per_node
        # if num_idle_nodes == 0:
        #     num_idle_nodes = np.sum(num_gpus_total - num_gpus_used < gpu_count_per_node // 2)
        #     num_nodes_per_job = 4
        #     gpu_count_per_node = gpu_count_per_node // 2

        num_jobs = (num_idle_nodes + num_nodes_per_job - 1) // num_nodes_per_job

        print(
            f"[{pool}] Num Jobs to Launch: {num_jobs} | Num GPUs Available: {num_gpus_on_idle_nodes}"
        )
        # get the str directory of this file
        directory = os.path.dirname(file_path)
        workflow_path = os.path.join(directory, "..", "workflow_spec", "dummy.yaml")

        for _ in tqdm(range(num_jobs), desc=f"[{pool}] Launching Jobs"):
            num_nodes = min(num_nodes_per_job, num_idle_nodes - _ * num_nodes_per_job)
            while True:
                try:
                    orun(
                        name=f"longrun-vilavideo_training_g{gpu_count_per_node*num_nodes}",
                        nodes=num_nodes,
                        gpu=gpu_count_per_node,
                        cpu=40,
                        memory="256Gi",
                        storage="256Gi",
                        pool=pool,
                        workdir="/mnt/amlfs-03/shared/jingwang/WORKSPACE/GearLaunchV2",
                        queue_timeout=f"{interval}m",
                    )
                    break
                except Exception as e:
                    print(f"Failed to launch job on {pool} due to: {e}")
                    time.sleep(30)
        
        if once:
            break

        for i in tqdm(
            range(interval + 5), desc=f"[{pool}] Sleeping for {interval + 5} minutes"
        ):  # queue_timeout=10m, 5min waiting for resource release
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h100", action="store_true", help="Launch on h100 pool.")
    parser.add_argument("--l40", action="store_true", help="Launch on l40 pool.")
    parser.add_argument("--interval", type=int, default=10, help="Interval between each check.")
    parser.add_argument("--once", action="store_true", help="Run once.")
    args = parser.parse_args()
    # launch_background_jobs()  # launch on all pools
    if args.h100:
        threading.Thread(target=odummy, args=("groot-h100-01", args.interval, args.once)).start()
        threading.Thread(target=odummy, args=("groot-h100-02", args.interval, args.once)).start()

    if args.l40:
        threading.Thread(target=odummy, args=("groot-l40-01", args.interval, args.once)).start()
        threading.Thread(target=odummy, args=("groot-l40-02", args.interval, args.once)).start()
        threading.Thread(target=odummy, args=("groot-l40-03", args.interval, args.once)).start()


if __name__ == "__main__":
    main()
