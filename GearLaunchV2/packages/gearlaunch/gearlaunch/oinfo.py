import os
import sys

sys.path.insert(0, os.getcwd())
from argparse import ArgumentParser

import pandas as pd

from .cached_cli import (
    osmo_cache_timestamp,
    osmo_pool_list,
    osmo_workflow_list,
    osmo_workflow_spec,
)
from .constants import POOLS
from .utils import green_print, map_async_with_thread, red_print, run_cmd, run_cmd_with_retry

pool_infos = [_.strip() for _ in osmo_pool_list().splitlines()]


def list_all_tasks(pool, user=None):
    all_users = False
    if user is None:
        all_users = True
    lines = osmo_workflow_list(
        pool,
        all_user=all_users,
        user=user,
    )

    lines = [line for line in lines.splitlines() if line.split()[-2] == "RUNNING"]

    # osmo only supports 1000 tasks for monitoring
    df = []
    for line in lines:
        user, task, _ = line.split(maxsplit=2)
        if task.startswith("obackground") or task.startswith("cancel_"):
            continue
        user = user.replace("@nvidia.com", "")
        df += [dict(user=user, task=task)]

    return pd.DataFrame(df)


def get_task_count_by_user(pool, user=None):
    df = list_all_tasks(pool, user=user)

    if len(df) == 0:
        return df

    def get_task_count(task):
        osmo_output = osmo_workflow_spec(task)

        # some workflow show no output, might be a bug in osmo
        if len(osmo_output.strip()) == 0:
            return 0

        gpu = 0
        node_count = 0
        for line in osmo_output.splitlines():
            if "gpu: " in line:
                gpu = int(line.split(":")[-1].strip())
            elif "- args:" in line:
                node_count += 1
        assert gpu <= 8, f"GPU count {gpu} exceeds 8."

        return node_count * gpu

    green_print([pool_info for pool_info in pool_infos if pool in pool_info][0])

    if user is not None:
        df = df[df["user"] == user]

    task_counts = map_async_with_thread(
        iterable=df["task"].values,
        func=get_task_count,
        verbose=False,
    )

    df["pool"] = pool
    df["#GPU"] = task_counts
    df["#GPU"] = df["#GPU"].astype(int)  # when task is not found, default value will be float

    df = df[df["#GPU"] > 0]

    return df


def get_args():
    parser = ArgumentParser(description="Parse arguments for oinfo.")
    parser.add_argument("-a", "--all", action="store_true", help="Show all information.")
    parser.add_argument("-u", "--user", type=str, help="Set the user.", default=os.getenv("USER"))
    parser.add_argument("--all_gpu", action="store_true", help="Show all GPU information (not just H100).", default=False)

    return parser.parse_args()


def oinfo():
    args = get_args()

    """
    ❯ gear info list --help
Usage: gear info list [OPTIONS]

  List workflows from the specified pool.

Options:
  -p, --pool TEXT                 The pool to list workflows from.
  --user TEXT                     The user for whom to list the workflows.
  --status [RUNNING|FAILED|COMPLETED|QUEUED|FAILED_TIMEOUT|FAILED_START_ERROR|FAILED_SERVER_ERROR|FAILED_BACKEND_ERROR|FAILED_QUEUE_TIMEOUT|FAILED_IMAGE_PULL|FAILED_UPSTREAM|FAILED_EVICTED|FAILED_START_ERROR|FAILED_CANCELED]
                                  The status of the workflows to list.
  --filter TEXT                   A regex filter to apply to the workflow
                                  names.
  --count INTEGER                 The maximum number of workflows to display.
  --help                          Show this message and exit.

  when --all is used, it will show the following information:
  for each pool:
    green_print <pool>
    gear info available -p <pool>
    gear info list -p <pool>
  all of below will be executed in parallel with multi-threading

  when --user is used, it will show the following information:
  for each pool:
    green_print <pool>
    gear info list --user <user>
  """
    pools = sorted(list(POOLS.values()))
    filter_pipe = "| grep -v '\-\-\-' | grep -v '===' | grep -v 'obackground'"

    if args.all:
        if not args.all_gpu:
            pools = [pool for pool in POOLS.values() if "h100" or "gb200" in pool.lower()]
        else:
            pools = POOLS.values()
        args = [
            (pool, cmd)
            for pool in pools
            for cmd in ["gear info available -p", "gear info list -p"]
        ]
        stdouts = map_async_with_thread(
            args,
            lambda x: run_cmd_with_retry(f"{x[1]} {x[0]} {filter_pipe}", max_retries=5).stdout,
            verbose=True,
            num_thread=8,
        )
        for i in range(0, len(stdouts), 2):
            green_print("=" * 20 + " " + pools[i // 2] + " " + "=" * 20)
            print(stdouts[i].replace("|", ""))
            print(stdouts[i + 1].replace("|", ""))
    elif args.user:
        args = [f"gear info list --user {args.user} -p {pool}" for pool in pools]
        stdouts = map_async_with_thread(
            args,
            lambda x: run_cmd_with_retry(f"{x} {filter_pipe}", max_retries=5).stdout,
            verbose=True,
            num_thread=8,
        )
        for i in range(0, len(stdouts)):
            green_print("=" * 20 + " " + pools[i] + " " + "=" * 20)
            print(stdouts[i].replace("|", ""))
    else:
        NotImplementedError("Please provide either --all or --user argument.")


if __name__ == "__main__":
    oinfo()
