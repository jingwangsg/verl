from argparse import ArgumentParser
import subprocess
import os
from .constants import POOLS
from .utils import red_print, map_async_with_thread, run_cmd


def list_matching_workflows(pool, status, job):
    workflow_names = []

    maybe_all_users = "--all-users" if ("longrun" in job or "obackground" in job) else ""

    outputs = (
        subprocess.run(
            f"osmo workflow list -c 1000 -s {status} {maybe_all_users} -p {pool} | grep {job} | awk '{{print $2}}'",
            shell=True,
            check=False,
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .split("\n")
    )
    workflow_names = [_ for _ in outputs if _]

    return workflow_names


def list_all_workflows(job):
    args = []
    pools = sorted(list(POOLS.values()))

    args += [(pool, status) for pool in pools for status in ["PENDING", "RUNNING"]]

    workflow_names = map_async_with_thread(
        iterable=args, func=lambda x: list_matching_workflows(x[0], x[1], job), verbose=False
    )
    # DEBUG
    # workflow_names = [list_matching_workflows(x[0], x[1], job) for x in args]

    # group by pool
    workflow_names_by_pool = {pool: [] for pool in pools}

    for pool_idx, pool in enumerate(pools):
        workflow_names_by_pool[pool] = (
            workflow_names[pool_idx * 2] + workflow_names[pool_idx * 2 + 1]
        )

    return workflow_names_by_pool


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--pool", type=str, help="Pool name", default=None)
    parser.add_argument("-j", "--job", type=str, help="name of the job", default="obackground")
    parser.add_argument("-n", "--num", type=int, help="number of jobs to cancel", default=None)
    args = parser.parse_args()

    if args.pool is None:
        red_print("You are about to cancel jobs in all pools")
        workflows_by_pool = list_all_workflows(args.job)
        to_cancel = []
        for pool, names in workflows_by_pool.items():
            if len(names) == 0:
                continue
            red_print(f"{pool}: {names}")
            if input("Continue? (y/n): ") == "y":
                to_cancel += names
                red_print(f"To cancel: {to_cancel}")

        # final yes
        if input("Start cancelling? (y/n): ") != "y":
            return

        to_cancel_bathces = [to_cancel[i : i + 8] for i in range(0, len(to_cancel), 8)]
        for to_cancel_batch in to_cancel_bathces:
            run_cmd(f"osmo workflow cancel {' '.join(to_cancel_batch)}", verbose=True)
    else:
        pool = POOLS[args.pool]
        workflow_names = map_async_with_thread(
            iterable=["PENDING", "RUNNING"],
            func=lambda x: list_matching_workflows(pool, x, args.job),
            verbose=False,
        )
        workflow_names = [item for sublist in workflow_names for item in sublist]
        red_print(f"{pool}: {workflow_names}")
        if input("Continue? (y/n): ") != "y":
            return
        to_cancel_bathces = [workflow_names[i : i + 8] for i in range(0, len(workflow_names), 8)]
        for to_cancel_batch in to_cancel_bathces:
            run_cmd(f"osmo workflow cancel {' '.join(to_cancel_batch)}", verbose=True)


if __name__ == "__main__":
    main()
