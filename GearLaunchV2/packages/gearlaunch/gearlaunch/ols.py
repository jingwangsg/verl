import time
import re

import argparse
import os
from collections import OrderedDict

from .cached_cli import osmo_cache_timestamp, osmo_pool_list, osmo_workflow_list
from .constants import POOLS
from .utils import green_print, red_print, run_cmd_with_retry, map_async_with_thread


def get_args():
    parser = argparse.ArgumentParser(description="osmo list")
    parser.add_argument("-a", "--all-users", action="store_true", help="List all users")
    parser.add_argument("-u", "--user", help="List workflows for a specific user", default=None)
    parser.add_argument("-r", "--regex", help="Regex to filter workflows", default=None)
    return parser.parse_args()


# def ols():
#     args = get_args()
#     timestamp = osmo_cache_timestamp()
#     print(f"Last updated: {timestamp}")

#     pool_infos = [_.strip() for _ in osmo_pool_list().splitlines()]
#     for pool in sorted(list(POOLS.keys())):
#         pool = POOLS[pool]
#         flags = []
#         if args.all_users:
#             flags.append("--all-users")
#         if args.user:
#             flags.append(f"--user {args.user}")

#         pool_info = [pool_info for pool_info in pool_infos if pool in pool_info]
#         if len(pool_info) == 0:
#             red_print(f"Pool {pool} not found in pool list")
#             continue
#         pool_info = pool_info[0].strip()
#         workflows = osmo_workflow_list(pool, all_user=args.all_users, user=args.user)
#         green_print(pool_info)

#         if workflows == "":
#             continue

#         print(workflows)
#         print()


def ols():
    args = get_args()
    flags = []
    if args.all_users:
        flags += ["--all-users"]
    if args.user:
        flags += [f"--user {args.user}"]

    flags = " ".join(flags)

    pools = sorted(list(POOLS.values()))

    if args.all_users:
        task_infos = map_async_with_thread(
            pools,
            lambda pool: run_cmd_with_retry(
                f"osmo task list {flags} -c 9999 -s RUNNING INITIALIZING SCHEDULING PROCESSING -p {pool} | grep nvidia || true",
                max_retries=5,
            ),
            verbose=False,
        )
        lines = []
        for task_info in task_infos:
            lines += [line for line in task_info.stdout.splitlines()]
    else:
        task_info = run_cmd_with_retry(
            f"osmo task list {flags} -c 9999 -s RUNNING INITIALIZING SCHEDULING PROCESSING | grep nvidia || true",
            max_retries=5,
        )
        lines = [line for line in task_info.stdout.splitlines()]

    lines = [line for line in lines if "obackground" not in line]

    df = dict()
    for line in lines:
        user, workflow_name, task, status, pool, node, date = line.split(maxsplit=6)
        if args.regex and not re.match(args.regex, workflow_name):
            continue
        df[pool] = df.get(pool, []) + [
            {
                "user": user,
                "workflow_name": workflow_name,
                "task": task,
                "status": status,
                "pool": pool,
                "node": node,
                "date": date,
            }
        ]

    pool_infos = [_.strip() for _ in osmo_pool_list().splitlines()]
    for pool in sorted(list(POOLS.keys())):
        pool = POOLS[pool]
        pool_info = [pool_info for pool_info in pool_infos if pool in pool_info]
        if len(pool_info) == 0:
            red_print(f"Pool {pool} not found in pool list")
            continue
        pool_info = pool_info[0].strip()
        green_print(pool_info)

        cur_df = df.get(pool, [])
        if len(cur_df) == 0:
            continue

        unique_df = OrderedDict()
        for item in cur_df:
            item.pop("node")
            item.pop("task")
            workflow_name = item["workflow_name"]
            if workflow_name not in unique_df:
                unique_df[workflow_name] = item
                unique_df[workflow_name]["nodes"] = 1
            else:
                unique_df[workflow_name]["nodes"] += 1

        # print it like a table
        for item in unique_df.values():
            user = item["user"].split("@")[0]
            print(
                f"{user:10} {item['workflow_name']:40} {item['status']:10} {item['pool']:15} {str(item['nodes'])+'Nodes':15} {item['date']:10}"
            )
        print()
