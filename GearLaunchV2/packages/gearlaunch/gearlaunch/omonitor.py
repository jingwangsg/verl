import argparse
import datetime
import os
import os.path as osp
import tempfile
import time

import pytz
import multiprocessing as mp
from tqdm import tqdm

from .constants import CACHE_ROOT, POOLS
from .utils import run_cmd, map_async_with_thread, red_print, green_print
from .okill import okill


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--autokill", action="store_true", default=False)
    return parser.parse_args()


def log_to_file(file, cmd, skip_exists=False, async_cmd=True, retry=False):
    os.makedirs(osp.dirname(file), exist_ok=True)
    if retry:
        assert not async_cmd, "async_cmd is not supported with retry"

    if skip_exists and is_valid_cache(file):
        return

    while True:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".log",
                delete=False,
            ) as f:
                run_cmd(
                    f"{cmd} > {f.name} && mv {f.name} {file}",
                    async_cmd=async_cmd,
                )
            break
        except Exception as e:
            time.sleep(3)
            red_print(f"Failed to run {cmd} due to: {e}")
            if not retry:
                break


def is_valid_cache(path):
    is_valid = osp.exists(path) and len(open(path).read().splitlines()) > 1
    return is_valid


def verify_outputs(lines):
    assert "Server responded with status code" not in lines, "Server responded with status code"
    assert "No tasks were found" not in lines, "No tasks were found"


def omonitor_spec(interval=10):
    log_dir = CACHE_ROOT

    while True:
        workflow_ids_all = []
        for pool in POOLS.values():
            workflow_ids = run_cmd(
                f"osmo workflow list --count 4000 --status RUNNING --pool {pool} --all-users | grep '@' | grep -v 'wenlix@nvidia.com' | awk '{{print $2}}'",
            ).stdout.splitlines()

            workflow_ids = [
                workflow_id.strip() for workflow_id in workflow_ids if workflow_id.strip() != ""
            ]

            workflow_ids_all += workflow_ids

            for workflow_id in tqdm(workflow_ids, desc=f"[spec] Fetching specs in {pool}"):
                if workflow_id.startswith("cancel_") or workflow_id.startswith("obackground"):
                    continue
                workflow_spec_cache = osp.join(log_dir, "workflows", f"{workflow_id}")
                os.makedirs(osp.dirname(workflow_spec_cache), exist_ok=True)

                if is_valid_cache(workflow_spec_cache):
                    continue

                while True:
                    try:
                        content = run_cmd(
                            f"osmo workflow spec {workflow_id}", fault_tolerance=True
                        ).stdout
                        verify_outputs(content)
                        break
                    except Exception as e:
                        time.sleep(5)

                with open(workflow_spec_cache, "w") as f:
                    f.write(content)

        # delete redundant files
        files_to_delete = []
        workflow_files = os.listdir(osp.join(log_dir, "workflows"))
        files_to_delete += [
            osp.join(log_dir, "workflows", _) for _ in (set(workflow_files) - set(workflow_ids_all))
        ]

        chunk_size = 100
        file_chunks_to_delete = [
            files_to_delete[i : i + chunk_size] for i in range(0, len(files_to_delete), chunk_size)
        ]
        for file_chunk_to_delete in file_chunks_to_delete:
            run_cmd(f"rm -rf {' '.join(file_chunk_to_delete)}", async_cmd=True)

        time.sleep(interval)


def omonitor_task(interval=10, autokill=False):
    while True:
        for pool in POOLS.values():
            users = run_cmd(
                f"osmo workflow list -p {pool} -s RUNNING PENDING -c 4000 -a | grep @ | awk '{{print $1}}' | sort | uniq"
            ).stdout.splitlines()

            # * only care about user below
            users = [user for user in users if user in ("jingwang@nvidia.com", "aoz@nvidia.com")]

            task_infos = []

            for user in tqdm(users, desc=f"[task] Fetching tasks in {pool}"):

                lines = run_cmd(
                    f"osmo task list -c 4000 -u {user} -p {pool} | grep '@'",
                    fault_tolerance=True,
                ).stdout
                verify_outputs(lines)
                lines = lines.splitlines()

                task_infos += [line for line in lines]
                print(f"Autokill: {autokill}")

                # * autokill jobs if my job is queueing
                if autokill:
                    assigned_nodes = [line.split()[5].strip() for line in lines]
                    num_queued = len([node for node in assigned_nodes if node == "-"])
                    if num_queued > 0:
                        red_print(f"Found {num_queued} nodes in queue for {user}")
                        red_print(f"Autokilling jobs in {pool} for {user}")
                        okill(pool, "obackground", yes=True, num=num_queued * 2)

            with open(osp.join(CACHE_ROOT, pool, "task.log"), "w") as f:
                f.write("\n".join(task_infos))

        time.sleep(interval)


def omonitor_basic(interval=5):
    log_dir = CACHE_ROOT

    print(f"Monitoring osmo status to {log_dir}")

    while True:
        for pool in POOLS.values():
            cur_log_dir = osp.join(log_dir, pool)
            log_to_file(
                osp.join(cur_log_dir, "workflow.log"),
                f"osmo workflow list --count 4000 --status PENDING RUNNING --pool {pool} --all-users",
            )

            log_to_file(
                osp.join(cur_log_dir, "resource.log"),
                f"osmo resource list --pool {pool}",
            )

        log_to_file(
            osp.join(log_dir, "pool.log"),
            "osmo pool list",
        )
        timestamp = datetime.datetime.now(pytz.timezone(os.getenv("TZ", "US/Pacific")))
        open(osp.join(log_dir, "timestamp.log"), "w").write(str(timestamp))
        time.sleep(interval)
        print(f"Last updated: {timestamp}")


def omonitor(interval=5, autokill=False):
    mp.Process(target=omonitor_basic, args=(interval,)).start()
    mp.Process(target=omonitor_spec, args=(interval,)).start()
    mp.Process(target=omonitor_task, args=(interval, autokill)).start()


def main():
    args = get_args()
    omonitor(interval=args.interval, autokill=args.autokill)
