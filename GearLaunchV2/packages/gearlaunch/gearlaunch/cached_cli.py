import os.path as osp

from .constants import CACHE_ROOT
from .utils import run_cmd

# FROM_CACHE = osp.isdir(CACHE_ROOT)
FROM_CACHE = None


def osmo_cache_timestamp():
    if not FROM_CACHE:
        return "No cache found"
    return open(osp.join(CACHE_ROOT, "timestamp.log")).readlines()[0]


def osmo_workflow_list(pool, all_user=False, user=None, from_cache=FROM_CACHE):
    # remove table header using grep @
    cache_file = osp.join(CACHE_ROOT, pool, "workflow.log")
    if not (osp.exists(cache_file) and from_cache):
        flags = []
        if all_user:
            flags.append("--all-users")
        if user:
            flags.append(f"--user {user}")
        return run_cmd(
            f"osmo workflow list --status PENDING RUNNING --pool {pool} {' '.join(flags)} | grep '@'",
            fault_tolerance=True,
        ).stdout
    lines = open(cache_file).readlines()
    lines = [line.strip() for line in lines if "@" in line]
    if all_user:
        return "\n".join(lines)
    elif user is not None:
        lines = [line for line in lines if user in line]
        return "\n".join(lines)
    else:
        NotImplementedError


def osmo_pool_list(from_cache=FROM_CACHE):
    cache_file = osp.join(CACHE_ROOT, "pool.log")
    if not (osp.exists(cache_file) and from_cache):
        return run_cmd("osmo pool list").stdout
    return open(cache_file).read()


def osmo_workflow_spec(workflow_id, from_cache=FROM_CACHE):
    cache_file = osp.join(CACHE_ROOT, "workflows", f"{workflow_id}")
    if not (osp.exists(cache_file) and from_cache):
        return run_cmd(f"osmo workflow spec {workflow_id}").stdout
    return open(cache_file).read()


def osmo_task(pool, from_cache=FROM_CACHE):
    cache_file = osp.join(CACHE_ROOT, pool, "tasks.log")
    if not (osp.exists(cache_file) and from_cache):
        return run_cmd(
            f"osmo task list -c 3000 -s RUNNING QUEUED --all-users -p {pool} | grep '@'",
        ).stdout
    return open(cache_file).read()
