import os
import os.path as osp
import sys

sys.path.insert(0, os.getcwd())
import argparse
import os
import sys
import time
import numpy as np
from loguru import logger

from .constants import DEFAULT_POOL, POOLS, RESOURCE_LIMIT
from .utils import green_print, run_cmd, red_print
from .ols import osmo_workflow_list


def get_args():

    parser = argparse.ArgumentParser(description="Parse arguments for owruns.")

    parser.add_argument("--name", default="debug", help="Set the workflow name.")
    parser.add_argument("--user", default=os.getenv("USER"), help="Set the username.")
    parser.add_argument("-g", "--gpu", type=int, default=8, help="Set the number of GPUs.")
    parser.add_argument("-c", "--cpu", type=int, default=80, help="Set the number of CPUs.")
    parser.add_argument("-n", "--nodes", type=int, default=1, help="Set the number of nodes.")
    parser.add_argument("-m", "--memory", default="1536Gi", help="Set the memory allocation.")
    parser.add_argument("-s", "--storage", default="768Gi", help="Set the storage size.")
    parser.add_argument("--anonymous", action="store_true", help="Set the anonymous flag.")
    parser.add_argument(
        "--image", default="nvcr.io/nvidian/gear-trinity-train:latest", help="Set the image."
    )
    # parser.add_argument("--image", default="nvcr.io/nvidian/gear-n2-eval:latest", help="Set the image.")

    parser.add_argument(
        "-p",
        "--pool",
        default=DEFAULT_POOL,
        help="Set the osmo pool.",
        choices=list(POOLS.keys()),
    )
    parser.add_argument("-u", "--upload_pdx", action="store_true", help="Upload code to PDX.")
    parser.add_argument("--workdir", default=os.getcwd(), help="Set the working directory.")
    parser.add_argument("--exclude", nargs="+", help="Set the exclude patterns.", default=[])
    parser.add_argument("--link", nargs="+", help="Set the link patterns.", default=[])
    parser.add_argument(
        "--cmd",
        nargs=argparse.REMAINDER,
        default=["echo 'Container ready for debugging'; sleep infinity"],
        help="Command to execute.",
    )

    args = parser.parse_args()

    args.workdir = osp.abspath(args.workdir)

    args.cmd = " ".join(args.cmd)
    args.pool = POOLS[args.pool]

    return args


def get_staged_path(workflow_id):
    if osp.exists("/mnt/amlfs-02/"):
        return "/mnt/amlfs-02/shared/osmo_staging/{}".format(workflow_id)
    elif osp.exists("/mnt/aws-lfs-01/"):
        return "/mnt/aws-lfs-01/shared/osmo_staging/{}".format(workflow_id)
    else:
        raise ValueError("No staging path found")


def get_dataset_name(workflow_id):
    return f"repo-{workflow_id}"


def stage_repo(workflow_id, exclude=[], link=[]):
    staged_path = get_staged_path(workflow_id)

    os.makedirs(staged_path, exist_ok=True)

    # cmd = " ".join(
    #     [
    #         "find . -type f",
    #         "-not -path '*/.*/*'",
    #         "-not -name '.*'",
    #         "-not -path '*/*.egg-info/*'",
    #         "-not -path '*/__pycache__/*'",
    #         "-not -path '*/checkpoint*/*'",
    #         "-not -path '*/wandb/*'",
    #         "-not -name 'core.*'",
    #         f"-print0 | rsync --files-from=- --from0 ./ {staged_path}",
    #     ]
    # )
    exclude_args = [
        "--exclude '.*'",
        "--exclude '*.egg-info/*'",
        "--exclude '__pycache__/*'",
        "--exclude 'wandb/*'",
        "--exclude 'core.*'",
        "--exclude 'libs'",
    ]
    for exclude_i in exclude + link:
        exclude_args.append(f"--exclude '{exclude_i}'")

    cmd = " ".join(
        [
            "fd . --type f",
            *exclude_args,
            f"--print0 | rsync --files-from=- --from0 ./ {staged_path}",
        ]
    )
    run_cmd(cmd)

    # prevent .gitignore ignoring these configs
    cmd = " ".join(
        [
            "fd . -e json -e yaml -e yml",
            "--unrestricted",
            *exclude_args,
            f"--print0 | rsync --files-from=- --from0 ./ {staged_path}",
        ]
    )
    run_cmd(cmd)

    cmd = f"cp -r .git {staged_path}"
    run_cmd(cmd, fault_tolerance=True)

    cmd = f"cp .gitignore {staged_path}"
    run_cmd(cmd, fault_tolerance=True)

    run_cmd(f"touch {staged_path}/.staged")
    green_print(f"Staged code at {staged_path}")
    return staged_path


def plan_resource_by_gpu(gpu, cpu, memory, storage, pool):
    pool_fullname = pool

    gpu = min(RESOURCE_LIMIT[pool_fullname]["gpu"], gpu)

    memory = min(RESOURCE_LIMIT[pool_fullname]["memory"] - 10, int(memory.split("Gi")[0]))
    memory = f"{memory}Gi"

    cpu = min(RESOURCE_LIMIT[pool_fullname]["cpu"] - 8, cpu)
    cpu = cpu

    storage = min(RESOURCE_LIMIT[pool_fullname]["storage"] - 10, int(storage.split("Gi")[0]))
    storage = f"{storage}Gi"

    return gpu, cpu, memory, storage


def fix_workdir_if_not_exist(workdir, user):
    if not osp.exists("/mnt/amlfs-01/"):
        # not on lustre
        return workdir
    if not osp.exists(workdir):
        workdir = f"/mnt/amlfs-03/shared/{user}"
    return workdir


def orun(
    name="debug",
    user=os.getenv("USER"),
    gpu=8,
    cpu=80,
    nodes=1,
    memory="1536Gi",
    storage="768Gi",
    image="nvcr.io/nvidian/gear-trinity-train:latest",
    pool=DEFAULT_POOL,
    upload_pdx=False,
    workdir=os.getcwd(),
    exclude=[],
    link=[],
    cmd="echo 'Container ready for debugging'; sleep infinity",
):
    # print(f"Excluding aks-gpu-24770890-vmss000064")
    if os.getenv("SKIP_SAME_NAME"):
        workflow_list = []
        for p in POOLS.values():
            workflow_list += osmo_workflow_list(p, user=user).splitlines()

        workflow_names = set([line.split()[1].rsplit("-", 1)[0] for line in workflow_list])
        if name in workflow_names:
            red_print(f"Workflow name {name} already exists. Skipping.")
            return

    gpu, cpu, memory, storage = plan_resource_by_gpu(gpu, cpu, memory, storage, pool=pool)
    workdir = fix_workdir_if_not_exist(workdir, user)
    dryrun = eval(os.getenv("DRYRUN", "0"))

    output = "\n".join(
        [
            f"Name:          {name}",
            f"User:          {user}",
            f"GPU:           {gpu}",
            f"CPU:           {cpu}",
            f"Pool:          {pool}",
            f"Nodes:         {nodes}",
            f"Memory:        {memory}",
            f"Image:         {image}",
            f"Storage:       {storage}",
            f"Workdir:       {workdir}",
            f"Upload PDX:    {upload_pdx}",
            f"Link:          {';'.join(link)}",
            f"Exclude:       {';'.join(exclude)}",
        ]
    )

    cmd_verbose = "\n".join([_.strip() for _ in cmd.split(";")])

    green_print(output)
    green_print(
        "\n".join(
            [
                "======================= COMMAND ========================",
                cmd_verbose,
                "========================================================",
            ]
        )
    )

    # Construct osmo workflow submission
    if pool == "groot-l40s-01":
        WORKFLOW_PATH = osp.abspath(osp.join(osp.dirname(__file__), "..", "workflow_spec/default.yaml"))
        homedir=f"/mnt/lfs/home/{user}"
    if pool == "groot-l40-04":
        WORKFLOW_PATH = osp.abspath(osp.join(osp.dirname(__file__), "..", "workflow_spec/default.yaml"))
        homedir=f"/mnt/amlfs-04/home/{user}"
    elif pool in ("groot-gb200-01"):
        WORKFLOW_PATH = osp.abspath(osp.join(osp.dirname(__file__), "..", "workflow_spec/default.yaml"))
        homedir = f"/mnt/aws-lfs-01/shared/{user}"
    else:
        WORKFLOW_PATH = osp.abspath(osp.join(osp.dirname(__file__), "..", "workflow_spec/default.yaml"))
        homedir=f"/mnt/amlfs-03/shared/{user}"

    job_cmd = " ".join(
        [
            f"osmo workflow submit --pool {pool} {WORKFLOW_PATH} --set",
            f"user={user} gpu={gpu} cpu={cpu} memory={memory} nodes={nodes}",
            f"storage={storage} image={image} homedir={homedir}",
            f'cmd="{cmd}" workdir={workdir} name={name} upload_pdx={upload_pdx}',
            f"link='{' '.join(link)}'" ,
        ]
    )

    logger.debug(job_cmd)

    while True and not dryrun:
        try:
            popen = run_cmd(job_cmd, verbose=True, fault_tolerance=False)
            break
        except Exception as e:
            red_print(e)
            time.sleep(30)
    
    if not dryrun:
        workflow_id = [line for line in popen.stdout.splitlines() if "Workflow ID" in line][0].split()[
            -1
        ]

    if upload_pdx and not dryrun:
        stage_repo(workflow_id, exclude=exclude, link=link)


def main():
    args = get_args()

    orun(
        name=args.name,
        user=args.user,
        gpu=args.gpu,
        cpu=args.cpu,
        nodes=args.nodes,
        memory=args.memory,
        storage=args.storage,
        image=args.image,
        pool=args.pool,
        upload_pdx=args.upload_pdx,
        exclude=args.exclude,
        link=args.link,
        workdir=args.workdir,
        cmd=args.cmd,
    )


if __name__ == "__main__":
    main()
