import argparse
import os
import os.path as osp
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.job_submission import JobSubmissionClient
import subprocess
import time
import uuid


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--nnodes", type=int, default=1, help="Number of nodes, default is 1."
    )
    parser.add_argument(
        "-g",
        "--ngpus",
        type=int,
        default=8,
        help="Number of GPUs per task, default is 8.",
    )
    parser.add_argument(
        "--ncpus", type=int, default=8, help="Number of CPUs per task, default is 8."
    )
    parser.add_argument(
        "-w",
        "--workdir",
        type=str,
        default=os.getcwd(),
        help="Working directory, default is the current working directory.",
    )
    parser.add_argument(
        "--address", type=str, default="auto", help="Ray address, default is auto."
    )
    parser.add_argument(
        "--env_vars",
        nargs="+",
        default=[],
        help="Environment variables, default is empty.",
    )
    parser.add_argument(
        "--submission_id",
        type=str,
        default="submission",
        help="Submission ID, default is submission.",
    )
    parser.add_argument(
        "--venv", type=str, default=None, help="Virtual environment, default is None."
    )
    parser.add_argument(
        "--max_retries", type=int, default=1, help="Maximum retries, default is 1."
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port set in environment variable, default is 29500.",
    )
    parser.add_argument("--cmd", nargs="+", required=True, help="Command to run.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    workdir = args.workdir

    cmd = " ".join(args.cmd)
    print(f"Running command: {cmd} under {args.workdir} on {args.nnodes} nodes with {args.ngpus} GPUs and {args.ncpus} CPUs per node")

    # setup environment variables
    env_vars = {kv_str.split("=")[0]: kv_str.split("=")[1] for kv_str in args.env_vars}
    print(f"Environment variables:")
    for k, v in env_vars.items():
        print(f"{k}={v}")

    # prepare for client submission
    client = JobSubmissionClient(address="auto")
    rayexec_entry_path = osp.join(osp.dirname(__file__), "rayexec_entry.py")
    entrypoint = f'python {rayexec_entry_path} --address {args.address} --ngpus {args.ngpus} --ncpus {args.ncpus} --master_port {args.master_port} --max_retries {args.max_retries} --nnodes {args.nnodes} --workdir {workdir} --cmd "{cmd}"'

    dummy_workdir = "/workspace/dummy_workdir"
    os.makedirs(dummy_workdir, exist_ok=True)

    submission_id = f"{args.submission_id}_{uuid.uuid4().hex[:8]}"

    runtime_env = {
        "env_vars": env_vars,
        "working_dir": dummy_workdir,
    }

    # run code in virtual environment if provided
    if args.venv:
        env_vars["PATH"] = f"{args.venv}/bin:$PATH"
        if "PATH" not in env_vars:
            runtime_env["env_vars"]["PATH"] = f"{args.venv}/bin:$PATH"
        else:
            runtime_env["env_vars"]["PATH"] = f"{args.venv}/bin:{env_vars['PATH']}"
        runtime_env["py_executable"] = f"{args.venv}/bin/python"

    # submit job
    submission_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        submission_id=submission_id,
    )

    print(f"Job submitted with submission ID: {submission_id}")


if __name__ == "__main__":
    main()
