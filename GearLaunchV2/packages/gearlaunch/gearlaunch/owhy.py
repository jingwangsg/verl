import pandas as pd
import argparse
from .utils import run_cmd, red_print
from .cached_cli import osmo_task


def task_info_parser(task_info):
    df_dict = []
    for line in task_info.splitlines():
        # jingwang@.com   cogx_mem_lr3e-4_7b24a54-12   worker_9    QUEUED   groot-h100-01   aks-gpu-75821773-vmss00007e   -
        user, workflow_id, task_id, status, pool, node, timestamp = line.split(maxsplit=6)
        df_dict += [
            dict(
                user=user,
                workflow_id=workflow_id,
                task_id=task_id,
                status=status,
                pool=pool,
                node=node,
                timestamp=timestamp,
            )
        ]

    return pd.DataFrame(df_dict)


def owhy(workflow_id):
    pool = run_cmd(
        f"osmo task list -c 1000 --workflow-id {workflow_id} -a -v | grep '@' | awk '{{print $5}}'"
    ).stdout.strip()
    if pool == "":
        red_print(f"Workflow {workflow_id} not found.")
        return
    pool = pool.splitlines()[0]

    pool_info = osmo_task(pool)

    cur_workflow_info = "\n".join([line for line in pool_info.splitlines() if workflow_id in line])
    pool_info = "\n".join([line for line in pool_info.splitlines() if workflow_id not in line])

    workflow_df = task_info_parser(cur_workflow_info)
    workflow_df = workflow_df.sort_values(by="task_id", ascending=False)
    pool_df = task_info_parser(pool_info)

    blocker_df = (
        pool_df.groupby(["node"])
        .agg({"workflow_id": list})
        .rename(columns={"workflow_id": "blocker_workflows"})
        .reset_index()
    )
    workflow_df["node"] = workflow_df["node"].replace("-", "Not Scheduled")
    merged_df = pd.merge(workflow_df, blocker_df, on="node", how="left")

    print(merged_df.to_markdown(index=False))

    blocker_workflows = list(
        filter(lambda x: isinstance(x, list), merged_df["blocker_workflows"].tolist())
    )
    blocker_workflows = [item for sublist in blocker_workflows for item in sublist]
    blocker_workflows = list(set(blocker_workflows))
    if len(blocker_workflows) > 0:
        red_print(f"Cancelling {len(blocker_workflows)} jobs: {blocker_workflows}")
        prompt = input("Continue to kill blockers? (y/n)")
        if prompt == "y":
            run_cmd(f"osmo workflow cancel --force {' '.join(blocker_workflows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_id", type=str)
    args = parser.parse_args()

    owhy(args.workflow_id)
