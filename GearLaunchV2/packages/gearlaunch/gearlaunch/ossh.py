from .orun import orun
import os
import os.path as osp
import argparse
from .constants import POOLS


def ossh(
    pool,
    gpu=False,
):

    script_path = osp.join(
        "/mnt/amlfs-03/shared/jingwang", "WORKSPACE", "GearLaunchV2", "scripts", "setup_entry.sh"
    )

    # gpu = 8 if gpu else 0
    gpu = 8 # always use gpu

    orun(
        name="entry",
        user=os.getenv("USER"),
        workdir="/mnt/amlfs-03/shared/jingwang/",
        gpu=gpu,
        cpu=120,
        memory="1000Gi",
        storage="6300Gi",
        pool=pool,
        cmd=f"bash {script_path}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=str)
    parser.add_argument("--gpu", action="store_true", default=False)
    args = parser.parse_args()

    pool = POOLS[args.pool]
    ossh(pool, gpu=args.gpu)
