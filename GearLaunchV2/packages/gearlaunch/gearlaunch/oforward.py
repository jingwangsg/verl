import argparse
import multiprocessing as mp
from .utils import run_cmd


def port_forward(workflow, src_port, dst_port, task="master"):
    # session_name = f"port_{workflow}_{task}"
    # run_cmd(f"tmux kill-session -t {session_name}", fault_tolerance=True)
    # run_cmd(f"tmux new-session -d -s {session_name}")

    print(f"{workflow}: {task} {dst_port}(local) -> {src_port}(workflow)")
    run_cmd(f"osmo workflow port-forward {workflow} {task} --port {dst_port}:{src_port}")


def oforward(workflow, src_ports, dst_ports, ports_per_task=1):
    for i, (src_port, dst_port) in enumerate(zip(src_ports, dst_ports)):
        node_rank = i // ports_per_task
        task = f"worker_{node_rank}" if node_rank > 0 else "master"
        mp.Process(target=port_forward, args=(workflow, src_port, dst_port, task)).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", type=str)
    parser.add_argument("ports", type=str, nargs="+")
    parser.add_argument("--ports_per_task", type=int, default=1)
    args = parser.parse_args()

    src_ports = []
    dst_ports = []
    for port in args.ports:
        if ":" not in port:
            src_port = port
            dst_port = port
        else:
            src_port, dst_port = port.split(":")
        src_ports += [src_port]
        dst_ports += [dst_port]
        run_cmd(
            f"ps aux | grep osmo | grep port | grep {src_port} | awk '{{print $2}}' | xargs kill -9",
            fault_tolerance=True,
        )
        assert len(src_ports) == len(
            dst_ports
        ), "Number of source ports and destination ports must be the same"

    oforward(args.workflow, src_ports, dst_ports, args.ports_per_task)
