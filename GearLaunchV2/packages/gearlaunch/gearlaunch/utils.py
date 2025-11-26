import subprocess
from concurrent.futures import ThreadPoolExecutor, wait

from termcolor import colored
from tqdm import tqdm


def green_print(message):
    print(colored(message, "green"))


def red_print(message):
    print(colored(message, "red"))


def run_cmd(cmd, verbose=False, async_cmd=False, fault_tolerance=False):
    if verbose:
        assert not async_cmd, "async_cmd is not supported when verbose=True"
        popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = []
        for line in popen.stdout:
            line = line.rstrip().decode("utf-8")
            print(line)
            lines.append(line)
        popen.wait()
        if popen.returncode != 0 and not fault_tolerance:
            raise RuntimeError(
                f"Failed to run command: {cmd}\nERROR {popen.stderr}\nSTDOUT{popen.stdout}"
            )
        popen.stdout = "\n".join(lines)
        return popen
    else:
        if not async_cmd:
            # decode bug fix: https://stackoverflow.com/questions/73545218/utf-8-encoding-exception-with-subprocess-run
            ret = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="cp437")
            if ret.returncode != 0 and not fault_tolerance:
                raise RuntimeError(
                    f"Failed to run command: {cmd}\nERROR {ret.stderr}\nSTDOUT{ret.stdout}"
                )
            return ret
        else:
            popen = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return popen

def run_cmd_with_retry(cmd, verbose=False, max_retries=1):
    try_count = 0
    while try_count < max_retries:
        try:
            ret = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="cp437")
            if ret.returncode != 0:
                raise RuntimeError(
                    f"Failed to run command: {cmd}\nERROR {ret.stderr}\nSTDOUT{ret.stdout}"
                )
            
            return ret
        except Exception as e:
            try_count += 1
            if try_count == max_retries:
                raise e


def map_async_with_thread(
    iterable,
    func,
    num_thread=30,
    desc="",
    verbose=True,
):

    with ThreadPoolExecutor(num_thread) as executor:

        results = []
        not_done = set()

        for i, x in enumerate(iterable):
            future = executor.submit(func, x)
            future.index = i
            not_done.add(future)

        results = {}
        pbar = tqdm(total=len(not_done), desc=desc, disable=not verbose)

        while len(not_done) > 0:
            done, not_done = wait(not_done, return_when="FIRST_COMPLETED")
            for future in done:
                results[future.index] = future.result()
            pbar.update(len(done))

        pbar.close()

        return [results[i] for i in range(len(iterable))]
