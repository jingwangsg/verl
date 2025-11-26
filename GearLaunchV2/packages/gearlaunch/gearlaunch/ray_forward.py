import subprocess
import time
import pandas as pd
from ray.job_submission import JobSubmissionClient
from datetime import datetime
import socket
import requests
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from termcolor import colored
import functools
from loguru import logger

# Constants
SILENT_SUBPROCESS_KWARGS = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
SUCCESS_MARKER = "Starting port forwarding"
RAY_DASHBOARD_MARKER = "You need to enable JavaScript to run this app."


# Custom Exceptions

class WorkflowNotRunningError(Exception):
    """Raised when workflow returns 410 status (workflow not running)."""
    pass


# Helper Functions

def calculate_backoff_delay(attempt, error_type='standard'):
    """Calculate backoff delay based on attempt number and error type.

    Args:
        attempt: Current attempt number (0-indexed)
        error_type: 'standard', '503', or 'linear'

    Returns:
        Delay in seconds
    """
    if error_type == '503':
        return min(12, 3 * (attempt + 1)) + random.uniform(0, 2)
    elif error_type == 'linear':
        return 2 + attempt
    else:  # standard exponential
        return min(8, 2 ** (attempt - 1)) + random.uniform(0, 1) if attempt > 0 else 0


def kill_port_forward_processes(workflow, port=None, src_port=None):
    """Kill port-forward processes for a workflow.

    Args:
        workflow: Workflow name
        port: Local port (optional - for specific port-forward)
        src_port: Source port (optional - for specific port-forward)
    """
    if port and src_port:
        # Kill specific port-forward
        cmd = f"ps aux | grep 'osmo workflow port-forward {workflow} master --port {port}:{src_port}' | grep -v grep | awk '{{print $2}}' | xargs -r kill -9"
    else:
        # Kill all port-forwards for workflow
        cmd = f"ps aux | grep 'osmo workflow port-forward {workflow} master' | grep -v grep | grep -v ':22' | awk '{{print $2}}' | xargs -r kill -9"

    subprocess.run(cmd, shell=True, **SILENT_SUBPROCESS_KWARGS)


def get_empty_job_details(job_id=None):
    """Return empty job details structure."""
    return {
        "job_id": job_id,
        "status_counts": {},
        "total_tasks": 0,
        "task_breakdown": {},
        "failure_rate": 0,
    }


def retry_on_network_error(
    max_retries=3,
    backoff_type='exponential',
    retry_exceptions=(requests.exceptions.RequestException, ConnectionError, TimeoutError),
    log_prefix=None
):
    """
    Decorator to add retry logic to network operations.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_type: 'exponential', 'linear', or '503'
        retry_exceptions: Tuple of exceptions to retry on
        log_prefix: Optional prefix for retry logs (e.g., workflow name)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = calculate_backoff_delay(attempt, backoff_type)
                        if log_prefix:
                            print(f"{log_prefix} - retry {attempt + 1}/{max_retries - 1} after {delay:.1f}s delay")
                        time.sleep(delay)
                    else:
                        # Final attempt failed, raise the exception
                        raise last_exception
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


@retry_on_network_error(max_retries=3, backoff_type='exponential',
                        retry_exceptions=(Exception,))
def _safe_list_jobs(client):
    """Wrapper for client.list_jobs() with retry logic.

    Args:
        client: Ray JobSubmissionClient

    Returns:
        List of jobs from the cluster
    """
    jobs = client.list_jobs()
    jobs_info = [{'job_id': j.job_id, 'submission_id': j.submission_id, 'status': str(j.status)} for j in jobs]
    logger.trace(f"$ client.list_jobs()  # address={client.get_address()}\n  < {jobs_info}")
    return jobs


def get_available_ports(num_ports, start_port=10000, end_port=20000, max_workers=50):
    """Preallocate multiple available ports using parallel detection.

    Algorithm:
    1. Start from start_port, test N candidate ports in parallel
    2. Collect available ports
    3. If collected < N, calculate remaining needed (N1 = N - collected)
    4. Shift forward, parallel test next batch
    5. Repeat until N ports collected

    Args:
        num_ports: Number of ports needed
        start_port: Starting port to search from (default: 10000)
        end_port: Maximum port to search up to (default: 20000)
        max_workers: Maximum parallel workers (default: 50, to avoid "Too many open files")

    Returns:
        List of available ports (sorted)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def check_port(port):
        """Check if a single port is available."""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Set socket options to allow reuse and avoid TIME_WAIT issues
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Set a short timeout to fail fast
            sock.settimeout(0.5)
            # connect_ex returns 0 if connection succeeds (port occupied)
            # returns non-zero if connection fails (port available)
            result = sock.connect_ex(("localhost", port))
            return port, result != 0
        except Exception:
            return port, False
        finally:
            # Explicitly close socket to free file descriptor immediately
            if sock:
                try:
                    sock.close()
                except:
                    pass

    available_ports = []
    current_start = start_port

    while len(available_ports) < num_ports and current_start < end_port:
        # Calculate how many more ports we need
        needed = num_ports - len(available_ports)

        # Prepare candidate ports for this batch
        batch_end = min(current_start + needed, end_port)
        candidate_ports = list(range(current_start, batch_end))

        # Limit max_workers to prevent "Too many open files" error
        actual_workers = min(max_workers, len(candidate_ports))

        # Parallel check this batch
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(check_port, p): p for p in candidate_ports}

            for future in as_completed(futures):
                port, is_available = future.result()
                if is_available:
                    available_ports.append(port)

        # Move to next batch (shift forward)
        current_start = batch_end

    # Sort and return exactly num_ports
    return sorted(available_ports[:num_ports])


def get_workflows(all_users=False):
    all_user_flag = "-a" if all_users else ""
    workflows = subprocess.run(
        f"osmo workflow list -c 9999 {all_user_flag} -s RUNNING | grep '@nvidia.com' | awk '{{print $2}}'",
        shell=True,
        capture_output=True,
        text=True,
    )
    return workflows.stdout.splitlines()


def is_ray_port(port, max_retries=1):
    # Increase retries to 3 for more reliable detection
    actual_retries = max(max_retries, 3)
    url = f"http://localhost:{port}"

    for attempt in range(actual_retries):
        try:
            # Create a new session for each attempt to avoid connection pooling issues
            # with port-forward tunnels
            with requests.Session() as session:
                # Disable keep-alive and connection pooling to work better with port-forward
                session.headers.update({'Connection': 'close'})
                # Disable connection pooling by setting max_retries to 0
                adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
                session.mount('http://', adapter)

                response = session.get(url, timeout=10)
                response_text = response.text

                if RAY_DASHBOARD_MARKER in response_text:
                    logger.trace(f"$ GET {url}\n  < text[:{min(200, len(response_text))}]: {response_text[:200]}\n  = True (Ray dashboard)")
                    return True
                elif response_text.strip():  # Got some response but not ray
                    logger.trace(f"$ GET {url}\n  < text[:{min(200, len(response_text))}]: {response_text[:200]}\n  = False (not Ray)")
                    return False

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # These are expected with port-forward, just retry without printing error
            logger.trace(f"$ GET {url}\n  ! Connection issue (attempt {attempt + 1}/{actual_retries}): {e}")
            # Don't print error for common connection issues - they're expected with port-forward

        except requests.exceptions.RequestException as e:
            logger.trace(f"$ GET {url}\n  ! {e}")
            # Only print error if it's not a RemoteDisconnected error
            if "RemoteDisconnected" not in str(e) and "Connection aborted" not in str(e):
                print(f"Error checking port {port}: {e}")

        except Exception as e:
            logger.trace(f"$ GET {url}\n  ! {e}")
            print(f"Error checking port {port}: {e}")

        if attempt < actual_retries - 1:  # Don't sleep on the last attempt
            delay = calculate_backoff_delay(attempt, 'linear')
            time.sleep(delay)

    logger.trace(f"$ GET {url}\n  = False (no response)")
    return False


def forward_workflow(workflow, port, src_port, max_retries=1, verbose=False):
    """Launch port-forward in background and monitor until 'Starting port forwarding from...' appears"""
    import select

    last_error_was_503 = False
    all_stderr_lines = []  # Collect all stderr for verbose output
    
    for attempt in range(max_retries):
        if attempt > 0:
            # Clean up any existing port-forward processes for this workflow/port before retrying
            print(f"Cleaning up existing port-forward processes for {workflow} port {src_port}")
            kill_port_forward_processes(workflow)

            # Use longer delays for 503/envoy overload errors
            error_type = '503' if last_error_was_503 else 'standard'
            delay = calculate_backoff_delay(attempt, error_type)
            backoff_label = " [503 backoff]" if last_error_was_503 else ""
            print(f"Retrying {workflow} port {src_port} after {delay:.1f}s delay (attempt {attempt + 1}){backoff_label}")
            time.sleep(delay)

        cmd = f"osmo workflow port-forward {workflow} master --port {port}:{src_port}"
        print(f"🚀 Launching: {cmd}")

        # Launch the command in the background
        popen = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )

        # Monitor output for the success message
        timeout = 15  # 15 second timeout
        start_time = time.time()
        success = False
        error_lines = []

        while time.time() - start_time < timeout:
            # Check if process has terminated
            if popen.poll() is not None:
                print(f"⚠️  Process terminated early for {workflow} port {src_port}")
                break

            # Try to read a line from stdout (non-blocking)
            try:
                if popen.stdout and select.select([popen.stdout], [], [], 0.1)[0]:
                    line = popen.stdout.readline()
                    if line:
                        line = line.strip()
                        if SUCCESS_MARKER in line:
                            print(f"✅ {workflow}:{src_port} - {line}")
                            print(f"✅ Port forwarding started for {workflow} port {src_port}")
                            success = True
                            break
                        else:
                            print(f"📝 {workflow}:{src_port} - {line}")
                            if "error" in line.lower() or "fail" in line.lower():
                                error_lines.append(line)
            except:
                pass

            # Try to read errors from stderr
            try:
                if popen.stderr and select.select([popen.stderr], [], [], 0.1)[0]:
                    err_line = popen.stderr.readline()
                    if err_line:
                        err_line = err_line.strip()
                        all_stderr_lines.append(err_line)  # Collect for verbose output
                        
                        # Don't show "Starting port forwarding" as an error
                        if SUCCESS_MARKER in err_line:
                            print(f"✅ {workflow}:{src_port} - {err_line}")
                            print(f"✅ Port forwarding started for {workflow} port {src_port}")
                            success = True
                            break
                        elif "error" in err_line.lower() or "fail" in err_line.lower():
                            print(f"❌ {workflow}:{src_port} ERROR - {err_line}")
                            error_lines.append(err_line)

                            # For serious asyncio errors, check if process has terminated
                            # If so, fail fast instead of waiting for timeout
                            if "task exception" in err_line.lower() or "exception was never retrieved" in err_line.lower():
                                time.sleep(0.5)  # Brief wait for process to terminate
                                if popen.poll() is not None:
                                    print(f"⚠️  Process terminated after serious error for {workflow} port {src_port}")
                                    break
                        else:
                            # In verbose mode, print all stderr lines
                            if verbose:
                                print(f"📝 {workflow}:{src_port} STDERR - {err_line}")
                            else:
                                print(f"📝 {workflow}:{src_port} - {err_line}")
            except:
                pass

            time.sleep(0.1)  # Small delay to prevent busy waiting

        if success:
            return True

        # If we get here, it failed
        exit_code = popen.poll()

        # Capture any remaining stderr
        remaining_stderr = ""
        try:
            if popen.stderr:
                remaining_stderr = popen.stderr.read()
                if remaining_stderr:
                    all_stderr_lines.append(remaining_stderr)
        except:
            pass

        # Check if remaining stderr contains the success marker
        # (Process may have terminated quickly after printing success message)
        if remaining_stderr and SUCCESS_MARKER in remaining_stderr:
            print(f"✅ {workflow}:{src_port} - Found success marker in remaining stderr")
            print(f"✅ Port forwarding started for {workflow} port {src_port}")
            return True

        # Report the failure with details
        failure_reason = f"Process exit code: {exit_code}"
        if error_lines:
            failure_reason += f", Errors: {'; '.join(error_lines)}"
        if remaining_stderr.strip():
            failure_reason += f", Stderr: {remaining_stderr.strip()}"

        # Check if this was a 410 error (workflow not running)
        full_error_text = failure_reason.lower()
        is_410_error = (
            "410" in full_error_text or
            "not running" in full_error_text or
            "error code: 410" in full_error_text
        )

        if is_410_error:
            # Clean up the failed process before raising
            if popen.poll() is None:
                popen.terminate()
                time.sleep(0.1)
                if popen.poll() is None:
                    popen.kill()
            kill_port_forward_processes(workflow, port, src_port)

            print(f"⚠️  Workflow {workflow} is not running (410) - skipping")
            raise WorkflowNotRunningError(f"Workflow {workflow} is not running: {failure_reason}")

        # Check if this was a 503/envoy overload error
        last_error_was_503 = ("503" in full_error_text or "envoy" in full_error_text or "overload" in full_error_text)

        # Print simplified message for 503 errors, full details for others
        if last_error_was_503:
            print(f"❌ HTTP 503 (envoy overload) - retrying {workflow}:{src_port} (attempt {attempt + 1}/{max_retries})")
        else:
            print(f"❌ Port forward failed for {workflow} port {src_port} (attempt {attempt + 1}/{max_retries}): {failure_reason}")
        
        # In verbose mode, print complete stderr output
        if verbose and all_stderr_lines:
            print(f"\n{'='*60}")
            print(f"VERBOSE: Complete stderr for {workflow}:{src_port}")
            print(f"{'='*60}")
            for line in all_stderr_lines:
                print(f"  {line}")
            print(f"{'='*60}\n")

        # Clean up the failed process
        if popen.poll() is None:
            popen.terminate()
            time.sleep(0.1)
            if popen.poll() is None:
                popen.kill()

        # Clean up any zombie processes for this specific port-forward
        kill_port_forward_processes(workflow, port, src_port)

    return False


def forward_port_with_410_handling(workflow, port, src_port, max_retry, verbose=False):
    """Wrapper for forward_workflow that handles 410 errors.

    Args:
        workflow: Workflow name
        port: Local port
        src_port: Source port
        max_retry: Maximum number of retries
        verbose: Print detailed error tracebacks

    Returns:
        Tuple of (success: bool, is_410: bool, error_msg: str or None)
        - (True, False, None): Success
        - (False, True, error_msg): 410 error (workflow not running)
        - (False, False, error_msg): Other error
    """
    try:
        success = forward_workflow(workflow, port, src_port, max_retry, verbose=verbose)
        if success:
            return True, False, None
        else:
            return False, False, f"Port forward failed for {workflow}:{src_port}"
    except WorkflowNotRunningError as e:
        return False, True, str(e)
    except Exception as e:
        return False, False, f"Unexpected error for {workflow}:{src_port}: {str(e)}"


# ============================================================================
# REFACTORED UNIFIED FUNCTIONS
# ============================================================================

def create_workflow_port_mappings(workflows, allocated_ports, ports_to_forward):
    """Create clean workflow->ports mappings.

    Args:
        workflows: List of workflow names
        allocated_ports: List of allocated port numbers
        ports_to_forward: List of source ports to forward (e.g., [8265, 8300] or [8265])

    Returns:
        List of dicts with 'workflow' and 'ports' keys
    """
    mappings = []
    ports_per_workflow = len(ports_to_forward)

    for i, workflow in enumerate(workflows):
        port_mapping = {}
        for j, src_port in enumerate(ports_to_forward):
            port_mapping[src_port] = allocated_ports[i * ports_per_workflow + j]

        mappings.append({
            'workflow': workflow,
            'ports': port_mapping  # e.g., {8265: 10001, 8300: 10002}
        })

    return mappings


def forward_ports_for_workflows(workflow_port_mappings, max_retry, max_workers=8, verbose=False):
    """Forward ports for a list of workflows.

    Args:
        workflow_port_mappings: List of dicts with 'workflow' and 'ports' keys
        max_retry: Maximum retry attempts
        max_workers: Maximum parallel workers
        verbose: Print detailed error tracebacks

    Returns:
        Dict with 'successful' (list of successful mappings), 'failed', and 'skipped_410'
    """
    print(f"  🚀 Forwarding ports for {len(workflow_port_mappings)} workflows...")

    # Track results
    workflow_results = {}  # workflow -> {'ports': {...}, 'success': {...}, 'is_410': bool, 'errors': [...]}

    # Create all forwarding tasks
    def forward_single_port(workflow, local_port, src_port, max_retry):
        """Forward a single port and return result."""
        success, is_410, error_msg = forward_port_with_410_handling(workflow, local_port, src_port, max_retry, verbose=verbose)
        return workflow, src_port, success, is_410, error_msg

    all_tasks = []
    for mapping in workflow_port_mappings:
        workflow = mapping['workflow']
        ports = mapping['ports']

        # Initialize result tracking
        workflow_results[workflow] = {
            'ports': ports,
            'success': {src_port: False for src_port in ports.keys()},
            'is_410': False,
            'errors': []
        }

        # Create forwarding tasks for each port
        for src_port, local_port in ports.items():
            all_tasks.append((workflow, local_port, src_port, max_retry))

    # Execute all tasks in parallel with staggered delays
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (workflow, local_port, src_port, max_retry) in enumerate(all_tasks):
            if i > 0:
                time.sleep(random.uniform(0.1, 0.3))
            future = executor.submit(forward_single_port, workflow, local_port, src_port, max_retry)
            futures.append(future)

        # Process results as they complete
        for future in as_completed(futures):
            try:
                workflow, src_port, success, is_410, error_msg = future.result()

                if is_410:
                    workflow_results[workflow]['is_410'] = True
                    if error_msg:
                        workflow_results[workflow]['errors'].append(error_msg)
                elif success:
                    workflow_results[workflow]['success'][src_port] = True
                else:
                    if error_msg:
                        workflow_results[workflow]['errors'].append(error_msg)

            except Exception as e:
                if verbose:
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"  ❌ Unexpected error in forwarding: {e}")

    # Categorize results
    successful = []
    skipped_410 = []
    failed = []

    for workflow, result in workflow_results.items():
        if result['is_410']:
            skipped_410.append(workflow)
            kill_port_forward_processes(workflow)
        elif all(result['success'].values()):
            # All ports forwarded successfully
            successful.append({'workflow': workflow, 'ports': result['ports']})
        else:
            # Some ports failed
            error_summary = "; ".join(result['errors']) if result['errors'] else "Unknown error"
            failed.append((workflow, error_summary))
            kill_port_forward_processes(workflow)

    print(f"  ✅ Forwarded: {len(successful)}/{len(workflow_port_mappings)}, "
          f"Skipped (410): {len(skipped_410)}, Failed: {len(failed)}")

    return {
        'successful': successful,
        'skipped_410': skipped_410,
        'failed': failed
    }


def detect_ray_on_ports(workflow_port_mappings, max_workers=10, verbose=False):
    """Detect Ray dashboard on forwarded ports.

    Args:
        workflow_port_mappings: List of dicts with 'workflow' and 'ports' keys
        max_workers: Maximum parallel workers
        verbose: Print detailed error tracebacks

    Returns:
        List of dicts with Ray-enabled workflows and their Ray port
    """
    print(f"  🔍 Detecting Ray on {len(workflow_port_mappings)} workflows...")

    # Give ports time to stabilize
    time.sleep(5)

    # Create detection tasks
    port_detection_tasks = []
    workflow_ports_map = {}  # workflow -> port_mapping

    for mapping in workflow_port_mappings:
        workflow = mapping['workflow']
        ports = mapping['ports']
        workflow_ports_map[workflow] = ports

        # Create detection task for each port
        for src_port, local_port in ports.items():
            port_detection_tasks.append((workflow, src_port, local_port))

    # Track detection results
    detection_results = {}  # workflow -> {src_port: bool}
    for workflow, ports in workflow_ports_map.items():
        detection_results[workflow] = {src_port: False for src_port in ports.keys()}

    def detect_single_port(workflow, src_port, local_port):
        """Detect if a single port has Ray dashboard."""
        is_ray = is_ray_port(local_port, max_retries=3)
        return workflow, src_port, local_port, is_ray

    # Execute port detection in batches
    batch_size = 20
    for batch_start in range(0, len(port_detection_tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(port_detection_tasks))
        batch_tasks = port_detection_tasks[batch_start:batch_end]

        if batch_start > 0:
            time.sleep(0.5)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_tasks))) as executor:
            futures = [
                executor.submit(detect_single_port, workflow, src_port, local_port)
                for workflow, src_port, local_port in batch_tasks
            ]

            for future in as_completed(futures):
                try:
                    workflow, src_port, local_port, is_ray = future.result()
                    detection_results[workflow][src_port] = is_ray
                    if is_ray:
                        print(f"  ✅ {workflow}: Ray detected on port {src_port} (localhost:{local_port})")
                except Exception as e:
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    else:
                        print(f"  ❌ Error detecting port: {e}")

    # Process results and determine Ray workflows
    ray_workflows = []
    no_ray = []

    for workflow, results in detection_results.items():
        ports = workflow_ports_map[workflow]
        ray_port = None

        # Find first port with Ray (prefer lower port numbers)
        for src_port in sorted(results.keys()):
            if results[src_port]:
                ray_port = ports[src_port]
                # Clean up other ports
                for other_src_port, other_local_port in ports.items():
                    if other_src_port != src_port:
                        kill_port_forward_processes(workflow, other_local_port, other_src_port)
                break

        if ray_port:
            ray_workflows.append({'workflow': workflow, 'ray_port': ray_port})
        else:
            print(f"  ❌ {workflow}: No Ray dashboard detected, excluding")
            kill_port_forward_processes(workflow)
            no_ray.append(workflow)

    print(f"  ✅ Ray detected: {len(ray_workflows)}, No Ray: {len(no_ray)}")

    return ray_workflows


def forward_and_detect_ray_ports(workflows, ports_to_check, max_retry, max_workers=8, verbose=False, workflow_type="auto"):
    """Unified function to forward ports and detect Ray for a group of workflows.

    Args:
        workflows: List of workflow names
        ports_to_check: List of port numbers to check (e.g., [8265, 8300] or [8265])
        max_retry: Max retry attempts for port forwarding
        max_workers: Max parallel workers
        verbose: Verbose output
        workflow_type: "auto" or "cmdline" for logging

    Returns:
        List of dicts with Ray-enabled workflows and their ray_port
    """
    if not workflows:
        return []

    port_desc = "+".join(map(str, ports_to_check))
    print(f"\n{'='*70}")
    print(f"Processing {len(workflows)} {workflow_type} workflows (ports: {port_desc})")
    print(f"{'='*70}")

    # Allocate ports
    num_ports_needed = len(workflows) * len(ports_to_check)
    allocated_ports = get_available_ports(num_ports_needed)
    print(f"  🔌 Allocated {num_ports_needed} ports: {allocated_ports[0]}-{allocated_ports[-1]}")

    # Create port mappings
    workflow_port_mappings = create_workflow_port_mappings(workflows, allocated_ports, ports_to_check)

    # Phase 1: Forward ports
    forward_result = forward_ports_for_workflows(workflow_port_mappings, max_retry, max_workers, verbose)

    if not forward_result['successful']:
        print(f"  ⚠️  No workflows successfully forwarded")
        return []

    # Phase 2: Detect Ray
    ray_workflows = detect_ray_on_ports(forward_result['successful'], max_workers, verbose)

    print(f"{'='*70}\n")

    return ray_workflows


def new_phase3_forward_monitoring_ports(ray_workflows, max_retry=1, max_workers=8, start_port=None, verbose=False):
    """Phase 3: Forward monitoring ports (3000 and 9090) for Ray workflows.

    Args:
        ray_workflows: List of tuples (workflow, ray_port)
        max_retry: Maximum number of retries for port forwarding
        max_workers: Maximum number of parallel workers
        start_port: Starting port for allocation (default: 10000)
        verbose: Print detailed error tracebacks

    Returns:
        Dictionary with keys:
        - 'successful': List of tuples (workflow, ray_port, port_3000, port_9090)
        - 'partial': List of tuples (workflow, ray_port, port_3000_or_none, port_9090_or_none)
            where at least one monitoring port failed but workflow is still kept
    """
    print(f"📍 PHASE 3: Monitoring Port Forwarding - Forwarding 3000 and 9090 for {len(ray_workflows)} workflows")

    if not ray_workflows:
        return {'successful': [], 'partial': []}

    # Use provided start_port or default to 10000
    if start_port is None:
        start_port = 10000

    # Preallocate ports for monitoring (2 per workflow)
    available_ports = get_available_ports(len(ray_workflows) * 2, start_port=start_port)

    # Track results
    workflow_results = {}  # workflow -> {'ray_port': int, 'ports': {...}, 'success': {...}}

    # Create all forwarding tasks
    def forward_single_port(workflow, ray_port, port, src_port, max_retry):
        """Forward a single monitoring port and return result."""
        try:
            success = forward_workflow(workflow, port, src_port, max_retry, verbose=verbose)
            return workflow, src_port, success, None
        except WorkflowNotRunningError as e:
            # 410 error during Phase 3 - this shouldn't happen but handle it
            return workflow, src_port, False, str(e)
        except Exception as e:
            return workflow, src_port, False, str(e)

    all_tasks = []
    for i, (workflow, ray_port) in enumerate(ray_workflows):
        port_3000 = available_ports[i * 2]
        port_9090 = available_ports[i * 2 + 1]
        
        # Create tasks for both monitoring ports
        all_tasks.append((workflow, ray_port, port_3000, 3000, max_retry))
        all_tasks.append((workflow, ray_port, port_9090, 9090, max_retry))

        # Initialize workflow result tracking
        workflow_results[workflow] = {
            'ray_port': ray_port,
            'ports': {3000: port_3000, 9090: port_9090},
            'success': {3000: False, 9090: False},
            'errors': []
        }

    # Execute all tasks in parallel with staggered delays
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (workflow, ray_port, port, src_port, max_retry) in enumerate(all_tasks):
            # Add small random delay between submissions to prevent envoy overload
            if i > 0:
                time.sleep(random.uniform(0.1, 0.3))
            future = executor.submit(forward_single_port, workflow, ray_port, port, src_port, max_retry)
            futures.append(future)

        # Process results as they complete
        for future in as_completed(futures):
            try:
                workflow, src_port, success, error_msg = future.result()

                if success:
                    workflow_results[workflow]['success'][src_port] = True
                else:
                    if error_msg:
                        workflow_results[workflow]['errors'].append(f"Port {src_port}: {error_msg}")
                    # Clean up failed port-forward
                    ports = workflow_results[workflow]['ports']
                    kill_port_forward_processes(workflow, ports[src_port], src_port)

            except Exception as e:
                if verbose:
                    import traceback
                    print(f"❌ Unexpected error in phase3:")
                    traceback.print_exc()
                else:
                    print(f"❌ Unexpected error in phase3: {e}")

    # Categorize results
    successful = []
    partial = []

    for workflow, result in workflow_results.items():
        ray_port = result['ray_port']
        ports = result['ports']
        success_3000 = result['success'][3000]
        success_9090 = result['success'][9090]

        if success_3000 and success_9090:
            # Both monitoring ports succeeded
            successful.append((workflow, ray_port, ports[3000], ports[9090]))
        else:
            # At least one monitoring port failed, but keep the workflow
            port_3000_result = ports[3000] if success_3000 else None
            port_9090_result = ports[9090] if success_9090 else None
            partial.append((workflow, ray_port, port_3000_result, port_9090_result))

    # Print summary
    total_workflows = len(successful) + len(partial)
    print(f"✅ Phase 3 Complete: {len(successful)} fully forwarded, {len(partial)} partially forwarded")

    if partial:
        print(f"⚠️  Partially forwarded workflows (missing some monitoring ports): {', '.join([w for w, _, _, _ in partial])}")

    # Show detailed monitoring port forwarding results
    if successful or partial:
        print(f"\n{'='*60}")
        print("PHASE 3 RESULTS - Monitoring Port Forwarding:")
        print(f"{'='*60}")
        for workflow, ray_port, port_3000, port_9090 in successful:
            print(f"  {workflow}:3000 -> http://localhost:{port_3000} (Grafana)")
            print(f"  {workflow}:9090 -> http://localhost:{port_9090} (Prometheus)")
        for workflow, ray_port, port_3000, port_9090 in partial:
            if port_3000:
                print(f"  {workflow}:3000 -> http://localhost:{port_3000} (Grafana)")
            else:
                print(f"  {workflow}:3000 -> FAILED")
            if port_9090:
                print(f"  {workflow}:9090 -> http://localhost:{port_9090} (Prometheus)")
            else:
                print(f"  {workflow}:9090 -> FAILED")
        print(f"{'='*60}\n")

    # Combine for total results
    all_results = successful + partial

    return {
        'successful': successful,
        'partial': partial,
        'all': all_results  # All workflows regardless of monitoring port status
    }


@retry_on_network_error(max_retries=3, backoff_type='linear', 
                        retry_exceptions=(requests.exceptions.RequestException, KeyError, Exception))
def get_cluster_status(port):
    """Get cluster GPU/CPU usage status with automatic retry."""
    api_url = f"http://localhost:{port}/api/cluster_status"
    
    try:
        response = requests.get(api_url, timeout=10).json()
        gpu_usage = response["data"]["clusterStatus"]["loadMetricsReport"]["usage"].get("GPU", [0, 0])
        cpu_usage = response["data"]["clusterStatus"]["loadMetricsReport"]["usage"].get("CPU", [0, 0])

        result = {
            "GPU": gpu_usage,
            "CPU": cpu_usage,
        }
        
        logger.trace(f"$ curl -X GET {api_url}\n  < {response}\n  = {result}")
        return result
    except Exception as e:
        logger.trace(f"$ curl -X GET {api_url}\n  ! {e}")
        # If all retries fail, return default values
        return {"GPU": [0, 0], "CPU": [0, 0]}


def get_job_task_details(job_id: str | None, port: int) -> dict:
    """
    Get detailed task information for a specific Ray job.
    Uses same timeout and retry logic as Ray dashboard.

    Returns:
        Dictionary with detailed task information including:
        - status_counts: Task counts by status
        - total_tasks: Total number of tasks
        - task_breakdown: Per-function task counts
        - failure_rate: Percentage of failed tasks
    """
    if job_id is None:
        return get_empty_job_details(job_id)

    # Use same timeout as Ray dashboard backend (30 seconds)
    timeout = 30
    retries = 3
    
    for attempt in range(retries):
        try:
            dashboard_url = f"http://localhost:{port}"
            progress_url = f"{dashboard_url}/api/v0/tasks/summarize?filter_keys=job_id&filter_predicates=%3D&filter_values={job_id}"
            
            # Use same 30-second timeout as Ray dashboard backend
            response = requests.get(progress_url, timeout=timeout)
            response.raise_for_status()

            data = response.json()

            # Same error handling as dashboard
            if not data.get("result", False):
                logger.trace(f"$ curl -X GET '{progress_url}'\n  < {data}\n  ! result=false")
                print(f"API returned result=false for job {job_id}: {data.get('msg', 'Unknown error')}")
                if attempt < retries - 1:
                    time.sleep(2)  # Brief delay between retries
                    continue
                # Return empty structure instead of None (like dashboard)
                return get_empty_job_details(job_id)

            task_summary = data["data"]["result"]["result"]["node_id_to_summary"]["cluster"]["summary"]
            total_tasks = data["data"]["result"]["num_filtered"]

            # Aggregate status counts
            status_counts = {}
            task_breakdown = {}

            for task_name, task_info in task_summary.items():
                func_name = task_info["func_or_class_name"]
                state_counts = task_info.get("state_counts", {})

                # Store per-function breakdown
                task_breakdown[func_name] = state_counts

                # Aggregate total counts
                for state, count in state_counts.items():
                    status_counts[state] = status_counts.get(state, 0) + count

            # Calculate failure rate
            failed_count = status_counts.get("FAILED", 0)
            failure_rate = (failed_count / total_tasks * 100) if total_tasks > 0 else 0

            result = {
                "job_id": job_id,
                "status_counts": status_counts,
                "total_tasks": total_tasks,
                "task_breakdown": task_breakdown,
                "failure_rate": failure_rate,
            }
            logger.trace(f"$ curl -X GET '{progress_url}'\n  < {data}\n  = {result}")
            return result

        except requests.exceptions.Timeout as e:
            logger.trace(f"$ curl -X GET '{progress_url}'\n  ! Timeout (attempt {attempt + 1}/{retries}): {e}")
            print(f"Timeout on attempt {attempt + 1}/{retries} for job {job_id}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            
        except requests.exceptions.RequestException as e:
            logger.trace(f"$ curl -X GET '{progress_url}'\n  ! RequestException (attempt {attempt + 1}/{retries}): {e}")
            print(f"Request error on attempt {attempt + 1}/{retries} for job {job_id}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
                continue
            
        except Exception as e:
            logger.trace(f"$ curl -X GET '{progress_url}'\n  ! Exception (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error on attempt {attempt + 1}/{retries} for job {job_id}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
                continue
    
    # If all retries failed, return empty structure (like dashboard does)
    logger.trace(f"$ curl -X GET '{progress_url}'\n  ! All retries exhausted")
    return get_empty_job_details(job_id)


def monitor_ray_cluster(args):
    max_retry = args.max_retry
    verbose = args.verbose
    print("🧹 Cleaning up existing port-forward processes...")
    # Clean up port 8265 and 8300 processes
    subprocess.run(
        "ps aux | grep osmo | grep port-forward | grep \":8265\" | awk '{print $2}' | xargs -r kill -9",
        shell=True,
        **SILENT_SUBPROCESS_KWARGS,
    )
    subprocess.run(
        "ps aux | grep osmo | grep port-forward | grep \":8300\" | awk '{print $2}' | xargs -r kill -9",
        shell=True,
        **SILENT_SUBPROCESS_KWARGS,
    )

    # Check file descriptor limits to prevent "Too many open files" errors
    try:
        import resource
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"📊 File descriptor limits: soft={soft_limit}, hard={hard_limit}")

        # Warn if soft limit is below recommended threshold
        recommended_limit = 1024
        if soft_limit < recommended_limit:
            print(f"⚠️  WARNING: File descriptor limit ({soft_limit}) is below recommended ({recommended_limit})")
            print(f"    You may encounter 'Too many open files' errors with many workflows")
            print(f"    To increase: ulimit -n {hard_limit if hard_limit < 10000 else 4096}")
    except Exception as e:
        if verbose:
            print(f"Could not check file descriptor limits: {e}")

    # ========================================================================
    # Stage 1: Get and classify workflows
    # ========================================================================
    print("📋 Getting workflows...")
    workflows = get_workflows(all_users=args.all_users)
    print(f"📋 Found {len(workflows)} total workflows")

    # Separate auto-discovered and command-line workflows
    auto_discovered_workflows = [w for w in workflows if "ray" in w.lower() or "cluster" in w.lower()]
    cmdline_workflows = args.workflow

    # Deduplicate: Remove cmdline workflows that are already in auto_discovered
    cmdline_workflows_original_count = len(cmdline_workflows)
    cmdline_workflows = [w for w in cmdline_workflows if w not in auto_discovered_workflows]
    duplicates_removed = cmdline_workflows_original_count - len(cmdline_workflows)

    if duplicates_removed > 0:
        print(f"🔄 Removed {duplicates_removed} duplicate workflow(s) from command-line list (already auto-discovered)")

    print(f"🎯 {len(auto_discovered_workflows)} auto-discovered workflows (8265+8300)")
    print(f"📌 {len(cmdline_workflows)} command-line workflows (8265 only)")

    if not auto_discovered_workflows and not cmdline_workflows:
        print("❌ No ray workflows found! Exiting...")
        return

    # ========================================================================
    # Stage 2: Process auto-discovered workflows (dual-port: 8265 + 8300)
    # ========================================================================
    auto_ray_workflows = forward_and_detect_ray_ports(
        workflows=auto_discovered_workflows,
        ports_to_check=[8265, 8300],
        max_retry=max_retry,
        max_workers=8,
        verbose=verbose,
        workflow_type="auto-discovered"
    )

    # ========================================================================
    # Stage 3: Process command-line workflows (single-port: 8265 only)
    # ========================================================================
    cmdline_ray_workflows = forward_and_detect_ray_ports(
        workflows=cmdline_workflows,
        ports_to_check=[8265],
        max_retry=max_retry,
        max_workers=8,
        verbose=verbose,
        workflow_type="command-line"
    )

    # ========================================================================
    # Stage 4: Combine results
    # ========================================================================
    all_ray_workflows = auto_ray_workflows + cmdline_ray_workflows

    if not all_ray_workflows:
        print("❌ No Ray clusters detected. Exiting...")
        return

    print(f"\n{'='*70}")
    print(f"📊 Summary: {len(all_ray_workflows)} Ray clusters detected")
    print(f"  - Auto-discovered: {len(auto_ray_workflows)}")
    print(f"  - Command-line: {len(cmdline_ray_workflows)}")
    print(f"{'='*70}\n")

    # ========================================================================
    # Stage 5: Forward monitoring ports (3000 and 9090) - ONLY for auto_ray
    # ========================================================================
    # Convert auto_ray workflows to format expected by new_phase3_forward_monitoring_ports
    # Only forward monitoring ports for auto-discovered workflows, not command-line workflows
    auto_ray_workflows_tuples = [(wf['workflow'], wf['ray_port']) for wf in auto_ray_workflows]

    # Forward monitoring ports only for auto-discovered workflows
    phase3_result = new_phase3_forward_monitoring_ports(
        auto_ray_workflows_tuples,
        max_retry=max_retry,
        max_workers=8,
        start_port=10000,
        verbose=verbose
    )

    # ========================================================================
    # Stage 6: Start cluster monitoring
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"📊 Starting cluster monitoring for {len(all_ray_workflows)} Ray clusters")
    print(f"{'='*70}\n")

    # Build port mappings for monitoring
    successful_workflows = []
    port_3000_mapping = {}
    port_9090_mapping = {}

    # Add auto-discovered workflows with monitoring ports
    for workflow, ray_port, port_3000, port_9090 in phase3_result['all']:
        successful_workflows.append((workflow, ray_port))
        if port_3000:
            port_3000_mapping[workflow] = port_3000
        if port_9090:
            port_9090_mapping[workflow] = port_9090

    # Add command-line workflows without monitoring ports (8265 only)
    for wf in cmdline_ray_workflows:
        successful_workflows.append((wf['workflow'], wf['ray_port']))

    # Parallelize JobSubmissionClient creation - Use N workers for N tasks
    clients = {}
    def create_ray_client(workflow_port):
        workflow, port = workflow_port
        max_retries = 5  # Increased from 3 to 5 for better reliability

        # Pre-check: Verify port is actually responding before attempting connection
        if not is_ray_port(port, max_retries=1):
            logger.trace(f"! {workflow} port {port} is not a Ray port")
            print(f"Pre-check failed for {workflow}: Port {port} not responding to Ray dashboard check")
            return None, None

        for attempt in range(max_retries):
            try:
                # Pre-verification: Test HTTP connectivity before creating client
                test_url = f"http://localhost:{port}/api/cluster_status"
                response = requests.get(test_url, timeout=5)

                if response.status_code != 200:
                    logger.trace(f"$ curl -X GET {test_url}\n  < HTTP {response.status_code}\n  ! Non-200 status")
                    raise ValueError(f"HTTP {response.status_code}")

                # Now create the client
                client_address = f"http://localhost:{port}"
                client = JobSubmissionClient(client_address)
                logger.trace(f"$ curl -X GET {test_url}\n  < HTTP {response.status_code}\n$ JobSubmissionClient('{client_address}')\n  = {client_address}")
                return (workflow, port), client

            except Exception as e:
                logger.trace(f"$ JobSubmissionClient('{workflow}' port {port})\n  ! attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    # Use longer exponential backoff: 2s, 4s, 8s, 16s, 32s
                    delay = 2 * (2 ** attempt)
                    time.sleep(delay)
                    continue
                # Final attempt failed
                print(f"Error creating JobSubmissionClient for {workflow}: {e}")
                return None, None

        return None, None

    print(f"🔗 Creating Ray clients for {len(successful_workflows)} workflows...")

    # Limit max_workers to prevent "Too many open files" error
    with ThreadPoolExecutor(max_workers=min(20, len(successful_workflows))) as executor:
        # Submit all client creation tasks
        client_futures = [executor.submit(create_ray_client, wp) for wp in successful_workflows]

        # Process client creation results
        for future in as_completed(client_futures):
            try:
                result = future.result()
                # Check if result is valid before unpacking
                if result is None or result == (None, None):
                    continue
                (workflow, port), client = result
                if workflow and port and client:
                    clients[(workflow, port)] = client
                    print(f"✅ Successfully connected to {workflow} on port {port}")
            except Exception as e:
                if verbose:
                    import traceback
                    print(f"❌ Error creating client:")
                    traceback.print_exc()
                else:
                    print(f"Error creating client: {e}")

    print(f"✅ Setup complete! Found {len(clients)} working ray clusters")

    # Show detailed Ray client results
    if clients:
        print(f"\n{'='*60}")
        print("PHASE 4 RESULTS - Ray Client Creation:")
        print(f"{'='*60}")
        for (workflow, port), client in clients.items():
            print(f"  {workflow} -> http://localhost:{port} (Ray Dashboard)")
        print(f"{'='*60}\n")

    while True:
        # Clear screen to refresh display
        output_str = f"Updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        failed_workflows = []

        st = time.time()

        def query_cluster_status(workflow, port, client):
            output_str = ""
            cluster_status = get_cluster_status(port)

            # print(f"Listing jobs for {workflow}")
            jobs = _safe_list_jobs(client)
            # print(f"Found {len(jobs)} jobs")

            running_jobs = [j for j in jobs if str(j.status) == "RUNNING"]
            output_str += colored(f"{workflow} ({len(running_jobs)} running jobs) http://localhost:{port}\n", "green")
            if port_3000_mapping.get(workflow, None):
                output_str += colored(f"Grafana: http://localhost:{port_3000_mapping[workflow]}/d/rayDefaultDashboard/default-dashboard?orgId=1&refresh=10s\n", "blue")
            if port_9090_mapping.get(workflow, None):
                output_str += colored(f"Prometheus: http://localhost:{port_9090_mapping[workflow]}/graph\n", "blue")

            gpu_usage = cluster_status["GPU"]
            cpu_usage = cluster_status["CPU"]
            output_str += colored(f"GPU: {gpu_usage[0]}/{gpu_usage[1]} CPU: {cpu_usage[0]}/{cpu_usage[1]}\n", "yellow")

            running_job_infos = []

            for job in running_jobs:
                # print(f"Querying job {job.job_id} for {workflow}")
                job_details = get_job_task_details(job.job_id, port)

                emoji_dict = {
                    "PENDING_NODE_ASSIGNMENT": "🟡",
                    "RUNNING": "🔵",
                    "FINISHED": "🟢",
                    "FAILED": "🔴",
                }
                status_list = ["PENDING_NODE_ASSIGNMENT", "RUNNING", "FINISHED", "FAILED"]

                status_counts_str = "".join(
                    f"{emoji_dict[k]}{job_details['status_counts'].get(k, 0):<4}"
                    for k in status_list
                )
                running_job_info = {
                    "job": job.submission_id,
                    "workflow": workflow,
                    "dashboard": (
                        "N/A"
                        if job.driver_info is None
                        else f"http://localhost:{port}/#/jobs/{job.driver_info.id}"
                    ),
                    "task_counts": status_counts_str,
                }
                running_job_infos.append(running_job_info)

            output_str += pd.DataFrame(running_job_infos).to_markdown(index=False) + "\n"
            output_str += "\n"

            return output_str

        # Limit max_workers to prevent "Too many open files" error
        with ThreadPoolExecutor(max_workers=min(20, len(clients))) as executor:
            future_to_workflow = {
                executor.submit(query_cluster_status, workflow, port, client): (
                    workflow,
                    port,
                    client,
                )
                for (workflow, port), client in clients.items()
            }
            for future in future_to_workflow:
                try:
                    workflow, port, client = future_to_workflow[future]
                    output_str += future.result()
                except Exception as e:
                    if verbose:
                        import traceback
                        print(f"❌ Error getting job info for {workflow}:")
                        traceback.print_exc()
                    else:
                        print(f"Error getting job info for {workflow}: {e}")
                    failed_workflows.append((workflow, port))

        for workflow_port in failed_workflows:
            removed = clients.pop(workflow_port, None)
            if removed:
                print(f"⚠️  Removed failed cluster {workflow_port[0]} from monitoring")
        time.sleep(15)

        # Clear screen and scrollback buffer (more thorough)
        print("\033[2J\033[3J\033[H", end="", flush=True)
        print(output_str)

        print(f"Query time taken: {time.time() - st:.2f}s")


def ray_forward():
    print("🔍 Starting ray_forward...")
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", type=str, nargs="*")
    parser.add_argument("-a", "--all_users", action="store_true", default=False)
    parser.add_argument("--max_retry", type=int, default=5, help="Maximum number of retries for port forwarding and ray detection")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Print detailed error information including full tracebacks")
    args = parser.parse_args()

    # Configure loguru logger based on verbose mode
    logger.remove()  # Remove default handler
    if args.verbose:
        logger.add(
            lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="TRACE"
        )
        logger.info("Verbose mode enabled - TRACE level logging activated")
    else:
        logger.add(
            lambda msg: print(msg, end=""),
            format="<level>{level: <8}</level> | <level>{message}</level>",
            level="INFO"
        )

    while True:
        try:
            monitor_ray_cluster(args)
        except Exception as e:
            if args.verbose:
                import traceback
                print(f"❌ Error in monitor_ray_cluster:")
                traceback.print_exc()
            else:
                print(f"Error in monitor_ray_cluster: {e}")


if __name__ == "__main__":
    ray_forward()
