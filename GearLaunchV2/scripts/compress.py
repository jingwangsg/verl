import os
import sys
import shutil
import subprocess
import tempfile
import argparse
import random
import time
import json
import datetime
from loguru import logger

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


import ray

def setup_logging(log_dir: str = "logs"):
    """Setup logging with loguru to both file and console."""
    # Remove default logger
    logger.remove()
    
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Add console handler with INFO level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    # Add file handler with DEBUG level
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_filename = f"compress_{timestamp}_pid{pid}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | PID:{process} | {level: <8} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="7 days"
    )
    
    logger.info(f"Logging initialized. Process PID: {pid}")
    logger.info(f"Detailed logs will be written to: {log_path}")


def get_all_files(directory: str = ".", exclude_patterns: list[str] = None) -> list[str]:
    """Get all files and symlinks in the directory recursively.
    Returns both regular files and symlinks.

    Args:
        directory: Base directory to search
        exclude_patterns: List of patterns to exclude (supports both glob patterns and directory names)
    """
    if exclude_patterns is None:
        exclude_patterns = []

    # Try to find fd binary (could be 'fd' or 'fdfind')
    fd_bin = None
    for candidate in ("fd", "fdfind"):
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True, check=False)
            if result.returncode == 0:
                fd_bin = candidate
                break
        except FileNotFoundError:
            continue

    if not fd_bin:
        logger.error("fd command not found. Please install fd-find package.")
        logger.error("On Ubuntu: sudo apt install fd-find")
        logger.error("On macOS: brew install fd")
        raise RuntimeError("fd command is required but not installed")

    try:
        # Build fd command with exclude patterns
        cmd = [fd_bin, "--type", "f", "--type", "l", "--follow", "--base-directory", directory]

        # Add exclude patterns
        for pattern in exclude_patterns:
            cmd.extend(["--exclude", pattern])

        # Add search pattern (match all)
        cmd.append(".")

        if exclude_patterns:
            logger.info(f"Excluding patterns: {exclude_patterns}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        files = [f for f in files if f]

        logger.info(f"Found {len(files)} files/symlinks with {fd_bin}")
        return files
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {fd_bin} command: {e.stderr}")
        logger.error(f"{fd_bin} command failed with return code: {e.returncode}")
        raise RuntimeError(f"File discovery failed: {e}")


def parse_size(size_str: str) -> int:
    """Parse size string like '5G', '500M' to bytes."""
    if not size_str:
        return 0

    size_str = size_str.upper().strip()

    # Extract number and unit
    import re
    match = re.match(r'(\d+(?:\.\d+)?)\s*([KMGT]?B?)', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    number, unit = match.groups()
    number = float(number)

    # Convert to bytes
    multipliers = {
        '': 1,
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'M': 1024**2,
        'MB': 1024**2,
        'G': 1024**3,
        'GB': 1024**3,
        'T': 1024**4,
        'TB': 1024**4,
    }

    if unit not in multipliers:
        raise ValueError(f"Unknown size unit: {unit}")

    return int(number * multipliers[unit])


@ray.remote
def filter_broken_symlinks_batch(files: list[str], base_dir: str) -> list[tuple[str, int]]:
    """Ray remote function to filter out broken symlinks and non-existent files."""
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    valid_files_with_sizes = []
    broken_count = 0

    for file_path in files:
        try:
            full_path = os.path.join(base_dir, file_path)
            # Check if file exists (works for regular files and valid symlinks)
            if os.path.exists(full_path):
                try:
                    file_size = os.path.getsize(full_path) if os.path.isfile(full_path) else 0
                except (OSError, IOError):
                    file_size = 0
                valid_files_with_sizes.append((file_path, file_size))
            # For symlinks that don't exist (broken symlinks), check if the symlink itself exists
            elif os.path.islink(full_path):
                # This is a broken symlink - skip it
                broken_count += 1
                logger.debug(f"Skipping broken symlink: {file_path}")
            else:
                # File doesn't exist at all
                broken_count += 1
                logger.debug(f"Skipping missing file: {file_path}")
        except Exception as e:
            broken_count += 1
            logger.debug(f"Error checking file {file_path}: {e}")

    if broken_count > 0:
        logger.info(f"Filtered out {broken_count} broken/missing files from batch of {len(files)}")

    return valid_files_with_sizes


@ray.remote
def get_tar_members_remote(tar_file: str) -> list[str]:
    """Ray remote function to get members of a single tar file."""
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    try:
        logger.info(f"Verifying {os.path.basename(tar_file)}...")
        cmd = ["tar", "-tf", tar_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        files_in_tar = result.stdout.strip().splitlines()
        # Filter out directories (ending with /) and empty entries
        members = [file for file in files_in_tar if file and not file.endswith("/")]
        logger.info(f"  Found {len(members)} files in {os.path.basename(tar_file)}")

        return members
    except Exception as e:
        logger.error(f"Failed to verify {tar_file}: {e}")
        return []

import os
import tarfile
import subprocess
import tempfile
from pathlib import Path
import posixpath as pp

def _norm(p: str) -> str:
    """标准化路径格式以确保验证时的一致性"""
    p = p.replace("\\", "/")
    p = p.lstrip("./")           # 去掉开头的 "./"
    p = pp.normpath(p)           # 折叠 "//"、"a/../b" 等
    return p

def create_tar_part(files: list[str], part_path: str, base_dir: str, max_retry: int = 3) -> bool:
    # Import logger locally to avoid pickle issues
    from loguru import logger
    """Create a tar archive for a subset of files with retry mechanism.

    files: 相对路径列表 (relative to base_dir)
    base_dir: 作为 tar 内部相对路径的根目录
    max_retry: 如果发现missing files，最大重试次数
    """
    try:
        base = Path(base_dir).resolve()
        logger.debug(f"Creating tar part {os.path.basename(part_path)} with {len(files)} files (max_retry={max_retry})")
        logger.debug(f"Base directory: {base}")
    except Exception as e:
        logger.error(f"[create_tar_part] Failed to resolve base directory {base_dir}: {e}")
        return False

    # 1) 处理路径，确保都是相对于base_dir的
    rel_paths = []
    failed_paths = []

    for p in files:
        try:
            # 如果是相对路径，直接使用；如果是绝对路径，转换为相对路径
            if os.path.isabs(p):
                p_abs = Path(p).resolve()
                try:
                    rel = p_abs.relative_to(base)
                    rel_paths.append(rel.as_posix())
                except ValueError:
                    logger.error(f"[create_tar_part] File outside base_dir: {p}")
                    failed_paths.append(p)
            else:
                # 验证相对路径文件是否存在
                full_path = base / p
                # Use lexists to check for symlinks too (exists() follows symlinks)
                if full_path.exists() or os.path.islink(str(full_path)):
                    # 规范化路径，处理特殊字符
                    normalized_path = Path(p).as_posix()
                    rel_paths.append(normalized_path)
                else:
                    logger.warning(f"[create_tar_part] File not found: {p}")
                    failed_paths.append(p)
        except Exception as e:
            logger.error(f"[create_tar_part] Error processing path {p}: {e}")
            failed_paths.append(p)

    if failed_paths:
        logger.error(f"[create_tar_part] {len(failed_paths)} files failed path processing")
        if len(failed_paths) <= 10:
            for fp in failed_paths:
                logger.error(f"  Failed path: {fp}")
        else:
            logger.error(f"  First 5 failed paths: {failed_paths[:5]}")
            logger.error(f"  ... and {len(failed_paths) - 5} more")
        # Do not continue if any files failed - this ensures no silent data loss
        raise RuntimeError(f"Failed to process {len(failed_paths)} files - aborting to prevent data loss")

    if not rel_paths:
        logger.error(f"[create_tar_part] No valid files to archive")
        return False

    # 去重并排序（可选）
    rel_paths = sorted(set(rel_paths))

    # Initialize variables for retry loop (avoid UnboundLocalError)
    missing_from_tar = None
    expected_set = set()
    members_set = set()

    # 2) Create empty tar file first
    try:
        # Create empty tar file
        result = subprocess.run(["tar", "-cf", part_path, "-T", "/dev/null"],
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"[create_tar_part] Failed to create empty tar: {result.stderr}")
            return False
        logger.debug(f"[create_tar_part] Created empty tar file: {part_path}")
    except Exception as e:
        logger.error(f"[create_tar_part] Exception creating empty tar: {e}")
        return False

    # 3) Initialize missing files as the complete set and create path mapping
    norm_to_orig = {_norm(p): p for p in rel_paths}
    missing_from_tar = set(norm_to_orig.keys())

    # 4) Retry loop: incrementally add missing files
    for attempt in range(max_retry + 1):


        if not missing_from_tar:
            # All files successfully added
            logger.info(f"[create_tar_part] Successfully verified {os.path.basename(part_path)}: all {len(rel_paths)} files present")
            break
        

        if attempt > 0:
            logger.warning(f"[create_tar_part] Retry attempt {attempt}/{max_retry} for {os.path.basename(part_path)}")
            # Small delay between retries
            import time
            time.sleep(0.5)

        logger.info(f"[create_tar_part] Attempting to add {len(missing_from_tar)} missing files to {part_path} (attempt {attempt + 1}/{max_retry + 1})")
        missing_sorted = sorted(missing_from_tar)
        logger.debug(f"  Missing files (first 5): {missing_sorted[:5]}")


        # Create temp file list for missing files
        temp_list = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
                try:
                    # Convert normalized missing paths back to original paths for tar command
                    missing_list = [norm_to_orig[norm_path] for norm_path in missing_from_tar]
                    file_list_content = b"\0".join(os.fsencode(s) for s in missing_list)
                    f.write(file_list_content)
                    temp_list = f.name
                    logger.debug(f"[create_tar_part] Written {len(missing_list)} missing file paths to {temp_list}")
                except Exception as e:
                    logger.error(f"[create_tar_part] File encoding error: {e}")
                    continue
            
            # Useful for debugging
            if attempt == 5:
                breakpoint()

            # Append missing files to tar
            cmd = ["tar", "-rf", part_path, "-C", str(base), "--dereference", "--ignore-failed-read", "--null", "-T", temp_list]
            logger.debug(f"[create_tar_part] Running tar append: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            if result.returncode != 0:
                logger.error(f"[create_tar_part] tar append failed with return code {result.returncode}")
                logger.error(f"[create_tar_part] tar stderr: {result.stderr}")

                if "Unexpected EOF" in result.stderr:
                    logger.error(f"[create_tar_part] Unexpected EOF in tar append")
                    subprocess.run(f"rm -rf {part_path}", shell=True, check=False)

                continue  # Try again next attempt

            logger.debug(f"[create_tar_part] tar append completed successfully")

        except Exception as e:
            logger.error(f"[create_tar_part] Exception during tar append: {e}")
            continue
        finally:
            if temp_list and os.path.exists(temp_list):
                try:
                    os.unlink(temp_list)
                except Exception:
                    pass

        # 5) Verify tar contents and update missing_from_tar
        try:
            with tarfile.open(part_path, "r") as tf:
                members_rel = [m.name for m in tf.getmembers() if (m.isfile() or m.issym())]
                logger.debug(f"[create_tar_part] Found {len(members_rel)} files/symlinks in tar")
        except Exception as e:
            logger.error(f"[create_tar_part] Failed to read tar for verification: {e}")
            continue

        # Update missing files set
        expected_set = {_norm(p) for p in rel_paths}
        members_set = {_norm(m) for m in members_rel}
        missing_from_tar = expected_set - members_set
        extra_in_tar = members_set - expected_set

        logger.info(f"[create_tar_part] Verification: {len(members_set)} files in tar, {len(missing_from_tar)} still missing")


        # Log progress
        if missing_from_tar:
            progress = len(rel_paths) - len(missing_from_tar)
            logger.info(f"[create_tar_part] Progress: {progress}/{len(rel_paths)} files added ({100*progress/len(rel_paths):.1f}%)")

        # Report issues for diagnostics (but continue trying)
        if extra_in_tar:
            logger.warning(f"[create_tar_part] Found {len(extra_in_tar)} unexpected files in tar")
            extra_sorted = sorted(extra_in_tar)
            logger.debug(f"  Extra files (first 3): {extra_sorted[:3]}")

    # 6) Final result check
    if missing_from_tar:
        logger.error(f"[create_tar_part] FAILED after {max_retry + 1} attempts: {len(missing_from_tar)} files still missing from {os.path.basename(part_path)}")
        missing_sorted = sorted(missing_from_tar)
        logger.error(f"  Still missing (first 5): {missing_sorted[:5]}")
        if len(missing_from_tar) > 5:
            logger.error(f"  ... and {len(missing_from_tar) - 5} more missing files")

        # Generate diagnostic file for final failure
        debug_file = part_path.replace(".tar", "_debug.log")
        try:
            with open(debug_file, "w") as f:
                f.write(f"TAR CREATION FAILED: {os.path.basename(part_path)}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Attempts: {max_retry + 1}\n")
                f.write(f"Expected files: {len(rel_paths)}\n")
                f.write(f"Files in tar: {len(members_set)}\n")
                f.write(f"Still missing: {len(missing_from_tar)}\n\n")

                f.write("MISSING FILES:\n")
                f.write("-" * 60 + "\n")
                for file in sorted(missing_from_tar):
                    f.write(f"{file}\n")

                f.write("\nEXPECTED FILES:\n")
                f.write("-" * 60 + "\n")
                for file in sorted(rel_paths):
                    f.write(f"{file}\n")

                f.write("\nACTUAL TAR CONTENTS:\n")
                f.write("-" * 60 + "\n")
                for file in sorted(members_set):
                    f.write(f"{file}\n")

            logger.error(f"[create_tar_part] Detailed failure analysis written to: {debug_file}")
        except Exception as e:
            logger.warning(f"[create_tar_part] Failed to write debug file: {e}")

        return False

    # Success!
    logger.info(f"[create_tar_part] SUCCESS: {os.path.basename(part_path)} contains all {len(rel_paths)} expected files")
    return True


@ray.remote(max_retries=3)
def create_tar_parts_batch(file_chunks: list[tuple[list[str], str]], base_dir: str) -> list[bool]:
    """Ray remote function to create multiple tar archives."""
    # Re-setup logging in Ray worker
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    results = []
    failed_parts = []
    
    for i, (chunk_files, part_path) in enumerate(file_chunks):
        logger.info(f"Creating tar part {i+1}/{len(file_chunks)}: {os.path.basename(part_path)}")
        logger.info(f"  Files in this part: {len(chunk_files)}")
        
        result = create_tar_part(chunk_files, part_path, base_dir, max_retry=20)
        results.append(result)
        
        if not result:
            failed_parts.append(part_path)
            logger.error(f"Failed to create tar part: {part_path}")
        else:
            logger.info(f"Successfully created: {os.path.basename(part_path)}")
    
    if failed_parts:
        logger.error(f"Failed to create {len(failed_parts)} tar parts:")
        for failed_part in failed_parts:
            logger.error(f"  {failed_part}")
    
    return results


def distribute_files_simple(all_files: list[str], num_parts: int) -> list[list[str]]:
    """Simple file distribution - randomly shuffles and splits into equal chunks."""
    start_time = time.time()

    random.seed(42)
    random.shuffle(all_files)

    # Simple distribution: split into equal chunks
    chunk_size = (len(all_files) + num_parts - 1) // num_parts
    parts = [all_files[i : i + chunk_size] for i in range(0, len(all_files), chunk_size)]

    logger.info(f"Distributed {len(all_files)} files into {len(parts)} parts in {time.time() - start_time:.4f} seconds")

    return parts


def distribute_files_by_size(files_with_sizes: list[tuple[str, int]], shard_size: int) -> list[list[str]]:
    """Distribute files by cumulative size, ensuring no shard exceeds shard_size."""
    start_time = time.time()

    if not files_with_sizes:
        return []

    parts = []
    current_part = []
    current_size = 0

    for file_path, file_size in files_with_sizes:
        # If adding this file would exceed shard_size, start a new shard
        if current_part and (current_size + file_size > shard_size):
            parts.append(current_part)
            current_part = []
            current_size = 0

        current_part.append(file_path)
        current_size += file_size

    # Add the last part if it has files
    if current_part:
        parts.append(current_part)

    logger.info(f"Distributed {len(files_with_sizes)} files into {len(parts)} size-based parts in {time.time() - start_time:.4f} seconds")

    return parts


def main():
    parser = argparse.ArgumentParser(
        description="Compress directory into tar parts with parallel processing"
    )
    parser.add_argument(
        "--num-shards",
        "-n",
        type=int,
        default=None,
        help="Number of shards to create (mutually exclusive with --shard-size)",
    )
    parser.add_argument(
        "--shard-size",
        type=str,
        default=None,
        help="Max size per shard (e.g., '5G', '500M') (mutually exclusive with --num-shards)",
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=".",
        help="Directory to compress (default: current directory)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help='Space-separated list of patterns to exclude (e.g., "videos_raw *.png __pycache__")',
    )

    args = parser.parse_args()

    # Validate mutual exclusivity
    if args.num_shards and args.shard_size:
        logger.error("Cannot specify both --num-shards and --shard-size")
        return 1
    if not args.num_shards and not args.shard_size:
        logger.error("Must specify either --num-shards or --shard-size")
        return 1

    # Setup logging
    setup_logging(args.log_dir)

    # Initialize Ray
    try:
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"RAY_DEBUG": "legacy"}})
        logger.info("Ray initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        return 1

    logger.info(f"Starting compression with Ray parallel processing")
    logger.info(f"Target directory: {args.directory}")
    logger.info(f"Log directory: {args.log_dir}")

    # Convert directory to absolute path
    args.directory = os.path.abspath(args.directory)

    # Parse exclude patterns from space-separated string
    exclude_patterns = args.exclude.split() if args.exclude else []
    # Always exclude the .archive directory
    exclude_patterns.append(".archive")

    # Get all files first (before creating .archive to avoid interference)
    logger.info("Step 1: Collecting all files...")
    all_files = get_all_files(args.directory, exclude_patterns=exclude_patterns)

    if not all_files:
        logger.error("No files found to archive")
        ray.shutdown()
        return 1

    # Global deduplication and sorting for stability
    original_count = len(all_files)
    all_files = sorted(set(all_files))
    if len(all_files) != original_count:
        logger.info(f"Removed {original_count - len(all_files)} duplicate files during global deduplication")

    # Filter out broken symlinks using Ray parallel processing and collect file sizes
    logger.info("Step 1.5: Filtering broken symlinks and collecting file sizes...")
    if all_files:
        batch_size = 100  # 100 files per Ray task for better parallelization
        batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]

        filter_tasks = [filter_broken_symlinks_batch.remote(batch, args.directory) for batch in batches]
        filtered_batches = ray.get(filter_tasks)

        filtered_files_with_sizes = [item for batch in filtered_batches for item in batch]

        if len(filtered_files_with_sizes) != len(all_files):
            removed_count = len(all_files) - len(filtered_files_with_sizes)
            logger.info(f"Filtered out {removed_count} broken symlinks/missing files")

        # Extract files for compatibility with existing code
        all_files = [f for f, _ in filtered_files_with_sizes]

        # Print out largest files
        largest_files = sorted(filtered_files_with_sizes, key=lambda x: x[1], reverse=True)
        logger.info(f"Largest files: {[(_[0], _[1] / 1024 / 1024 / 1024) for _ in largest_files[:100]]}")

    else:
        filtered_files_with_sizes = []

    logger.info(f"Found {len(all_files)} valid files to archive")

    # Setup parts directory inside the target directory (after file scanning)
    parts_dir = os.path.join(args.directory, ".archive")

    # If .archive already exists, remove it completely
    if os.path.exists(parts_dir):
        logger.info(f"Removing existing archive directory: {parts_dir}")
        shutil.rmtree(parts_dir)

    # Create fresh archive directory
    os.makedirs(parts_dir)

    # Distribute files based on chosen method
    logger.info("Step 2: Distributing files into parts...")
    if args.shard_size:
        shard_size_bytes = parse_size(args.shard_size)
        logger.info(f"Using size-based distribution with max {args.shard_size} ({shard_size_bytes:,} bytes) per shard")
        file_chunks = distribute_files_by_size(filtered_files_with_sizes, shard_size_bytes)
    else:
        num_shards = args.num_shards or os.cpu_count() or 4
        logger.info(f"Using count-based distribution with {num_shards} shards")
        file_chunks = distribute_files_simple(all_files, num_parts=num_shards)
    logger.info(f"Split {len(all_files)} files into {len(file_chunks)} parts")
    
    # Verify no files were lost during distribution
    total_files_in_chunks = sum(len(chunk) for chunk in file_chunks)
    if total_files_in_chunks != len(all_files):
        logger.error(f"CRITICAL: File count mismatch during distribution! Expected {len(all_files)}, got {total_files_in_chunks}")
        ray.shutdown()
        return 1

    # Create partial archives
    logger.info("Step 3: Creating partial uncompressed archives...")
    file_chunks_with_paths = [
        (chunk, os.path.join(parts_dir, f"part_{i:04d}.tar")) for i, chunk in enumerate(file_chunks)
    ]
    
    # Create one Ray task per tar file for reliable processing
    tasks = [create_tar_parts_batch.remote([(chunk, part_path)], args.directory)
             for chunk, part_path in file_chunks_with_paths]
    
    # Wait for all tasks to complete
    try:
        all_results = ray.get(tasks)
        results = [r[0] for r in all_results]  # Each task returns a list with one result
    except Exception as e:
        logger.error(f"Ray task execution failed: {e}")
        ray.shutdown()
        return 1

    if not all(results):
        logger.error("Some parts failed to create")
        failed_count = sum(1 for r in results if not r)
        logger.error(f"Failed to create {failed_count} out of {len(results)} tar parts")
        ray.shutdown()
        return 1

    # Generate manifest for integrity verification
    logger.info("Step 4: Generating manifest for integrity verification...")
    manifest = {
        "total_files": len(all_files),
        "files": all_files,
        "tar_parts": [f"part_{i:04d}.tar" for i in range(len(file_chunks))],
        "timestamp": datetime.datetime.now().isoformat(),
        "source_directory": args.directory
    }
    
    manifest_path = os.path.join(parts_dir, "manifest.json")
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written to {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to write manifest: {e}")
        ray.shutdown()
        return 1

    # Verify integrity by checking tar contents using Ray
    logger.info("Step 5: Verifying archive integrity with Ray parallel processing...")
    
    # Create Ray tasks for each tar file verification
    tar_paths = [os.path.join(parts_dir, f"part_{i:04d}.tar") for i in range(len(file_chunks))]
    verify_tasks = [get_tar_members_remote.remote(tar_path) for tar_path in tar_paths]
    
    # Wait for all verification tasks to complete
    try:
        logger.info(f"Running {len(verify_tasks)} parallel verification tasks...")
        start_verify = time.time()
        all_tar_members = ray.get(verify_tasks)
        verify_time = time.time() - start_verify
        logger.info(f"Verification completed in {verify_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Ray verification tasks failed: {e}")
        ray.shutdown()
        return 1
    
    # Aggregate all archived files and check for cross-part duplicates
    archived_files = set()
    from collections import Counter
    counter = Counter()

    for i, members in enumerate(all_tar_members):
        if not members:
            logger.error(f"Failed to verify part_{i:04d}.tar - no members returned")
            ray.shutdown()
            return 1
        archived_files.update(members)
        counter.update(members)
        logger.debug(f"part_{i:04d}.tar contains {len(members)} files")

    # Report cross-part duplicates (useful for optimization)
    duplicated_across_parts = [p for p, c in counter.items() if c > 1]
    if duplicated_across_parts:
        logger.warning(f"Detected {len(duplicated_across_parts)} paths duplicated across parts (first 5): {duplicated_across_parts[:5]}")
        if len(duplicated_across_parts) > 5:
            logger.warning(f"  ... and {len(duplicated_across_parts) - 5} more duplicated paths")

    logger.info(f"Total unique files archived: {len(archived_files)}")
    
    # Check for missing files
    missing_files = set(all_files) - archived_files
    if missing_files:
        logger.error(f"CRITICAL: {len(missing_files)} files were not archived!")
        for f in list(missing_files)[:10]:
            logger.error(f"  Missing: {f}")
        if len(missing_files) > 10:
            logger.error(f"  ... and {len(missing_files) - 10} more files")
        ray.shutdown()
        return 1
    
    logger.success(f"Successfully verified all {len(all_files)} files in archive")
    print(f"Done. Final archive is {args.directory}/.archive")
    
    # Clean shutdown
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())