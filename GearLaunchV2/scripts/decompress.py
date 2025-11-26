#!/usr/bin/env python3
import os
import sys
import shutil
import tarfile
import argparse
import time
import json
from typing import List, Tuple, Optional
from loguru import logger
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
    
    # File handler with DEBUG level
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_filename = f"decompress_{timestamp}_pid{pid}.log"
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


# ---------- Path safety utilities ----------

def _safe_join(root: str, name: str) -> str:
    """Join and normalize ensuring the path stays within root."""
    root = os.path.abspath(root)
    # Reject absolute and null bytes early
    if os.path.isabs(name) or ("\x00" in name):
        raise ValueError(f"Unsafe path (absolute or contains NUL): {name}")
    # Normalize against root
    dest = os.path.normpath(os.path.join(root, name))
    if not (dest == root or dest.startswith(root + os.sep)):
        raise ValueError(f"Unsafe path (escapes root): {name}")
    return dest


def _is_supported_member(m: tarfile.TarInfo) -> bool:
    """We support regular files, directories, and hard links only.

    Note: Symlinks are not supported since compress.py uses --dereference
    and stores actual file content instead of symlinks.
    """
    return m.isreg() or m.isdir() or m.islnk()

# ---------- Tar bomb protection ----------

def _check_tar_bomb(members: List[tarfile.TarInfo], max_total_size: Optional[int] = None) -> None:
    """Check for tar bomb conditions before extraction."""
    from loguru import logger
    
    if max_total_size is None:
        return  # Protection disabled
    
    total_size = 0
    file_count = 0
    
    for member in members:
        if member.isfile():
            total_size += member.size
            file_count += 1
    
    if total_size > max_total_size:
        raise RuntimeError(
            f"Tar bomb protection: total size {total_size:,} bytes exceeds limit {max_total_size:,} bytes "
            f"({file_count:,} files)"
        )
    
    logger.info(f"Tar bomb check passed: {total_size:,} bytes, {file_count:,} files")

def _parse_size_string(size_str: str) -> int:
    """Parse size string like '10GB', '500MB' to bytes."""
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

# ---------- Link deferral utilities ----------

def _save_failed_link(link_info: dict, failed_links_file: str):
    """Save failed link info to a JSON lines file for later retry."""
    import json
    from loguru import logger
    
    try:
        with open(failed_links_file, "a") as f:
            f.write(json.dumps(link_info) + "\n")
    except Exception as e:
        logger.warning(f"Failed to save failed link info: {e}")


def _load_failed_links(failed_links_file: str) -> list:
    """Load failed link info from JSON lines file."""
    import json
    from loguru import logger
    
    failed_links = []
    if not os.path.exists(failed_links_file):
        return failed_links
        
    try:
        with open(failed_links_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    failed_links.append(json.loads(line))
        logger.info(f"Loaded {len(failed_links)} failed links for retry")
    except Exception as e:
        logger.warning(f"Failed to load failed links: {e}")
    
    return failed_links


@ray.remote
def retry_failed_links(failed_links: list, output_dir: str, max_retries: int = 3) -> int:
    """Ray remote function to retry creating failed hardlinks.
    
    Note: Only handles hardlinks since compress.py doesn't create symlinks.
    """
    from loguru import logger
    import time
    import random
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    success_count = 0
    
    for link_info in failed_links:
        link_type = link_info.get("type")
        name = link_info.get("name")
        linkname = link_info.get("linkname")
        
        if not all([link_type, name, linkname]):
            logger.warning(f"Invalid link info: {link_info}")
            continue
            
        # Only process hardlinks now
        if link_type != "hardlink":
            logger.warning(f"Skipping unsupported link type '{link_type}' for {name}")
            continue

        # SAFE JOIN - protect against path traversal attacks
        try:
            dst = _safe_join(output_dir, name)
            link_target = _safe_join(output_dir, linkname)
        except Exception as e:
            logger.warning(f"Unsafe link path for {name} -> {linkname}: {e}")
            continue

        # Ensure parent directory exists
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create parent directory for {name}: {e}")
            continue
        
        for attempt in range(1, max_retries + 1):
            try:
                if not os.path.exists(link_target):
                    if attempt < max_retries:
                        # Target still doesn't exist, wait and retry
                        time.sleep(min(2 ** attempt, 5) + random.uniform(0, 1))
                        continue
                    else:
                        logger.error(f"Hardlink target still missing after {max_retries} attempts: {linkname}")
                        break
                
                # Remove existing file/link if it exists
                if os.path.lexists(dst):
                    os.unlink(dst)
                os.link(link_target, dst)
                logger.debug(f"Retry created hard link: {dst} -> {link_target}")
                success_count += 1
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Hardlink retry attempt {attempt}/{max_retries} failed for {name}: {e}")
                    time.sleep(min(2 ** attempt, 5) + random.uniform(0, 1))
                else:
                    logger.error(f"Failed to create hardlink {name} after {max_retries} attempts: {e}")
    
    logger.info(f"Successfully created {success_count}/{len(failed_links)} deferred hardlinks")
    return success_count


# ---------- Extraction core (tarfile-based) ----------

def _prescan_members(tar_path: str, output_dir: str) -> List[tarfile.TarInfo]:
    # Import logger locally to avoid pickle issues
    from loguru import logger
    """
    Pre-scan tar members:
      - ensure names are safe
      - only keep supported types (regular files, dirs, hardlinks)
    Return the filtered, sorted (dirs first) member list.

    Note: Symlinks are not supported since compress.py uses --dereference.
    """
    members: List[tarfile.TarInfo] = []
    try:
        with tarfile.open(tar_path, mode="r:*") as tf:
            for m in tf.getmembers():
                if not _is_supported_member(m):
                    logger.info(f"[{tar_path}] Skipping unsupported member type: {m.name} ({m.type})")
                    continue
                # Will raise on unsafe
                _ = _safe_join(output_dir, m.name)
                members.append(m)
    except Exception as e:
        raise RuntimeError(f"Pre-scan failed for {tar_path}: {e}")

    # Ensure directories are created before files
    members.sort(key=lambda x: 0 if x.isdir() else 1)
    return members


def _extract_member(tf: tarfile.TarFile, m: tarfile.TarInfo, output_dir: str, failed_links_file: str = None) -> Optional[str]:
    # Import logger locally to avoid pickle issues
    from loguru import logger
    """
    Extract a single member safely. Returns destination path for files, or None for dirs.
    Performs size check for regular files and sets mode/mtime.
    Defers hardlinks that fail due to missing targets.
    
    Note: Symlinks are not handled since compress.py uses --dereference and stores actual file content.
    """
    dst = _safe_join(output_dir, m.name)

    if m.isdir():
        os.makedirs(dst, exist_ok=True)
        try:
            os.chmod(dst, m.mode & 0o777)
        except Exception as e:
            logger.warning(f"Failed to set permissions on directory {dst}: {e}")
        return None

    if m.isreg():
        # Ensure parent exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # Stream copy in chunks
        src_f = tf.extractfile(m)
        if src_f is None:
            raise RuntimeError(f"extractfile() returned None for {m.name}")

        # Write atomically: write to tmp then rename
        tmp_dst = dst + ".partial.__tmp__"
        try:
            with src_f as fin, open(tmp_dst, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)  # 1 MiB chunks

            # Size verification
            st = os.stat(tmp_dst)
            if st.st_size != m.size:
                raise RuntimeError(
                    f"Size mismatch for {m.name}: wrote {st.st_size} vs header {m.size}"
                )
        except Exception:
            # Clean up temp file on any error
            try:
                if os.path.exists(tmp_dst):
                    os.remove(tmp_dst)
            except Exception as cleanup_e:
                logger.warning(f"Failed to clean up temp file {tmp_dst}: {cleanup_e}")
            raise

        # Set mode/mtime then rename into place
        try:
            os.chmod(tmp_dst, m.mode & 0o777)
        except Exception as e:
            logger.warning(f"Failed to set permissions on file {m.name}: {e}")
        try:
            # Set both atime and mtime to m.mtime
            os.utime(tmp_dst, (m.mtime, m.mtime))
        except Exception as e:
            logger.warning(f"Failed to set timestamp on file {m.name}: {e}")

        # Atomic replace
        os.replace(tmp_dst, dst)
        return dst

    if m.islnk():
        # Create hard link with deferral for missing targets
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        link_target = _safe_join(output_dir, m.linkname)
        
        # Remove existing file/link if it exists  
        if os.path.lexists(dst):
            os.unlink(dst)

        # Check if target exists before attempting link
        if not os.path.exists(link_target):
            if failed_links_file:
                logger.debug(f"Deferring hardlink creation (target missing): {m.name} -> {m.linkname}")
                _save_failed_link({
                    "type": "hardlink", 
                    "name": m.name,
                    "linkname": m.linkname
                }, failed_links_file)
                return None  # Mark as deferred, not failed
            else:
                logger.warning(f"Hardlink target missing and no deferral file: {m.linkname}")
                
        try:
            os.link(link_target, dst)
            logger.debug(f"Created hard link: {dst} -> {link_target}")
            return dst
        except Exception as e:
            # If link creation fails and we have deferral enabled, defer it
            if failed_links_file and ("No such file" in str(e) or "not found" in str(e).lower()):
                logger.debug(f"Deferring hardlink creation (link failed): {m.name} -> {m.linkname}")
                _save_failed_link({
                    "type": "hardlink",
                    "name": m.name, 
                    "linkname": m.linkname
                }, failed_links_file)
                return None  # Mark as deferred, not failed
            logger.error(f"Failed to create hard link {dst} -> {link_target}: {e}")
            raise RuntimeError(f"Hard link creation failed for {m.name}: {e}")

    # Unsupported member type - should not reach here due to _is_supported_member
    logger.warning(f"Unsupported tar member type: {m.name} ({m.type})")
    return None


def extract_tar_part(tar_file: str, output_dir: str, max_retries: int = 20, max_total_size: Optional[int] = None) -> Tuple[bool, int, int]:
    # Import logger locally to avoid pickle issues
    from loguru import logger
    """
    Extract a single tar safely using tarfile with pre-scan and per-file verification.
    Returns (success, expected_files, extracted_files) for validation.
    Retries on transient errors with granular file tracking.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create failed links file for this tar part
    failed_links_file = tar_file.replace(".tar", "_failed_links.jsonl")
    # Clean any existing failed links file from previous runs
    if os.path.exists(failed_links_file):
        os.remove(failed_links_file)

    # Pre-scan for safety and to decide the plan
    try:
        members = _prescan_members(tar_file, output_dir)
    except Exception as e:
        logger.error(str(e))
        return False, 0, 0

    if not members:
        logger.info(f"[{tar_file}] No extractable members (after filtering).")
        return True, 0, 0

    # Tar bomb protection check
    try:
        _check_tar_bomb(members, max_total_size)
    except Exception as e:
        logger.error(f"[{tar_file}] {e}")
        return False, 0, 0

    expected_files = len(members)
    
    # Track which members still need extraction
    remaining_members = set(m.name for m in members)
    extracted_count = 0
    
    for attempt in range(1, max_retries + 1):
        if not remaining_members:
            # All files successfully extracted
            logger.info(f"[{os.path.basename(tar_file)}] Successfully extracted all {expected_files} items")
            break
            
        if attempt > 1:
            logger.warning(f"[{tar_file}] Retry attempt {attempt}/{max_retries}")
            # Small delay between retries with jitter
            import random
            time.sleep(min(2 ** (attempt - 1), 10) + random.uniform(0, 1))

        logger.info(f"[{tar_file}] Attempting to extract {len(remaining_members)} remaining items (attempt {attempt}/{max_retries})")
        
        # Track which files we successfully extract in this attempt
        extracted_this_attempt = set()
        
        try:
            with tarfile.open(tar_file, mode="r:*") as tf:
                # Build a dict for fast lookup
                member_dict = {m.name: m for m in members}
                
                # Only attempt to extract remaining members
                for member_name in list(remaining_members):
                    if member_name not in member_dict:
                        logger.warning(f"[{tar_file}] Member {member_name} not found in tar, skipping")
                        remaining_members.discard(member_name)
                        continue
                        
                    member = member_dict[member_name]
                    try:
                        result = _extract_member(tf, member, output_dir, failed_links_file)
                        if result is not None or member.isdir():  # Successfully extracted
                            extracted_this_attempt.add(member_name)
                            logger.debug(f"[{tar_file}] Successfully extracted: {member_name}")
                        elif result is None and member.islnk():
                            # Hardlink was deferred, don't count as extracted yet
                            logger.debug(f"[{tar_file}] Deferred hardlink (not counting as extracted): {member_name}")
                    except Exception as e:
                        logger.warning(f"[{tar_file}] Failed to extract {member_name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"[{tar_file}] Extract attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                continue
            else:
                logger.error(f"[{tar_file}] All {max_retries} attempts failed")
                break
        
        # Update remaining members and extracted count
        remaining_members -= extracted_this_attempt
        extracted_count = expected_files - len(remaining_members)
        
        # Log progress
        if extracted_this_attempt:
            logger.info(f"[{tar_file}] Extracted {len(extracted_this_attempt)} items this attempt. Progress: {extracted_count}/{expected_files} ({100*extracted_count/expected_files:.1f}%)")
        
        # If we successfully extracted everything, break
        if not remaining_members:
            logger.info(f"[{os.path.basename(tar_file)}] Successfully extracted all {expected_files} items")
            break
    
    # Final result check
    if remaining_members:
        logger.error(f"[{tar_file}] FAILED after {max_retries} attempts: {len(remaining_members)} items still not extracted")
        remaining_sorted = sorted(remaining_members)
        logger.error(f"  Still missing (first 5): {remaining_sorted[:5]}")
        if len(remaining_members) > 5:
            logger.error(f"  ... and {len(remaining_members) - 5} more missing items")
            
        # Generate diagnostic file for final failure
        debug_file = tar_file.replace(".tar", "_extract_debug.log")
        try:
            import datetime
            with open(debug_file, "w") as f:
                f.write(f"TAR EXTRACTION FAILED: {os.path.basename(tar_file)}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Attempts: {max_retries}\n")
                f.write(f"Expected items: {expected_files}\n")
                f.write(f"Successfully extracted: {extracted_count}\n")
                f.write(f"Still missing: {len(remaining_members)}\n\n")

                f.write("MISSING ITEMS:\n")
                f.write("-" * 60 + "\n")
                for item in sorted(remaining_members):
                    f.write(f"{item}\n")

            logger.error(f"[{tar_file}] Detailed failure analysis written to: {debug_file}")
        except Exception as e:
            logger.warning(f"[{tar_file}] Failed to write debug file: {e}")
            
        return False, expected_files, extracted_count
    
    # Success!
    logger.info(f"[{os.path.basename(tar_file)}] SUCCESS: extracted all {expected_files} expected items")
    return True, expected_files, extracted_count


# ---------- Ray integration ----------

@ray.remote
def extract_tar_parts_batch(tar_files: List[str], output_dir: str, max_total_size: Optional[int] = None) -> List[Tuple[bool, int, int]]:
    # Re-setup logging in Ray worker
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    results = []
    for tar_file in tar_files:
        result = extract_tar_part(tar_file, output_dir, max_retries=20, max_total_size=max_total_size)
        results.append(result)
    return results


# ---------- Cleanup utilities ----------

@ray.remote
def cleanup_temp_files_remote(directory: str, subdirs: List[str]) -> int:
    """Ray remote function to clean up temp files in specific subdirectories."""
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    count = 0
    for subdir in subdirs:
        search_dir = os.path.join(directory, subdir) if subdir else directory
        if not os.path.exists(search_dir):
            continue
            
        try:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.partial.__tmp__'):
                        temp_file = os.path.join(root, file)
                        try:
                            os.unlink(temp_file)
                            logger.info(f"Cleaned up leftover temp file: {temp_file}")
                            count += 1
                        except Exception as e:
                            logger.warning(f"Failed to clean temp file {temp_file}: {e}")
        except Exception as e:
            logger.warning(f"Failed to scan for temp files in {search_dir}: {e}")
    
    return count


def cleanup_temp_files(directory: str) -> int:
    # Import logger locally to avoid pickle issues
    from loguru import logger
    """Clean up any leftover .partial.__tmp__ files from previous runs."""
    count = 0
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.partial.__tmp__'):
                    temp_file = os.path.join(root, file)
                    try:
                        os.unlink(temp_file)
                        logger.info(f"Cleaned up leftover temp file: {temp_file}")
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to clean temp file {temp_file}: {e}")
    except Exception as e:
        logger.warning(f"Failed to scan for temp files in {directory}: {e}")
    
    return count


# ---------- Verification utilities ----------

def _list_extracted_paths(root: str) -> set:
    """List all extracted files on disk as relative paths.
    
    Note: Only lists regular files since compress.py uses --dereference
    and doesn't create symlinks in the tar archives.
    """
    from pathlib import Path
    from loguru import logger
    
    rootp = Path(root)
    extracted_files = set()
    
    try:
        for p in rootp.rglob("*"):
            try:
                st = p.lstat()  # Don't follow any potential symlinks
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Failed to stat {p}: {e}")
                continue
                
            if p.is_file():
                # Use POSIX relative paths for consistency with manifest
                try:
                    rel_path = p.relative_to(rootp).as_posix()
                    extracted_files.add(rel_path)
                except ValueError as e:
                    logger.warning(f"Path outside root {p}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Failed to scan directory {root}: {e}")
        
    return extracted_files

def _verify_tar_members_on_disk(tar_path: str, output_dir: str) -> list[str]:
    """Verify that all tar members actually exist on disk."""
    from loguru import logger
    
    missing = []
    try:
        with tarfile.open(tar_path, mode="r:*") as tf:
            for m in tf.getmembers():
                if not (m.isreg() or m.islnk() or m.isdir()):
                    continue
                try:
                    dst = _safe_join(output_dir, m.name)
                except Exception:
                    missing.append(m.name)
                    continue
                
                if m.isdir():
                    if not os.path.isdir(dst):
                        missing.append(m.name)
                else:
                    if not os.path.exists(dst):
                        missing.append(m.name)
    except Exception as e:
        logger.error(f"Failed to verify tar members for {tar_path}: {e}")
        # Return all members as missing if we can't read the tar
        try:
            with tarfile.open(tar_path, mode="r:*") as tf:
                missing = [m.name for m in tf.getmembers() if m.isreg() or m.islnk() or m.isdir()]
        except Exception:
            logger.error(f"Cannot read tar file {tar_path} for verification")
            
    return missing


@ray.remote
def verify_extracted_files_on_disk(output_dir: str, subdirs: list = None) -> set:
    """Ray remote function to verify extracted files by scanning actual disk files."""
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    if subdirs is None:
        return _list_extracted_paths(output_dir)

    # If specific subdirs provided, scan each and merge
    all_files = set()
    for subdir in subdirs:
        search_dir = os.path.join(output_dir, subdir) if subdir else output_dir
        if os.path.exists(search_dir):
            subdir_files = _list_extracted_paths(search_dir)
            # Prefix with subdir if not root
            if subdir:
                subdir_files = {f"{subdir}/{f}" for f in subdir_files}
            all_files.update(subdir_files)
    
    logger.info(f"Found {len(all_files)} extracted files on disk")
    return all_files

# This function has been replaced by verify_extracted_files_on_disk
# which verifies actual files on disk instead of tar members


# ---------- CLI & orchestration ----------

def _discover_tar_parts(input_path: str, patterns: Optional[List[str]] = None) -> List[str]:
    """Discover tar part files with flexible naming patterns supporting various compression formats."""
    import glob
    from loguru import logger
    
    if patterns is None:
        patterns = ["*.tar", "*.tar.gz", "*.tgz", "*.tar.xz", "*.tar.bz2", "part_*.tar*"]
    
    search_dir = input_path if os.path.exists(input_path) else input_path
    logger.info(f"Searching for tar parts in: {search_dir}")
    
    tar_files = []
    for pattern in patterns:
        found_files = glob.glob(os.path.join(search_dir, pattern))
        tar_files.extend(f for f in found_files if os.path.isfile(f))
    
    # Remove duplicates and sort
    tar_files = sorted(set(tar_files))
    
    logger.info(f"Found {len(tar_files)} tar parts with patterns {patterns}")
    
    if not tar_files:
        logger.warning(f"No tar files found in {search_dir}")
        # Try to list what's actually there
        try:
            all_files = os.listdir(search_dir)
            logger.info(f"Directory contains {len(all_files)} items: {all_files[:10]}...")
        except Exception:
            pass
    
    return tar_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decompress directory with tar parts into target folder. Handles files, directories, and hardlinks only (no symlinks since compress.py uses --dereference)."
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=".",
        help="Directory to decompress (default: current directory)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for log files (default: logs)"
    )
    parser.add_argument(
        "--max-total-size", type=str, default=None,
        help="Maximum total size for tar bomb protection (e.g., '10GB', '500MB'). Disabled by default."
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_dir)

    # Parse max total size if specified
    max_total_size = None
    if args.max_total_size:
        try:
            max_total_size = _parse_size_string(args.max_total_size)
            logger.info(f"Tar bomb protection enabled: max size {max_total_size:,} bytes ({args.max_total_size})")
        except ValueError as e:
            logger.error(f"Invalid --max-total-size format: {e}")
            return 1
    
    # Initialize Ray
    try:
        ray.init(runtime_env={"pip": ["loguru"]})
        logger.info("Ray initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        return 1

    args.directory = os.path.abspath(args.directory)
    input_path = os.path.join(args.directory, ".archive")
    output_dir = args.directory

    if not os.path.isdir(input_path):
        logger.error(f"Archive directory not found: {input_path}")
        logger.error("Make sure to run compress.py first to create .archive directory")
        return 1

    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up any leftover temporary files from previous runs
    temp_cleaned = cleanup_temp_files(output_dir)
    if temp_cleaned > 0:
        logger.info(f"Cleaned up {temp_cleaned} leftover temporary files")
    
    tar_files = _discover_tar_parts(input_path)
    if not tar_files:
        logger.error(f"No tar part files found in {input_path}")
        ray.shutdown()
        return 1
    
    # Check for manifest file and verify if it exists
    manifest_path = os.path.join(input_path, "manifest.json")
    expected_files = None
    if os.path.exists(manifest_path):
        logger.info(f"Found manifest file: {manifest_path}")
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            expected_files = set(manifest.get("files", []))
            logger.info(f"Manifest indicates {len(expected_files)} files should be extracted")
        except Exception as e:
            logger.warning(f"Failed to read manifest file: {e}")
            # Continue without manifest verification

    logger.info(f"Starting decompression with Ray parallel processing")
    logger.info(f"Target directory: {args.directory}")
    logger.info(f"Archive directory: {input_path}")
    logger.info(f"Found {len(tar_files)} tar parts to extract")
    logger.info("Extracting tar parts...")

    # Use Ray for parallel processing - create one task per tar file for better I/O scheduling
    logger.info(f"Creating {len(tar_files)} Ray tasks (one per tar file)")
    tasks = [extract_tar_parts_batch.remote([tar_file], output_dir, max_total_size) for tar_file in tar_files]
    all_results = ray.get(tasks)
    results = [r[0] for r in all_results]  # Each task returns a list with one result

    # Analyze results and provide detailed statistics
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    
    total_expected = sum(r[1] for r in results)
    total_extracted = sum(r[2] for r in results)
    
    logger.info(f"Extraction summary:")
    logger.info(f"  Total tar files: {len(results)}")
    logger.info(f"  Successful: {len(successes)}")
    logger.info(f"  Failed: {len(failures)}")
    logger.info(f"  Total files expected: {total_expected}")
    logger.info(f"  Total files extracted: {total_extracted}")
    
    if failures:
        # Create pairs to properly map tar files to their results
        pairs = list(zip(tar_files, results))
        fail_pairs = [(tf, r) for tf, r in pairs if not r[0]]

        logger.error(f"Failed to extract {len(fail_pairs)} tar files:")
        for i, (tf, (success, expected, extracted)) in enumerate(fail_pairs):
            if i < 5:  # Show first 5 failures
                logger.error(f"  {tf}: {extracted}/{expected} files")
        if len(fail_pairs) > 5:
            logger.error(f"  ... and {len(fail_pairs) - 5} more failures")
        ray.shutdown()
        return 1
    
    # Verify against manifest if available
    if expected_files is not None:
        logger.info("Verifying extracted files against manifest on disk...")

        # Use the new disk-based verification instead of tar member checking
        verify_start = time.time()
        try:
            extracted_files = _list_extracted_paths(output_dir)
            verify_time = time.time() - verify_start
            logger.info(f"Disk verification completed in {verify_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to verify extracted files on disk: {e}")
            ray.shutdown()
            return 1

        logger.info(f"Total unique files found on disk: {len(extracted_files)}")

        # Check for missing files
        missing_files = expected_files - extracted_files
        if missing_files:
            logger.error(f"CRITICAL: {len(missing_files)} files in manifest not found on disk!")
            for f in list(missing_files)[:10]:
                logger.error(f"  Missing: {f}")
            if len(missing_files) > 10:
                logger.error(f"  ... and {len(missing_files) - 10} more files")
            ray.shutdown()
            return 1

        # Check for extra files (less critical but worth noting)
        extra_files = extracted_files - expected_files
        if extra_files:
            logger.warning(f"Found {len(extra_files)} files not in manifest (new or directories converted to files?)")
            for f in list(extra_files)[:5]:
                logger.warning(f"  Extra: {f}")

        logger.success("Manifest verification passed (on-disk match)")

    if total_extracted != total_expected:
        logger.error(f"CRITICAL: File count mismatch: expected {total_expected}, extracted {total_extracted}")
        ray.shutdown()
        return 1  # This is a critical failure - data integrity issue

    # Post-extraction: retry any deferred links
    logger.info("Step 4: Retrying deferred links...")
    failed_links_files = []
    for tar_file in tar_files:
        failed_links_file = tar_file.replace(".tar", "_failed_links.jsonl")
        if os.path.exists(failed_links_file):
            failed_links_files.append(failed_links_file)

    if failed_links_files:
        logger.info(f"Found {len(failed_links_files)} files with deferred links")

        # Collect all failed links
        all_failed_links = []
        for failed_links_file in failed_links_files:
            failed_links = _load_failed_links(failed_links_file)
            all_failed_links.extend(failed_links)

        if all_failed_links:
            logger.info(f"Retrying {len(all_failed_links)} deferred links using Ray...")

            # Split deferred links into batches for parallel processing
            batch_size = max(10, len(all_failed_links) // 10)  # Use fixed divisor instead of args.jobs
            link_batches = [all_failed_links[i:i + batch_size] for i in range(0, len(all_failed_links), batch_size)]

            # Create Ray tasks for parallel link retry
            retry_tasks = [retry_failed_links.remote(batch, output_dir, max_retries=5) for batch in link_batches]

            try:
                retry_results = ray.get(retry_tasks)
                total_success = sum(retry_results)
                logger.info(f"Successfully created {total_success}/{len(all_failed_links)} deferred links")

                # Clean up failed links files after successful retry
                for failed_links_file in failed_links_files:
                    try:
                        os.remove(failed_links_file)
                    except Exception as e:
                        logger.warning(f"Failed to clean up {failed_links_file}: {e}")

            except Exception as e:
                logger.warning(f"Failed to retry deferred links: {e}")
        else:
            logger.info("No deferred links found to retry")
    else:
        logger.info("No deferred links found")

    # Final verification for cases without manifest - check tar members on disk
    if expected_files is None:
        logger.info("Step 5: Verifying all tar members exist on disk (no manifest available)...")
        all_missing = []
        for tar_file in tar_files:
            missing = _verify_tar_members_on_disk(tar_file, output_dir)
            all_missing.extend(missing)

        if all_missing:
            logger.error(f"CRITICAL: {len(all_missing)} extracted items missing on disk without manifest")
            for f in all_missing[:10]:
                logger.error(f"  Missing: {f}")
            if len(all_missing) > 10:
                logger.error(f"  ... and {len(all_missing) - 10} more missing items")
            ray.shutdown()
            return 1

        logger.info(f"All tar members verified on disk: {sum(len(_verify_tar_members_on_disk(tf, output_dir)) == 0 for tf in tar_files)}/{len(tar_files)} tar files passed")

    logger.success(f"Successfully extracted all parts to {output_dir}")
    
    # Clean shutdown
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())