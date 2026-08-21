import shutil
from pathlib import Path

from dirsync import sync


def _reconcile_sizes(remote_path: Path, local_path: Path) -> int:
    """Re-copy any local file that is missing or whose size differs from the source.

    dirsync only compares modification times, so a truncated/partial local copy that
    happens to have a newer mtime than the source is never repaired. This pass walks
    the source tree and re-copies (shutil.copy2) any file whose local size mismatches
    or is absent. Comparison is stat-only (no content hashing), so it stays cheap.

    Returns the number of files re-copied.
    """
    remote_path = Path(remote_path)
    local_path = Path(local_path)
    recopied = 0
    for src in remote_path.rglob("*"):
        if not src.is_file():
            continue
        dst = local_path / src.relative_to(remote_path)
        try:
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
        except OSError:
            pass  # fall through and re-copy
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Re-copying size-mismatched/missing file: {dst}")
        shutil.copy2(src, dst)
        recopied += 1
    print(f"Size reconciliation complete: {recopied} file(s) re-copied due to size mismatch.")
    return recopied


def _remote_readable(remote_path: Path) -> bool:
    """True only if the remote source dir exists AND can actually be listed.

    A stale FUSE/network mount still 'exists' as a dentry but raises OSError
    ("Transport endpoint is not connected") on access. Probing with a listdir
    distinguishes a live mount from a dead one.
    """
    try:
        if not remote_path.is_dir():
            return False
        next(remote_path.iterdir(), None)  # force an actual access
        return True
    except OSError:
        return False


def sync_from_cloud_if_needed(remote_path: Path, local_path: Path, run_type: str) -> bool:
    """If running in cloud, sync data from network mount to local storage first.

    Degrades gracefully: if the remote source is missing or unreadable (never
    mounted, or a stale mount that dropped its backing server) but a populated
    local copy already exists, skip the sync with a warning and use the local
    data instead of hard-failing.
    """
    if run_type == 'cloud':
        remote_path = Path(remote_path)
        local_path = Path(local_path)
        if not _remote_readable(remote_path):
            local_has_data = local_path.is_dir() and any(local_path.iterdir())
            if local_has_data:
                print(f"WARNING: remote dataset dir {remote_path} is missing or "
                      f"unreadable (stale/absent mount). Using existing local "
                      f"data at {local_path} without syncing.")
                return False
            raise ValueError(
                f"Remote dataset dir {remote_path} is missing/unreadable and no "
                f"local data found at {local_path}. Re-mount the data share or "
                f"point --remote-dataset-dir at a readable source."
            )
        print(f"Cloud run detected. Syncing data from {remote_path} to {local_path}...")
        local_path.mkdir(parents=True, exist_ok=True)
        sync(str(remote_path), str(local_path), 'sync', only_newer=True, verbose=True)
        _reconcile_sizes(remote_path, local_path)
        print(f"Dataset sync complete.")
        return True
    return False


def sync_to_cloud_if_needed(local_path: Path, remote_path: Path, run_type: str) -> bool:
    """If running in cloud, sync trained models from local storage to network mount."""
    if run_type == 'cloud':
        print(f"Cloud run detected. Syncing trained models from {local_path} to {remote_path}...")
        remote_path.mkdir(parents=True, exist_ok=True)
        sync(str(local_path), str(remote_path), 'sync', only_newer=True, verbose=True)
        print(f"Trained models sync complete.")
        return True
    return False
