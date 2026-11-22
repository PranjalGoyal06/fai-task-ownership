"""Bulk bare Git acquisition with cache accounting and atomic clone publication."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
from tempfile import TemporaryDirectory, TemporaryFile

from src.lab.storage import MIB, StorageLimit, lock, usage


def checked_transfer(args, store, progress):
    """Serialize pack writes with the cache budget and stop Git's process group on overflow."""
    with lock(store.cache / ".budget"), TemporaryFile() as errors:
        initial, _, room = store.space()
        store.check(0)
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=errors,
                                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                                start_new_session=os.name == "posix")
        try:
            while True:
                try:
                    proc.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
                growth = usage(store.directory) - initial
                if growth > room:
                    raise StorageLimit("Git transfer exceeded the repository or total cache budget.")
                if shutil.disk_usage(store.cache).free < store.limits.reserve:
                    raise StorageLimit("Git transfer reached the reserved free disk space.")
                progress(f"Receiving Git history · {max(0, growth) / MIB:.1f} MiB received")
                if proc.poll() is not None:
                    break
            if proc.returncode:
                errors.seek(0)
                raise OSError("Git transfer failed: " + errors.read().decode(errors="replace")[-1200:])
        finally:
            if proc.poll() is None:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait()


def acquire(store, url, estimate, revisions, progress):
    repo = store.directory / "history.git"
    with lock(store.directory / ".history-lock"):
        if not repo.exists():
            store.check(estimate)
            with TemporaryDirectory(dir=store.directory, prefix=".clone-") as tmp:
                staging = Path(tmp) / "history.git"
                checked_transfer(["git", "clone", "--bare", "--no-local", "--no-tags", url, str(staging)],
                                 store, progress)
                staging.rename(repo)
        refs = sorted(set(revisions))
        if refs:
            result = batch_check(repo, refs)
            missing = [ref for ref, row in zip(refs, result, strict=True) if row.endswith(" missing")]
            for i in range(0, len(missing), 64):
                checked_transfer(["git", "--git-dir", str(repo), "fetch", "--no-tags", "--no-auto-maintenance",
                                  url, *missing[i:i+64]], store, progress)
    return repo


def batch_check(repo, refs):
    return subprocess.check_output(["git", "--no-lazy-fetch", "--git-dir", str(repo), "cat-file",
                                    "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                                   input="".join(ref + "\n" for ref in refs), text=True).splitlines()


def entries(repo, revision):
    data = subprocess.check_output(["git", "--git-dir", str(repo), "ls-tree", "-r", "-z", "-l", revision])
    files = {}
    for row in data.split(b"\0"):
        if row:
            meta, path = row.split(b"\t", 1)
            mode, kind, sha, size = meta.split()
            if mode in (b"100644", b"100755") and kind == b"blob":
                files[os.fsdecode(path)] = [sha.decode(), int(size)]
    return files


def acquire_selected(store, url, revisions, objects, progress):
    """Fetch shallow pre-fix trees and explicitly requested blobs in bulk Git packs."""
    repo = store.directory / "selected.git"
    with lock(store.directory / ".history-lock"):
        if not repo.exists():
            with TemporaryDirectory(dir=store.directory) as tmp:
                staging = Path(tmp) / "selected.git"
                subprocess.run(["git", "init", "--bare", "-q", staging], check=True)
                staging.rename(repo)
        for kind, ids in (("trees", sorted(set(revisions))), ("source", list(objects))):
            rows = batch_check(repo, ids)
            missing = [sha for sha, row in zip(ids, rows, strict=True) if row.endswith(" missing")]
            if kind == "source":
                store.check(sum(objects[sha] for sha in missing))
            for i in range(0, len(missing), 512):
                batch = missing[i:i+512]
                args = ["git", "--no-lazy-fetch", "-c", "fetch.unpackLimit=1", "--git-dir", str(repo), "fetch",
                        "--no-auto-maintenance", "--no-tags", "--no-write-fetch-head"]
                if kind == "trees":
                    args.extend(["--depth=1", "--filter=blob:none"])
                progress(f"Fetching {kind} · {i+len(batch)}/{len(missing)} objects")
                checked_transfer([*args, url, *batch], store, progress)
        commands = "".join(f"update refs/lab/snapshots/{ref} {ref}\n" for ref in sorted(set(revisions)))
        subprocess.run(["git", "--git-dir", str(repo), "update-ref", "--stdin"], input=commands,
                       text=True, check=True)
    return repo
