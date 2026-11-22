"""Bounded atomic writes shared by repository setup, source downloads and training."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

try:
    import fcntl
except ImportError:
    fcntl = None

from src.lab.preset import ROOT

MIB = 1024 ** 2
LAB_CACHE = ROOT / "results" / "repositories"


class StorageLimit(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    snapshot: int = 100 * MIB
    repository: int = 512 * MIB
    total: int = 2048 * MIB
    reserve: int = 2048 * MIB

    @classmethod
    def configured(cls):
        defaults = cls()
        values = {}
        for field in cls.__dataclass_fields__:
            value = int(os.getenv("LAB_" + field.upper() + "_MIB", getattr(defaults, field) // MIB))
            if value <= 0:
                raise ValueError(f"LAB_{field.upper()}_MIB must be positive")
            values[field] = value * MIB
        return cls(**values)


def usage(directory):
    total = 0
    for path in directory.rglob("*"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        total += max(stat.st_size, getattr(stat, "st_blocks", 0) * 512)
    return total


@contextmanager
def lock(path, *, shared=False, wait=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if fcntl is not None:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(handle, mode if wait else mode | fcntl.LOCK_NB)
            yield
        else:
            import msvcrt
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise BlockingIOError("Cache lock unavailable") from error
            yield


class Store:
    def __init__(self, cache, repository, limits=None):
        if not re.fullmatch(r"[a-zA-Z0-9-]+/[a-zA-Z0-9_.-]+", repository) or repository.split("/")[1] in {".", ".."}:
            raise ValueError("Invalid repository identifier.")
        self.cache = Path(cache)
        self.directory = self.cache.joinpath(*repository.split("/"))
        self.limits = limits or Limits.configured()

    def active(self):
        return lock(self.directory / ".active", shared=True)

    def space(self):
        used = usage(self.directory)
        total = usage(self.cache)
        if self.cache == LAB_CACHE:
            total += sum(usage(p) for p in self.cache.parent.glob("sklearn-*") if p.is_dir())
        available = shutil.disk_usage(self.cache).free
        return used, total, min(self.limits.repository - used, self.limits.total - total,
                                available - self.limits.reserve)

    def check(self, size):
        used, total, room = self.space()
        if size > room:
            raise StorageLimit(f"Storage budget exceeded. Need {size:,} more bytes with {max(room, 0):,} "
                               f"available. Repository uses {used:,} bytes and lab caches use {total:,}. "
                               "Free cache space or adjust the configured storage budget.")

    def write(self, path, data):
        path = Path(path)
        if ".." in path.relative_to(self.directory).parts:
            raise ValueError("Invalid cache path.")
        with lock(self.cache / ".budget"):
            if path.exists():
                return
            block = os.statvfs(self.cache).f_frsize if hasattr(os, "statvfs") else 4096
            allocated = ((len(data) + block - 1) // block) * block
            missing_dirs = sum(not parent.exists() for parent in path.parents)
            self.check(allocated + block * missing_dirs)
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                try:
                    tmp.write(data)
                    tmp.close()
                    Path(tmp.name).replace(path)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

    def json(self, path, value):
        self.write(path, json.dumps(value, separators=(",", ":"), allow_nan=False).encode())

    def clear_downloads(self):
        try:
            with lock(self.directory / ".active", wait=False):
                with lock(self.cache / ".budget"):
                    for name in ("windows", "trees", "blobs", "evidence", "history.git", "selected.git"):
                        target = self.directory / name
                        if target.is_dir():
                            shutil.rmtree(target)
        except BlockingIOError as error:
            raise ValueError("The repository is in use. Retry after the current stage finishes.") from error
