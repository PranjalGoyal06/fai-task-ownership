from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
import hashlib
import subprocess

from src.lab import history

import pytest

from src.lab.layout import discover, import_path
from src.lab.ownership import Ownership, rule_counts, suggest_from_tree, verify_rules
from src.lab.repository import (Interrupted, check_period_support, inspect, missing_bytes, preflight,
                                prepare, slug)
from src.lab.storage import Limits, Store

START, END = date(2024, 1, 1), date(2024, 3, 31)


class Repository:
    def __init__(self):
        self.blobs = {}
        self.downloads = []
        self.searches = []
        self.trees = {}
        self.rows = []
        self.interrupt = False
        before = {"src/shop/core.py": self.file(b"from . import store\n"),
                  "src/shop/store.py": self.file(b"def load(): pass\n")}
        for i, month in enumerate(range(1, 4), 1):
            parent, merge = f"{2*i:040x}", f"{2*i+1:040x}"
            self.trees[parent] = before
            after = dict(before)
            after["src/shop/store.py"] = self.file(f"def load(): return {i}\n".encode())
            self.trees[merge] = after
            before = after
            day = f"2024-{month:02d}-01"
            self.rows.append(dict(number=i, mergedAt=day + "T03:00:00Z",
                                  mergeCommit=dict(oid=merge, parents=dict(nodes=[
                                      dict(oid=parent, committedDate=day + "T02:00:00Z")])),
                                  files=dict(totalCount=1, nodes=[dict(path="src/shop/store.py")]),
                                  closingIssuesReferences=dict(totalCount=1, nodes=[
                                      dict(number=i+10, title=f"Store {i}", body="load", lastEditedAt=None,
                                           createdAt=day + "T01:00:00Z", repository=dict(nameWithOwner="test/shop"))])))
        self.head = merge

    def file(self, raw):
        sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        self.blobs[sha] = raw
        return [sha, len(raw)]

    def get(self, suffix=""):
        return {"private": False, "default_branch": "main"} if not suffix else {"sha": self.head}

    def tree(self, ref):
        return self.trees[ref]

    def merged(self, start, end):
        self.searches.append(start.month)
        if self.interrupt and start.month == 2:
            raise Interrupted("quota")
        return [self.rows[start.month - 1]]

    url = "fixture"

    def history_size(self):
        return sum(map(len, self.blobs.values()))


@pytest.fixture
def fake_history(monkeypatch):
    def configure(remote):
        def acquire(store, url, estimate, revisions, progress):
            store.check(estimate)
            repo = store.directory / "history.git"
            if not repo.exists():
                subprocess.run(["git", "init", "--bare", "-q", repo], check=True)
                for raw in remote.blobs.values():
                    subprocess.run(["git", "--git-dir", repo, "hash-object", "-w", "--stdin"],
                                   input=raw, capture_output=True, check=True)
            return repo
        monkeypatch.setattr(history, "acquire", acquire)
        monkeypatch.setattr(history, "entries", lambda repo, ref: remote.tree(ref))
    return configure


def test_preflight_resume_selective_downloads_and_cleanup(tmp_path, fake_history):
    remote = Repository()
    fake_history(remote)
    kw = dict(cache=tmp_path, client=remote)
    info = inspect("https://github.com/test/shop.git", START, END, **kw)
    assert info["roots"] == ["src/shop"] and info["source_files"] == 2
    remote.interrupt = True
    with pytest.raises(Interrupted):
        preflight(info, info["roots"], **kw)
    remote.interrupt = False
    scan = preflight(info, info["roots"], **kw)
    assert remote.searches == [1, 2, 2, 3]
    store = Store(tmp_path, "test/shop")
    assert not remote.downloads
    remaining = missing_bytes(scan, store)
    assert remaining == 0
    with pytest.raises(ValueError, match="budget"):
        prepare(scan, limits=replace(Limits(), repository=store.space()[0]-1), **kw)
    assert not remote.downloads
    prepare(scan, **kw)
    prepare(scan, **kw)
    assert not remote.downloads and (store.directory / "history.git").is_dir()
    saved = store.directory / "runs" / "one" / "inputs.json.gz"
    store.write(saved, b"saved")
    with store.active(), pytest.raises(ValueError, match="in use"):
        store.clear_downloads()
    store.clear_downloads()
    assert saved.read_bytes() == b"saved"
    assert all(not (store.directory / d).exists() for d in ("blobs", "trees", "windows"))


def test_oversized_and_incomplete_preflight_never_fetch_source(tmp_path):
    remote = Repository()
    with pytest.raises(ValueError, match="snapshot"):
        inspect("https://github.com/test/shop", START, END, cache=tmp_path,
                client=remote, limits=replace(Limits(), snapshot=1))
    assert not remote.downloads
    with pytest.raises(ValueError, match="three chronological"):
        check_period_support([{"opened": "2024-01-01", "merged": "2024-01-03"}] * 4)


def test_simultaneous_writers_share_one_budget(tmp_path):
    stores = [Store(tmp_path, "test/" + name) for name in ("one", "two")]
    payload = b"x" * 65536
    for store in stores:
        (store.directory / "evidence").mkdir(parents=True)
    before = stores[0].space()[1]
    for store in stores:
        store.limits = Limits(total=before + len(payload) + 8192)
    def write(store):
        try:
            store.write(store.directory / "evidence" / "row", payload)
            return True
        except ValueError as error:
            assert "budget" in str(error)
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, stores)) == [False, True]
    assert stores[0].space()[1] <= stores[0].limits.total


def test_layout_groups_and_exclusions():
    paths = ["src/shop/a/x.py", "src/shop/b/y.py", "src/shop/config.py", "src/shop/__init__.py"]
    roots = discover(paths + ["docs/conf.py", "tests/test_shop.py", "docs/web/app.js", "setup.py"])
    assert roots == ["src/shop"]
    assert import_path(paths[0], roots) == "shop/a/x.py"
    groups = suggest_from_tree(paths, roots)
    assert {g["name"] for g in groups} == {"a", "b", "Package files"}
    mapping = Ownership(groups, roots)
    verify_rules(rule_counts(mapping, paths, paths))
    assert all(mapping.owner(p) is not None for p in paths)
    flat = [f"mod{i}.py" for i in range(15)]
    many = suggest_from_tree(flat, discover(flat))
    assert len(many) == 9 and len(many[-1]["paths"]) == 7
    assert many == suggest_from_tree(list(reversed(flat)), discover(flat))
    assert len(suggest_from_tree(paths+["other/service.py"], ["src/shop", "other"])) == 2
    with pytest.raises(ValueError, match="two eligible"):
        suggest_from_tree(["only.py"], ["only.py"])
    assert discover(["lib.py", "src/lib.pyx"]) == ["lib.py"]
    assert discover(["app/api.py", "frontend/main.tsx"]) == ["app"]
    for files in [["index.ts"], ["README.md"]]:
        with pytest.raises(ValueError, match="No eligible"):
            discover(files)
    with pytest.raises(ValueError):
        slug("https://github.com/o/../r")


def test_changed_file_lists_include_both_sides_of_a_rename(tmp_path, fake_history):
    remote = Repository()
    fake_history(remote)
    merge = remote.rows[-1]["mergeCommit"]["oid"]
    old, new = "src/shop/store.py", "src/shop/db.py"
    remote.trees[merge][new] = remote.trees[merge].pop(old)
    remote.rows[-1]["files"]["nodes"] = [{"path": new}]
    remote.patch_paths = lambda number, expected: {old, new}
    kw = dict(cache=tmp_path, client=remote)
    info = inspect("https://github.com/test/shop", START, END, **kw)
    scan = preflight(info, info["roots"], **kw)
    assert scan["cases"][-1]["paths"] == sorted([old, new])


@pytest.mark.parametrize("patch", ["python", "compiled", "mixed", "deleted", "renamed"])
def test_preflight_with_compiled_components(tmp_path, fake_history, patch):
    remote = Repository()
    fake_history(remote)
    compiled = "src/shop/_libs/algos.pyx"
    declaration = "src/shop/_libs/algos.pxd"
    auxiliary = "docs/example.cpp"
    source = "src/shop/store.py"
    for entries in remote.trees.values():
        entries[compiled] = remote.file(b"cdef int counter")
        entries[declaration] = remote.file(b"cdef int version")
    row = remote.rows[-1]
    previous = remote.trees[row["mergeCommit"]["parents"]["nodes"][0]["oid"]]
    latest = remote.trees[remote.head]
    reported = {source}
    if patch == "python":
        latest[auxiliary] = remote.file(b"// documentation example")
        reported.add(auxiliary)
    elif patch in {"compiled", "mixed"}:
        latest[compiled] = remote.file(b"cdef long counter")
        reported.add(compiled)
        if patch == "compiled":
            latest[source] = previous[source]
            reported.remove(source)
    else:
        removed = latest.pop(compiled)
        if patch == "renamed":
            latest[auxiliary] = removed
            reported.add(auxiliary)
            remote.patch_paths = lambda number, expected: reported | {compiled}
        else:
            reported.add(compiled)
    row["files"] = dict(totalCount=len(reported), nodes=[dict(path=p) for p in sorted(reported)])

    kw = dict(cache=tmp_path, client=remote)
    info = inspect("https://github.com/test/shop", START, END, **kw)
    assert info["roots"] == ["src/shop"]
    assert info["source_files"] == 2
    assert declaration in info["unsupported_examples"]
    assert auxiliary not in info["unsupported_examples"]

    scan = preflight(info, info["roots"], **kw)
    assert [c["pr"] for c in scan["cases"]] == ([1, 2, 3] if patch == "python" else [1, 2])
    assert all(c["paths"] == [source] for c in scan["cases"])
    assert scan["counts"]["unsupported"] == (patch != "python")
    assert all(remote.trees[ref][declaration][0] not in scan["objects"] for ref in remote.trees)
    if patch != "python":
        assert scan["excluded"]["3"] == f"changes unsupported production source: {compiled}"
        with pytest.raises(ValueError, match="Too few usable linked changes"):
            prepare(scan, **kw)
    else:
        prepare(scan, **kw)
    assert preflight(info, info["roots"], **kw) == scan
    assert remote.searches == [1, 2, 3]


def test_reserve_and_failed_blob_validation(tmp_path, monkeypatch, fake_history):
    from src.lab import storage

    remote = Repository()
    fake_history(remote)
    kw = dict(cache=tmp_path, client=remote)
    info = inspect("https://github.com/test/shop", START, END, **kw)
    scan = preflight(info, info["roots"], **kw)
    original = storage.shutil.disk_usage(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(storage.shutil, "disk_usage", lambda _: original._replace(free=Limits().reserve-1))
        with pytest.raises(ValueError, match="budget"):
            prepare(scan, **kw)
    assert not remote.downloads
    scan["objects"]["a"*40] = 7
    with pytest.raises(Interrupted, match="object unavailable"):
        prepare(scan, **kw)
    assert not list((Store(tmp_path, "test/shop").directory / "blobs").glob("*"))


def test_prepared_flat_and_src_evidence_reaches_shared_replay(tmp_path, fake_history):
    from src.lab.preset import load_prepared
    from src.lab.replay import Settings, replay
    from src.lab.training import train

    remote = Repository()
    fake_history(remote)
    for row in remote.rows:
        row["closingIssuesReferences"]["nodes"][0]["body"] = "shop.core calls the store"
    kw = dict(cache=tmp_path, client=remote)
    info = inspect("https://github.com/test/shop", START, END, **kw)
    scan = preflight(info, info["roots"], **kw)
    prepare(scan, **kw)
    groups = suggest_from_tree(["src/shop/core.py", "src/shop/store.py"], info["roots"])
    path = train(scan, groups, ["2024-02-01", "2024-03-01"], cache=tmp_path)
    preset = load_prepared(path)
    snapshot = next(iter(preset.snapshots.values()))
    assert snapshot["edges"] == [[0, 1, "src/shop/core.py", 1]]
    assert snapshot["features"][0][2:] == [1, 1]
    assert preset.manifest["repository"] == "test/shop"
    assert preset.outcomes.tolist() == [[0, 1]]
    assert len(replay(preset, Settings(strategic_agents=1)).issues) == 1


def test_bulk_clone_and_legacy_snapshot_read(tmp_path):
    from src.lab.repository import FileSnapshot
    origin = tmp_path / "origin"
    subprocess.run(["git", "init", "-q", origin], check=True)
    (origin / "lib.py").write_text("value = 1\n")
    subprocess.run(["git", "-C", origin, "add", "lib.py"], check=True)
    subprocess.run(["git", "-C", origin, "-c", "user.name=test", "-c", "user.email=test@example.org",
                    "commit", "-qm", "Initial"], check=True)
    rev = subprocess.check_output(["git", "-C", origin, "rev-parse", "HEAD"], text=True).strip()
    store = Store(tmp_path / "cache", "test/project")
    with store.active():
        repo = history.acquire(store, str(origin), 4096, [rev], lambda _: None)
        sha, size = history.entries(repo, rev)["lib.py"]
        assert size == 10 and not (repo / "lib.py").exists()
        reader = FileSnapshot(store.directory, ["lib.py"])
        try:
            assert reader.blob(sha) == "value = 1\n"
        finally:
            reader.close()
        assert history.acquire(store, "unreachable", 0, [rev], lambda _: None) == repo


def test_shallow_bulk_fetch_reuses_packs(tmp_path):
    origin = tmp_path / "remote"
    subprocess.run(["git", "init", "-q", origin], check=True)
    (origin / "one.py").write_text("sample\n")
    (origin / "two.py").write_text("unused\n")
    subprocess.run(["git", "-C", origin, "add", "."], check=True)
    subprocess.run(["git", "-C", origin, "-c", "user.name=test", "-c", "user.email=test@example.org",
                    "commit", "-qm", "Initial"], check=True)
    subprocess.run(["git", "-C", origin, "config", "uploadpack.allowFilter", "true"], check=True)
    rev = subprocess.check_output(["git", "-C", origin, "rev-parse", "HEAD"], text=True).strip()
    objects = subprocess.check_output(["git", "-C", origin, "rev-parse", "HEAD:one.py", "HEAD:two.py"],
                                      text=True).splitlines()
    store = Store(tmp_path / "cache", "test/shallow")
    with store.active():
        repo = history.acquire_selected(store, origin.as_uri(), [rev], {objects[0]: 7}, lambda _: None)
        assert history.batch_check(repo, objects) == [objects[0]+" blob 7", objects[1]+" missing"]
        assert history.acquire_selected(store, "unreachable", [rev], {objects[0]: 7}, lambda _: None) == repo


def test_nested_tools_is_production():
    from src.lab.layout import python_source
    assert python_source("libs/core/pkg/tools/base.py")
    assert not python_source("tools/build.py")
