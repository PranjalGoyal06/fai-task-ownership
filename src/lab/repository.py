"""Headless metadata preflight and selective acquisition of public Python snapshots."""
from calendar import monthrange
from src.lab import history
from datetime import date, timedelta
import hashlib
import json
import re
import subprocess
from urllib.parse import quote, urlsplit

from src.e1.evidence import Snapshot
from src.lab.intake import QUERY, read
from src.lab.layout import discover, python_source, under, unsupported_source
from src.lab.storage import LAB_CACHE, Store, StorageLimit

SHA = re.compile(r"[0-9a-f]{40}")


class Interrupted(RuntimeError):
    """An incomplete remote read which may succeed on retry."""


class Unavailable(Interrupted):
    pass


def slug(url):
    u = urlsplit(url.strip())
    value = u.path.rstrip("/").removesuffix(".git")
    if (u.scheme != "https" or u.netloc.lower() != "github.com" or u.query or u.fragment or
            not re.fullmatch(r"/[a-zA-Z0-9-]+/[a-zA-Z0-9_.-]+", value) or value.split("/")[-1] in {".", ".."}):
        raise ValueError("Enter a public repository URL such as https://github.com/owner/project")
    return value[1:].lower()


def oid(value):
    if not SHA.fullmatch(value):
        raise Interrupted("GitHub returned an invalid object ID.")
    return value


class GitHub:
    def __init__(self, repository):
        self.repository = repository
        self.url = "https://github.com/" + repository + ".git"

    def call(self, *args, raw=False):
        try:
            result = subprocess.run(["gh", "api", *args], capture_output=True, timeout=90)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Interrupted(f"GitHub request interrupted: {error}") from error
        if result.returncode:
            detail = result.stderr.decode(errors="replace").strip()
            if "HTTP 404" in detail:
                raise Unavailable(detail)
            raise Interrupted(detail or "GitHub request failed. Check access and retry.")
        if raw:
            return result.stdout
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise Interrupted("Incomplete JSON response from GitHub. Retry the current stage.") from error

    def get(self, suffix=""):
        return self.call(f"repos/{self.repository}" + suffix)

    def merged(self, start, end):
        query = f"repo:{self.repository} is:pr is:merged merged:{start}..{end} sort:created-asc"
        rows, cursor = [], None
        while True:
            args = ["graphql", "-f", f"query={QUERY}", "-f", f"q={query}"]
            if cursor:
                args.extend(["-f", f"after={cursor}"])
            response = self.call(*args)
            if response.get("errors"):
                raise Interrupted(str(response["errors"]))
            page = response["data"]["search"]
            if page["issueCount"] > 1000:
                if start == end:
                    raise ValueError("Over 1,000 merged changes in one day. Choose another scan window.")
                mid = start + (end - start) // 2
                return self.merged(start, mid) + self.merged(mid + timedelta(days=1), end)
            rows.extend(page["nodes"])
            if not page["pageInfo"]["hasNextPage"]:
                if len({r["number"] for r in rows}) != page["issueCount"]:
                    raise Interrupted("Incomplete search results. Retry this batch.")
                return rows
            cursor = page["pageInfo"]["endCursor"]

    def patch_paths(self, number, expected):
        paths = set()
        for page in range(1, (expected + 99) // 100 + 1):
            rows = self.get(f"/pulls/{int(number)}/files?per_page=100&page={page}")
            if len(rows) != min(100, expected - 100 * (page - 1)):
                return None
            for row in rows:
                paths.add(row["filename"])
                if row.get("previous_filename"):
                    paths.add(row["previous_filename"])
        return paths

    def tree(self, revision):
        response = self.get(f"/git/trees/{oid(revision)}?recursive=1")
        if response["truncated"]:
            raise Interrupted("GitHub returned an incomplete file tree. No source downloaded for this tree.")
        return {e["path"]: [oid(e["sha"]), e["size"]] for e in response["tree"]
                if e["type"] == "blob" and e["mode"] in {"100644", "100755"}}

    def history_size(self):
        owner, name = self.repository.split("/")
        result = self.call("graphql", "-f", "query=query($o:String!,$n:String!){repository(owner:$o,name:$n){diskUsage}}",
                           "-f", f"o={owner}", "-f", f"n={name}")
        return result["data"]["repository"]["diskUsage"] * 1024


def history_estimate(store, github):
    if (store.directory / "history.git").exists():
        return 0
    path = store.directory / "history-size.json"
    if not path.exists():
        store.json(path, github.history_size())
    return read(path)


def tree(store, github, revision):
    path = store.directory / "trees" / (oid(revision) + ".json")
    if not path.exists():
        repo = store.directory / "history.git"
        store.json(path, history.entries(repo, revision) if repo.exists() else github.tree(revision))
    return read(path)


def sources(entries, roots, limits):
    selected = {p: v for p, v in entries.items() if python_source(p) and under(p, roots)}
    size = sum(v[1] for v in selected.values())
    if size > limits.snapshot:
        raise StorageLimit(f"Production snapshot is {size:,} bytes, above the {limits.snapshot:,}-byte limit.")
    return selected


def inspect(url, start, end, *, cache=LAB_CACHE, limits=None, client=None):
    repo = slug(url)
    if end < start:
        raise ValueError("The scan end precedes its start.")
    store = Store(cache, repo, limits)
    github = client or GitHub(repo)
    window = store.directory / "windows" / f"{start}_{end}"
    pin = window / "pin.json"
    with store.active():
        if not pin.exists():
            try:
                meta = github.get()
            except Unavailable as error:
                raise ValueError("Repository is missing or is not accessible as a public repository.") from error
            if meta.get("full_name", repo).lower() != repo:
                raise ValueError(f"Repository has moved. Use https://github.com/{meta['full_name']}")
            if meta["private"]:
                raise ValueError("Only public repositories are supported.")
            commit = github.get("/commits/" + quote(meta["default_branch"], safe=""))
            store.json(pin, {"repository": repo, "revision": oid(commit["sha"]), "window": [str(start), str(end)]})
        info = read(pin)
        entries = tree(store, github, info["revision"])
        roots = discover(entries)
        files = sources(entries, roots, store.limits)
        unsupported_examples = sorted(p for p in entries if unsupported_source(p))[:4]
    return dict(info, roots=roots, source_files=len(files), source_bytes=sum(v[1] for v in files.values()),
                unsupported_examples=unsupported_examples)


def batches(start, end):
    while start <= end:
        last = min(end, start.replace(day=monthrange(start.year, start.month)[1]))
        yield start, last
        start = last + timedelta(days=1)


def scan_path(window, roots):
    signature = "layout3-" + hashlib.sha256(json.dumps(roots).encode()).hexdigest()
    return window / signature / "scan.json"


def preflight(info, roots, *, cache=LAB_CACHE, limits=None, client=None, progress=print):
    repo = info["repository"]
    roots = sorted(set(roots))
    if not roots or not set(roots) <= set(info["roots"]):
        raise ValueError("Choose at least one detected production root.")
    github = client or GitHub(repo)
    store = Store(cache, repo, limits)
    window = store.directory / "windows" / "_".join(info["window"])
    path = scan_path(window, roots)
    with store.active():
        if path.exists():
            return read(path)
        rows = []
        for start, end in batches(*map(date.fromisoformat, info["window"])):
            progress(f"Reading merged changes · {start} to {end}")
            batch = window / f"{start}_{end}.json"
            if not batch.exists():
                store.json(batch, github.merged(start, end))
            rows.extend(read(batch))
        refs = {oid(parent["oid"]) for row in rows if row["closingIssuesReferences"]["totalCount"] == 1
                and row["mergeCommit"] for parent in [row["mergeCommit"], *row["mergeCommit"]["parents"]["nodes"]]}
        if refs and not (store.directory / "selected.git").exists() and history_estimate(store, github) <= store.space()[2]:
            history.acquire(store, github.url, history_estimate(store, github), refs, progress)
        cases, excluded, objects = [], {}, {}
        counts = dict(merged=len(rows), linked=0, recoverable=0, python=0, unsupported=0)
        for row in rows:
            number = str(row["number"])
            link = row["closingIssuesReferences"]
            if link["totalCount"] != 1 or link["nodes"][0]["repository"]["nameWithOwner"].lower() != repo:
                excluded[number] = "requires one same-repository closing issue"
                continue
            counts["linked"] += 1
            merge = row["mergeCommit"]
            if not merge or not merge["parents"]["nodes"]:
                excluded[number] = "missing merge or parent"
                continue
            parent = merge["parents"]["nodes"][0]
            files = row["files"]
            if not files:
                excluded[number] = "unavailable changed-path list"
                continue
            reported = {f["path"] for f in files["nodes"]}
            progress(f"Checking pre-fix tree · PR #{number}")
            try:
                before = tree(store, github, parent["oid"])
                after = tree(store, github, merge["oid"])
            except Unavailable:
                excluded[number] = "unavailable Git snapshot"
                continue
            counts["recoverable"] += 1
            delta = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
            if files["totalCount"] != len(reported) or delta != reported:
                # The search result is bounded. The paginated PR endpoint also gives both sides of a rename.
                detail = window / "patches" / (oid(merge["oid"]) + "-" + number + ".json")
                if not detail.exists():
                    full = github.patch_paths(number, files["totalCount"])
                    store.json(detail, sorted(full) if full is not None else None)
                reported = set(read(detail) or [])
            if delta != reported:
                excluded[number] = "changed paths differ from verified tree difference"
                continue
            unsupported = sorted(p for p in delta if unsupported_source(p))
            if unsupported:
                counts["unsupported"] += 1
                examples = ", ".join(unsupported[:4])
                if len(unsupported) > 4:
                    examples += f" (+{len(unsupported) - 4} more)"
                excluded[number] = f"changes unsupported production source: {examples}"
                continue
            changed = sorted(p for p in delta if python_source(p))
            if not changed:
                excluded[number] = "no production Python change"
                continue
            if any(not under(p, roots) for p in changed):
                excluded[number] = "changed Python path outside confirmed roots"
                continue
            selected = sources(before, roots, store.limits)
            if not selected:
                excluded[number] = "no eligible production Python source in pre-fix snapshot"
                continue
            counts["python"] += 1
            issue = link["nodes"][0]
            opened = max(parent["committedDate"], issue["createdAt"])
            if opened >= row["mergedAt"] or (issue["lastEditedAt"] and issue["lastEditedAt"] > opened):
                excluded[number] = "issue text or fix unavailable at replay time"
                continue
            objects.update(selected.values())
            cases.append(dict(id=f"{repo.replace('/', '-')}-{number}", issue=issue["number"], pr=int(number),
                              base=parent["oid"], merge=merge["oid"], opened=opened, merged=row["mergedAt"],
                              title=issue["title"], text=issue["title"] + "\n" + (issue["body"] or ""),
                              paths=changed, compiled=[]))
        cases.sort(key=lambda c: (c["opened"], c["pr"]))
        scan = dict(repository=repo, revision=info["revision"], roots=roots, window=info["window"],
                    cases=cases, excluded=excluded, counts=counts, objects=objects)
        store.json(path, scan)
        return scan


def check_period_support(cases):
    if len(cases) < 3:
        raise ValueError("Too few usable linked changes for three time periods.")
    first_merge = min(c["merged"][:10] for c in cases)
    later = [c for c in cases if c["opened"][:10] > first_merge]
    if not later or max(c["opened"][:10] for c in later) <= min(c["merged"][:10] for c in later):
        raise ValueError("No three chronological periods have available labels. Choose a wider window.")


def missing_bytes(scan, store):
    remaining = {sha: size for sha, size in scan["objects"].items() if not (store.directory / "blobs" / sha).exists()}
    for name in ("history.git", "selected.git"):
        if (store.directory / name).exists():
            rows = history.batch_check(store.directory / name, remaining)
            remaining = {sha: remaining[sha] for sha, row in zip(remaining, rows, strict=True) if row.endswith(" missing")}
    return sum(remaining.values())


def prepare(scan, *, cache=LAB_CACHE, limits=None, client=None, progress=print):
    store = Store(cache, scan["repository"], limits)
    github = client or GitHub(scan["repository"])

    with store.active():
        store.check(0)
        check_period_support(scan["cases"])
        for ref in {c["base"] for c in scan["cases"]}:
            sources(read(store.directory / "trees" / (oid(ref) + ".json")), scan["roots"], store.limits)
        if not any((store.directory / name).exists() for name in ("history.git", "selected.git")) and not missing_bytes(scan, store):
            return
        estimate = history_estimate(store, github)
        refs = [oid(c["base"]) for c in scan["cases"]]
        if estimate > store.space()[2] or (store.directory / "selected.git").exists():
            objects = {oid(k): v for k, v in scan["objects"].items() if not (store.directory / "blobs" / k).exists()}
            repo = history.acquire_selected(store, github.url, refs, objects, progress)
        else:
            repo = history.acquire(store, github.url, estimate, refs, progress)
        objects = scan["objects"]
        for sha, row in zip(objects, history.batch_check(repo, objects), strict=True):
            if row != f"{sha} blob {objects[sha]}" and not (store.directory / "blobs" / sha).exists():
                raise Interrupted(f"Required historical object unavailable: {sha}")


class FileSnapshot(Snapshot):
    def __init__(self, directory, roots):
        self.directory = directory
        self.roots = roots
        self.reader = None

    def files(self, ref, root):
        entries = read(self.directory / "trees" / (oid(ref) + ".json"))
        return [(v[0], p) for p, v in sorted(entries.items()) if python_source(p) and under(p, self.roots)]

    def blob(self, ref):
        old = self.directory / "blobs" / oid(ref)
        if old.exists():
            return old.read_bytes().decode("utf-8", errors="replace")
        if self.reader is None:
            self.reader = Snapshot(self.directory / ("selected.git" if (self.directory / "selected.git").exists() else "history.git"))
        return self.reader.blob(ref)

    def close(self):
        self.parse.cache_clear()
        if self.reader is not None:
            self.reader.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    year = date.today().year - 1
    parser.add_argument("--start", type=date.fromisoformat, default=date(year, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(year, 12, 31))
    parser.add_argument("--roots", nargs="+", help="Detected roots to include")
    parser.add_argument("--prepare", action="store_true", help="Download preflighted source versions")
    args = parser.parse_args()
    info = inspect(args.url, args.start, args.end)
    if info["unsupported_examples"]:
        print("Only Python source is analyzed. Unsupported source present: " +
              ", ".join(info["unsupported_examples"]))
    store = Store(LAB_CACHE, info["repository"])
    estimate = history_estimate(store, GitHub(info["repository"]))
    print(f"Estimated additional full bare history: {estimate:,} bytes; available budget: {store.space()[2]:,} bytes.")
    if not args.prepare:
        parser.exit(message="Use --prepare to scan the issue history and acquire the Git objects.\n")
    scan = preflight(info, args.roots or info["roots"])
    store = Store(LAB_CACHE, scan["repository"])
    size = history_estimate(store, GitHub(scan["repository"])) if missing_bytes(scan, store) else 0
    print(f"{len(scan['cases'])} usable changes; {scan['counts']['unsupported']} excluded for unsupported "
          f"production source; {size:,} estimated additional Git history bytes.")
    check_period_support(scan["cases"])
    prepare(scan)
    print(scan_path(store.directory / "windows" / "_".join(scan["window"]), scan["roots"]))
