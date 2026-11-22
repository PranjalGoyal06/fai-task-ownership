"""Acquire the bounded scikit-learn pilot from GitHub and check its historical fixes."""
import argparse
import calendar
import json
import os
import re
from collections import Counter
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory

from src.lab.preset import ROOT

REPO = "scikit-learn/scikit-learn"
URL = "https://github.com/" + REPO
CACHE = ROOT / "results/sklearn-2022"
QUERY = """
query($q:String!, $after:String) {
  search(query:$q,type:ISSUE,first:50,after:$after) {
    issueCount pageInfo {hasNextPage endCursor}
    nodes { ... on PullRequest {
      number mergedAt mergeCommit {oid parents(first:1){nodes{oid committedDate}}}
      closingIssuesReferences(first:2){totalCount nodes{
        number title body createdAt lastEditedAt repository{nameWithOwner}
      }}
      files(first:50){totalCount nodes{path}}
    }}
  }
}
"""


def save(path, value):
    """Publish a complete checkpoint, including when two sessions finish the same stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as tmp:
        try:
            json.dump(value, tmp, separators=(",", ":"), allow_nan=False)
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def read(path):
    return json.loads(path.read_bytes())


def github_month(month):
    end = calendar.monthrange(2022, month)[1]
    query = f"repo:{REPO} is:pr is:merged merged:2022-{month:02}-01..2022-{month:02}-{end} sort:created-asc"
    rows, cursor = [], None
    while True:
        args = ["gh", "api", "graphql", "-f", f"query={QUERY}", "-f", f"q={query}"]
        if cursor:
            args.extend(["-f", f"after={cursor}"])
        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f"GitHub search for 2022-{month:02} failed: {result.stderr.strip()}")
        response = json.loads(result.stdout)
        if response.get("errors"):
            raise ValueError(str(response["errors"]))
        page = response["data"]["search"]
        rows.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            if len(rows) != page["issueCount"] or len({r["number"] for r in rows}) != len(rows):
                raise ValueError("Incomplete monthly search. Retry this month.")
            return rows
        cursor = page["pageInfo"]["endCursor"]


def scan_months(cache, progress, fetch=github_month):
    rows = []
    for month in range(1, 13):
        path = cache / "months" / f"2022-{month:02}.json"
        progress(f"Reading merged pull requests · 2022-{month:02}")
        if not path.exists():
            save(path, fetch(month))
        rows.extend(read(path))
    return rows


def git(repository, *args):
    return subprocess.check_output(["git", "--git-dir", str(repository), *args])


def production(path):
    return path.startswith("sklearn/") and "/tests/" not in path and not Path(path).name.startswith("test_")


def eligible(path):
    return production(path) and path.endswith(".py") and Path(path).name not in {"setup.py", "conftest.py"}


def available_merge(repository, oid):
    local = subprocess.run(["git", "--git-dir", str(repository), "cat-file", "-e", oid + "^{commit}"],
                           stderr=subprocess.DEVNULL)
    if local.returncode == 0:
        return True
    # Some historical merge objects are accessible only by exact ID.
    fetch = subprocess.run(["git", "--git-dir", str(repository), "fetch", "--no-tags", "origin",
                            f"{oid}:refs/lab/merges/{oid}"], capture_output=True, text=True)
    if fetch.returncode and "not our ref" in fetch.stderr:
        return False
    fetch.check_returncode()
    return True


def changed_paths(repository, base, merge):
    fields = git(repository, "diff", "--name-status", "-z", "-M", base, merge).decode().split("\0")
    paths, reported = set(), set()
    i = 0
    while i < len(fields) - 1:
        status, path = fields[i:i + 2]
        i += 2
        paths.add(path)
        if status.startswith("R"):
            path = fields[i]
            i += 1
            paths.add(path)
        reported.add(path)
    return sorted(paths), reported


def crosscheck_swe_bench(cache, parquet):
    """Compare a separately cached reference corpus after freezing the intake records."""
    import pandas as pd

    cases = {c["pr"]: c for c in read(cache / "scan.json")["cases"]}
    scanned = {row["number"] for month in range(1, 13)
               for row in read(cache / "months" / f"2022-{month:02}.json")}
    reference = pd.read_parquet(parquet, columns=["instance_id", "patch"], filters=[("repo", "=", REPO)])
    overlaps = []
    for row in reference.itertuples():
        number = int(row.instance_id.rsplit("-", 1)[1])
        if number in scanned:
            pairs = re.findall(r"^diff --git a/(\S+) b/(\S+)$", row.patch, re.MULTILINE)
            paths = {p for pair in pairs for p in pair if eligible(p)}
            overlaps.append({"pr": number, "intake_eligible": number in cases,
                             "same_python_paths": paths == set(cases[number]["paths"]) if number in cases else None})
    return {"reference_cases": len(reference), "overlapping_records": overlaps}


def prepare_scan(url=URL, cache=CACHE, progress=print):
    if url.strip().removesuffix("/").removesuffix(".git") != URL:
        raise ValueError(f"The pilot supports {URL}")
    rows = scan_months(cache, progress)
    repo = cache / "source.git"
    if not repo.is_dir():
        with TemporaryDirectory(dir=cache) as staging:
            clone = Path(staging) / "source.git"
            subprocess.run(["git", "clone", "--bare", "--no-tags", URL + ".git", str(clone)], check=True)
            if not repo.exists():
                clone.rename(repo)
    revision = git(repo, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    progress("Checking pre-fix revisions and patch paths")
    counts = Counter(merged=len(rows), linked=0, recoverable=0, python=0, compiled=0)
    excluded, cases = {}, []
    for r in rows:
        number = str(r["number"])
        links = r["closingIssuesReferences"]
        if not links["totalCount"]:
            excluded[number] = "no closing issue link"
            continue
        if links["totalCount"] != 1:
            excluded[number] = "ambiguous closing links"
            continue
        if links["nodes"][0]["repository"]["nameWithOwner"] != REPO:
            excluded[number] = "cross-repository closing link"
            continue
        counts["linked"] += 1
        merge = r["mergeCommit"]
        if not merge or not merge["parents"]["nodes"]:
            excluded[number] = "missing merge or pre-fix parent"
            continue
        parent = merge["parents"]["nodes"][0]
        if not available_merge(repo, merge["oid"]):
            excluded[number] = "unavailable snapshot"
            continue
        paths, reported = changed_paths(repo, parent["oid"], merge["oid"])
        counts["recoverable"] += 1
        files = r["files"]
        if not files or len(reported) != files["totalCount"] or reported != {f["path"] for f in files["nodes"]}:
            excluded[number] = "unverified patch path list"
            continue
        python = [path for path in paths if eligible(path)]
        compiled = [p for p in paths if production(p) and Path(p).suffix in {".pyx", ".pxd", ".pxi", ".c", ".cpp", ".h"}]
        if compiled:
            counts["compiled"] += 1
        if not python:
            excluded[number] = "no production Python change"
            continue
        counts["python"] += 1
        issue = links["nodes"][0]
        opened = max(parent["committedDate"], issue["createdAt"])
        if opened >= r["mergedAt"] or (issue["lastEditedAt"] and issue["lastEditedAt"] > opened):
            excluded[number] = "fix or edited issue text unavailable at replay time"
            continue
        cases.append(dict(id=f"sklearn-{number}", pr=int(number), issue=issue["number"], base=parent["oid"],
                          merge=merge["oid"], opened=opened, merged=r["mergedAt"], title=issue["title"],
                          text=issue["title"] + "\n" + issue["body"], paths=python, compiled=compiled))
    cases.sort(key=lambda c: (c["opened"], c["pr"]))
    scan = dict(repository=REPO, revision=revision,
                counts=dict(counts), excluded=excluded, cases=cases)
    save(cache / "scan.json", scan)
    return scan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--cross-check", type=Path, help="Compare the finished scan with a local SWE-bench parquet")
    opts = parser.parse_args()
    print(prepare_scan(cache=opts.cache)["counts"])
    if opts.cross_check:
        print(json.dumps(crosscheck_swe_bench(opts.cache, opts.cross_check), indent=2))
