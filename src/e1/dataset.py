"""Prepare chronological Django issue inputs and sealed outcome records."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pyarrow.parquet as parquet

DIGEST = "d4f5a245c75319fa8240c540674958c4d491e82edf274b144d43836bdcbc4567"
MODULES = ("db/models", "db/backends", "contrib/admin", "contrib",
           "core", "forms", "template", "utils", "other")
TRAIN_END = "2021-01-01"
VALID_END = "2022-01-01"


def module(path):
    if not path.startswith("django/") or not path.endswith(".py"):
        return None
    relative = path.removeprefix("django/")
    return next((i for i, prefix in enumerate(MODULES[:-1])
                 if relative.startswith(prefix + "/")), len(MODULES) - 1)


def owners(patch):
    paths = re.findall(r"^diff --git a/(\S+) b/(\S+)$", patch, re.M)
    return sorted({i for pair in paths for path in pair if (i := module(path)) is not None})


def dump(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, sort_keys=True) + "\n")
    temporary.replace(path)


def merge_dates(rows, directory):
    path = directory / "merges.json"
    dates = json.loads(path.read_text()) if path.exists() else {}
    missing = [r["instance_id"].split("-")[-1] for r in rows
               if r["instance_id"].split("-")[-1] not in dates]
    for start in range(0, len(missing), 40):
        numbers = missing[start:start + 40]
        fields = " ".join(f"p{n}:pullRequest(number:{int(n)}){{mergedAt}}" for n in numbers)
        query = '{repository(owner:"django",name:"django"){' + fields + '}}'
        raw = subprocess.check_output(["gh", "api", "graphql", "-f", "query=" + query])
        found = json.loads(raw)["data"]["repository"]
        dates.update({n: found["p" + n]["mergedAt"] for n in numbers})
        dump(path, dates)
    return dates


def prepare(directory):
    corpus = directory / "test.parquet"
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected corpus checksum")
    cols = ["repo", "instance_id", "created_at", "base_commit", "problem_statement", "patch"]
    rows = parquet.read_table(corpus, columns=cols, filters=[("repo", "=", "django/django")]).to_pylist()
    dates = merge_dates(rows, directory)
    inputs, labels, excluded = [], {}, {}
    for row in sorted(rows, key=lambda r: (r["created_at"], r["instance_id"])):
        key = row["instance_id"]
        opened = row["created_at"]
        merged = dates[key.split("-")[-1]]
        partition = "train" if opened < TRAIN_END else "validation" if opened < VALID_END else "test"
        deadline = {"train": TRAIN_END, "validation": VALID_END, "test": "9999"}[partition]
        target = owners(row["patch"])
        if not target or not merged or merged >= deadline:
            excluded[key] = "no production Python" if not target else "unavailable fix by cutoff"
            continue
        inputs.append(dict(id=key, opened=opened, split=partition,
                           base=row["base_commit"], text=row["problem_statement"]))
        labels[key] = dict(owners=target, merged=merged)
    dump(directory / "issues.json", inputs)
    dump(directory / "labels.json", labels)
    dump(directory / "excluded.json", excluded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    prepare(parser.parse_args().directory)
