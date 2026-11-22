"""Archive issue-level outputs, provenance and calibration from a completed replay."""
import argparse
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from src.e1.dataset import DIGEST, MODULES, TRAIN_END, VALID_END, dump
from src.e1.evidence import DOCS
from src.e1.replay import SEED


def archive(directory, run, output):
    rows = json.loads((directory / "issues.json").read_text())
    labels = json.loads((directory / "labels.json").read_text())
    docs = json.loads((directory / "descriptions.json").read_text())
    frozen, syntax_failures = [], 0
    for row in rows:
        cached = json.loads((directory / "evidence" / (row["id"] + ".json")).read_text())
        syntax_failures += cached["syntax_failures"]
        label = labels[row["id"]]
        frozen.append(dict(issue=row["id"], split=row["split"], as_of=row["opened"],
                           created=row["created"], merged=label["merged"], base=row["base"],
                           owners=",".join(MODULES[i] for i in label["owners"])))
    pd.DataFrame(frozen).to_csv(output / "cohort.csv", index=False)
    for name in ("replay.csv", "predictions.csv"):
        raw = (run / name).read_bytes()
        (output / (name + ".gz")).write_bytes(gzip.compress(raw, mtime=0))
    shutil.copyfile(run / "diagnostics.json", output / "diagnostics.json")
    for name in ("excluded.json", "merges.json"):
        shutil.copyfile(directory / name, output / name)
    counts = Counter(row["split"] for row in rows)
    revision = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", "src/e1"], text=True).strip()
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
               if p.suffix in (".csv", ".gz", ".json", ".txt", ".svg") and p.name != "manifest.json"}
    dump(output / "manifest.json", dict(source_revision=revision, python=platform.python_version(),
         dataset="SWE-bench/SWE-bench", revision="c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2",
         file="data/test-00000-of-00001.parquet", sha256=DIGEST, repository="django/django",
         train_cutoff=TRAIN_END, validation_cutoff=VALID_END, split_counts=counts, modules=MODULES,
         docs=list(DOCS), docs_revision=docs["base"], seeds=dict(policy=SEED, held_out=SEED+1,
         bootstrap=SEED, source_pilot=331), file_snapshots_with_syntax_errors=syntax_failures,
         output_sha256=digests))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("directory", "run", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    archive(args.directory, args.run, args.output)
