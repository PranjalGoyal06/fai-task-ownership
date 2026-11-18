"""Check specific directed import citations at their reported pre-fix revision."""
import argparse
import ast
import json
import importlib.util
import random
import warnings
from pathlib import Path

from src.e1.dataset import dump
from src.e1.evidence import Snapshot


def verify(source, ref, path, line, claim):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source.blob(ref + ":" + path))
    except (SyntaxError, ValueError):
        return "unverifiable"
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != line:
            continue
        if isinstance(node, ast.ImportFrom):
            package = path.rpartition("/")[0].replace("/", ".")
            try:
                absolute = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            except ImportError:
                return "unverifiable"
            if claim == absolute or any(claim == f"{absolute}.{alias.name}" for alias in node.names):
                return "confirmed"
        if isinstance(node, ast.Import) and any(alias.name == claim for alias in node.names):
            return "confirmed"
    return "contradicted"


def pilot(directory, repository):
    sample = random.Random(331)
    issues = json.loads((directory / "issues.json").read_text())
    source = Snapshot(repository)
    rows = []
    try:
        for issue in sample.sample([r for r in issues if r["split"] == "test"], 24):
            record = json.loads((directory / "evidence" / (issue["id"] + ".json")).read_text())
            if not record["edges"]:
                rows.append(dict(issue=issue["id"], status="unverifiable"))
                continue
            i, j, path, line, claimed, *_ = sample.choice(record["edges"])
            for target in [claimed, claimed + ".missing_pilot_symbol"]:
                status = verify(source, issue["base"], path, line, target)
                rows.append(dict(issue=issue["id"], base=issue["base"], path=path, line=line,
                                 source=i, neighbor=j, claim=target, altered=target != claimed,
                                 status=status))
    finally:
        source.close()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dump(args.output, pilot(args.directory, args.repository))
