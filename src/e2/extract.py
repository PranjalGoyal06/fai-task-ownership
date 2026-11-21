"""CLI for generating cached module-local E2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from src.e1.dataset import MODULES
from src.e2.evidence import extract_issue_module
from src.e2.providers import provider_from_environment


def run(
    directory: Path,
    repository: str,
    max_issues: int | None = None,
    split: str | None = None,
) -> None:
    issues = json.loads((directory / "issues.json").read_text())
    if split is not None:
        issues = [issue for issue in issues if issue["split"] == split]
    if max_issues is not None:
        issues = issues[:max_issues]
    cache = directory / "e2_evidence"
    cache.mkdir(parents=True, exist_ok=True)
    provider = provider_from_environment()
    module_count = len(MODULES)
    failures: list[dict] = []
    extractor_hash = hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("evidence.py").read_bytes()
    ).hexdigest()

    for issue_index, issue in enumerate(issues, start=1):
        print(
            f"Extracting {issue['split']} issue {issue_index}/{len(issues)}: {issue['id']}",
            flush=True,
        )
        issue_cache = cache / issue["id"]
        issue_cache.mkdir(parents=True, exist_ok=True)
        for module_id in range(module_count):
            output = issue_cache / f"{module_id}.json"
            signature = json.dumps(
                {
                    "issue_id": issue["id"],
                    "base": issue["base"],
                    "module_id": module_id,
                    "model": getattr(provider, "model", type(provider).__name__),
                    "cutoff": os.getenv("E2_MODEL_CUTOFF", "unknown"),
                    "issue_text_hash": hashlib.sha256(
                        issue["text"].encode("utf-8")
                    ).hexdigest(),
                    "extractor_hash": extractor_hash,
                },
                sort_keys=True,
            )
            if output.exists():
                cached = json.loads(output.read_text(encoding="utf-8"))
                if cached.get("cache_signature") == signature:
                    continue
            try:
                record = extract_issue_module(repository, issue, module_id, provider)
            except Exception as exc:
                failures.append(
                    {
                        "issue_id": issue["id"],
                        "module_id": module_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                (directory / "e2-extraction-failures.json").write_text(
                    json.dumps(failures, indent=2),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    f"E2 extraction failed for {issue['id']} module {module_id}"
                ) from exc
            record["cache_signature"] = signature
            output.write_text(json.dumps(record, indent=2), encoding="utf-8")

    (directory / "e2-extraction-failures.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("repository")
    parser.add_argument("--max-issues", type=int)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    args = parser.parse_args()
    run(args.directory, args.repository, args.max_issues, args.split)
