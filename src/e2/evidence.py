"""Module-restricted, citation-validated LLM evidence extraction."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

from src.e1.dataset import MODULES, module
from src.e2.providers import LLMProvider, provider_from_environment


class ModuleToolbox:
    """Deterministic source tools restricted to one module and one base commit."""

    def __init__(self, repository: str, module_id: int, base_commit: str):
        self.git = ["git", "--git-dir=" + str(repository)]
        self.module_id = module_id
        self.base_commit = base_commit
        self.prefix = MODULES[module_id].replace("/", ".")

    def files(self) -> list[str]:
        tree = subprocess.check_output(
            self.git + ["ls-tree", "-r", "--name-only", self.base_commit, "django"],
            text=True,
        )
        paths = [
            path
            for path in tree.splitlines()
            if path.endswith(".py") and module(path) == self.module_id
        ]
        return sorted(paths)

    def read(self, path: str, start: int = 1, end: int | None = None) -> dict:
        if module(path) != self.module_id:
            raise ValueError("Cross-module source access is not allowed")
        if start < 1 or (end is not None and end < start):
            raise ValueError("Source line range must be positive and ordered")
        source = subprocess.check_output(
            self.git + ["show", f"{self.base_commit}:{path}"],
            text=True,
            errors="replace",
        )
        lines = source.splitlines()
        start = max(1, int(start))
        end = min(len(lines), int(end if end is not None else len(lines)))
        if start > len(lines):
            raise ValueError("Source line range starts beyond end of file")
        snippet = "\n".join(lines[start - 1:end])
        return {
            "path": path,
            "start_line": start,
            "end_line": end,
            "snippet": snippet,
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }

    def grep(self, query: str, limit: int = 20) -> list[dict]:
        matches: list[dict] = []
        query_re = re.compile(re.escape(query), re.IGNORECASE)
        for path in self.files():
            source = subprocess.check_output(
                self.git + ["show", f"{self.base_commit}:{path}"],
                text=True,
                errors="replace",
            )
            for number, line in enumerate(source.splitlines(), start=1):
                if query_re.search(line):
                    matches.append(
                        self.read(path, max(1, number - 2), number + 2)
                        | {"matched_line": number, "query": query}
                    )
                    if len(matches) >= limit:
                        return matches
        return matches

    def symbols(self) -> list[dict]:
        symbols: list[dict] = []
        for path in self.files():
            source = subprocess.check_output(
                self.git + ["show", f"{self.base_commit}:{path}"],
                text=True,
                errors="replace",
            )
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    end_line = getattr(node, "end_lineno", node.lineno)
                    symbols.append(
                        {
                            "path": path,
                            "name": node.name,
                            "kind": type(node).__name__,
                            "start_line": node.lineno,
                            "end_line": end_line,
                        }
                    )
        return symbols


def validate_findings(
    findings: Iterable[dict],
    toolbox: ModuleToolbox,
    allowed_citations: Iterable[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    allowed_ranges = (
        None if allowed_citations is None else list(allowed_citations)
    )
    for finding in findings:
        claim = str(finding.get("claim", "")).strip()
        citations = finding.get("citations", [])
        if not claim or not isinstance(citations, list) or not citations:
            rejected.append({"finding": finding, "reason": "missing_claim_or_citation"})
            continue
        validated: list[dict] = []
        valid = True
        for citation in citations:
            path = citation.get("path")
            if not isinstance(path, str) or module(path) != toolbox.module_id:
                valid = False
                break
            try:
                start = int(citation.get("start_line", 0))
                end = int(citation.get("end_line", 0))
                source = toolbox.read(path, start, end)
            except (subprocess.SubprocessError, TypeError, ValueError):
                valid = False
                break
            expected = citation.get("snippet")
            if expected is not None and expected != source["snippet"]:
                valid = False
                break
            if allowed_ranges is not None and not any(
                item.get("path") == path
                and int(item.get("start_line", 0)) <= start
                and int(item.get("end_line", 0)) >= end
                for item in allowed_ranges
            ):
                valid = False
                break
            validated.append(
                {
                    "path": path,
                    "start_line": start,
                    "end_line": end,
                    "snippet": source["snippet"],
                    "sha256": source["sha256"],
                }
            )
        if valid:
            accepted.append(
                {
                    "claim": claim,
                    "evidence_type": finding.get("evidence_type", "semantic"),
                    "citations": validated,
                }
            )
        else:
            rejected.append({"finding": finding, "reason": "invalid_or_cross_module_citation"})
    return accepted, rejected


def extract_issue_module(
    repository: str,
    issue: dict,
    module_id: int,
    provider: LLMProvider | None = None,
) -> dict:
    """Extract semantically relevant evidence for one module at one base commit."""
    provider = provider or provider_from_environment()
    toolbox = ModuleToolbox(repository, module_id, issue["base"])

    file_paths = toolbox.files()
    symbols = toolbox.symbols()
    system = (
        "You are a source-evidence extractor. Use only the supplied issue and "
        "pre-fix module snapshot. Never use patches, future commits, pull-request "
        "discussions, tests added after the base commit, or another module's source."
    )

    search_terms_prompt = (
        "Return JSON with exactly one key 'terms'. "
        "Given this issue, propose 3 to 8 short source-search terms likely to find "
        "relevant functions or classes in the listed module. "
        f"Issue: {issue['text']}\n"
        f"Module: {MODULES[module_id]}\n"
        f"Files: {json.dumps(file_paths[:200])}\n"
        f"Symbols: {json.dumps(symbols[:300])}"
    )
    tool_plan = json.loads(provider.complete(system, search_terms_prompt))
    terms = [str(term) for term in tool_plan.get("terms", [])][:8]

    tool_trace: list[dict] = [
        {"tool": "list_files", "arguments": {}, "matches": len(file_paths)},
        {"tool": "list_symbols", "arguments": {}, "matches": len(symbols)},
    ]
    retrieved: list[dict] = []
    for term in terms:
        matches = toolbox.grep(term)
        tool_trace.append({"tool": "grep", "arguments": {"query": term}, "matches": len(matches)})
        retrieved.extend(matches)

    finding_prompt = (
        "Return JSON with exactly one key 'findings'. "
        "Each finding must contain claim, evidence_type, and citations. "
        "Each citation must contain path, start_line, end_line, and snippet. "
        "Only cite the supplied retrieved snippets. "
        "Do not invent facts or cite another module.\n"
        f"Issue: {issue['text']}\n"
        f"Retrieved evidence: {json.dumps(retrieved[:160])}"
    )
    result = json.loads(provider.complete(system, finding_prompt))
    accepted, rejected = validate_findings(
        result.get("findings", []),
        toolbox,
        allowed_citations=retrieved,
    )

    semantic_features = [
        float(len(accepted)),
        float(sum(finding["evidence_type"] == "trace_match" for finding in accepted)),
        float(sum(finding["evidence_type"] == "symbol_match" for finding in accepted)),
        float(len(rejected)),
    ]
    return {
        "issue_id": issue["id"],
        "base": issue["base"],
        "module_id": module_id,
        "findings": accepted,
        "rejected": rejected,
        "tool_trace": tool_trace,
        "semantic_features": semantic_features,
        "provider": getattr(provider, "model", type(provider).__name__),
        "model_cutoff": os.getenv("E2_MODEL_CUTOFF", "unknown"),
        "prompt_hashes": {
            "search_terms": hashlib.sha256(search_terms_prompt.encode("utf-8")).hexdigest(),
            "findings": hashlib.sha256(finding_prompt.encode("utf-8")).hexdigest(),
        },
        "sampling": getattr(provider, "sampling", "provider default"),
    }
