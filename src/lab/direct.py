"""Generate owner probabilities directly from bounded pre-fix source snippets."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from src.e1.replay import comparators
from src.e1.evidence import Snapshot
from src.e1.dataset import MODULES, TRAIN_END, VALID_END
from src.e2.providers import LLMProvider, OllamaProvider
from src.lab.intake import read
from src.lab.ownership import Ownership, label_cases, partition
from src.lab.repository import FileSnapshot
from src.lab.storage import LAB_CACHE, Store
from src.lab.training import extract_one, fingerprint


def direct_prompt(case, name, rate, snippets):
    system = (
        "Estimate whether resolving the historical issue requires a production code change "
        "owned by the named module. Do not infer or describe the later patch. Return JSON only."
    )
    prompt = json.dumps({
        "issue": case["text"], "owner": name, "earlier_owner_rate": rate,
        "source_snippets": snippets,
        "response": {"probability": "number between 0 and 1", "evidence": ["short statement"],
                     "citations": [{"path": "source path", "start_line": 1, "end_line": 1,
                                     "snippet": "exact supplied text"}]},
    }, ensure_ascii=False)
    return system, prompt


def snippets(source, ownership, case, owner, limit=12):
    words = {word.lower() for word in case["text"].replace("/", " ").replace("-", " ").split()
             if len(word) >= 5}
    found = []
    root = ownership.roots[0] if isinstance(source, Snapshot) and len(ownership.roots) == 1 else None
    for oid, path in source.files(case["base"], root):
        if ownership.owner(path) != owner:
            continue
        lines = source.blob(oid).splitlines()
        for number, line in enumerate(lines, 1):
            if words and any(word in line.lower() for word in words):
                first, last = max(1, number - 2), min(len(lines), number + 2)
                found.append({"path": path, "start_line": first, "end_line": last,
                              "snippet": "\n".join(lines[first - 1:last])})
                if len(found) == limit:
                    return found
    return found


def validate(raw, allowed):
    value = json.loads(raw)
    if isinstance(value, dict) and "response" in value and isinstance(value["response"], dict):
        value = {**value["response"], **value}
    probability = value.get("probability") if isinstance(value, dict) else None
    if isinstance(probability, str):
        try:
            probability = float(probability.strip())
        except ValueError:
            probability = None
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("Model response has no numeric probability")
    probability = float(probability)
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model probability is outside 0 to 1")
    allowed_by_range = {(item["path"], item["start_line"], item["end_line"], item["snippet"])
                        for item in allowed}
    citations = []
    for item in value.get("citations", []) if isinstance(value.get("citations", []), list) else []:
        if not isinstance(item, dict):
            continue
        try:
            path, first, last, text = item["path"], int(item["start_line"]), int(item["end_line"]), item["snippet"]
        except (KeyError, TypeError, ValueError):
            continue
        if (path, first, last, text) in allowed_by_range:
            citations.append({"path": path, "start_line": first, "end_line": last, "snippet": text})
    evidence = value.get("evidence", []) if isinstance(value.get("evidence", []), list) else []
    return {"probability": probability, "evidence": [str(item) for item in evidence], "citations": citations}


def _model_details(provider):
    details = {"provider": type(provider).__name__, "model": getattr(provider, "model", type(provider).__name__),
               "endpoint": getattr(provider, "base_url", None), "temperature": 0.0,
               "thinking": False, "model_cutoff": "unknown"}
    if hasattr(provider, "metadata"):
        try:
            details.update(provider.metadata())
        except Exception:
            details["model_digest"] = "unknown"
    return details


def django_setup(root: Path):
    """Adapt the frozen Django E1 inputs to the shared direct-probability flow."""
    inputs, source = root / "results" / "inputs", root / "results" / "django.git"
    if not (inputs / "issues.json").exists() or not source.is_dir():
        raise ValueError("Django direct probabilities need results/inputs and results/django.git. Follow the E1 reproduction guide first.")
    labels = read(inputs / "labels.json")
    cases = []
    for issue in read(inputs / "issues.json"):
        number = int(issue["id"].rsplit("-", 1)[-1])
        merged_ts = labels[issue["id"]]["merged"]
        cases.append({**issue, "issue": number, "pr": number, "merge": merged_ts, "merged": merged_ts,
                      "title": issue["text"].splitlines()[0], "paths": [], "compiled": [],
                      "owners": labels[issue["id"]]["owners"]})
    paths = ["django/db/models", "django/db/backends", "django/contrib/admin", "django/contrib", "django/core",
             "django/forms", "django/template", "django/utils", "django"]
    return ({"repository": "django/django", "revision": "frozen-e1", "roots": ["django"], "window": [TRAIN_END, VALID_END],
             "cases": cases, "excluded": {}, "counts": {}, "source_git": str(source)},
            [{"name": name, "paths": [path]} for name, path in zip(MODULES, paths, strict=True)],
            [TRAIN_END, VALID_END])


def prepare(scan, groups, cutoffs, *, selected_per_period=10, max_issue_length=None, cache=LAB_CACHE, progress=print,
            on_progress=None, provider: LLMProvider | None = None):
    """Return an offline replay input after generating complete owner probabilities."""
    provider = provider or OllamaProvider(model=os.getenv("E2_LLM_MODEL", "qwen2.5-coder:3b"))
    ownership = Ownership(groups, scan["roots"])
    if scan["cases"] and "owners" in scan["cases"][0]:
        eligible, unmapped = scan["cases"], {}
    else:
        eligible, unmapped = label_cases(scan, ownership)
    cases, unavailable = partition(eligible, cutoffs)

    if max_issue_length is None:
        env_val = os.getenv("MAX_ISSUE_LENGTH")
        max_issue_length = int(env_val) if env_val else 5000

    length_excluded = {}
    if max_issue_length and max_issue_length > 0:
        filtered_cases = []
        for case in cases:
            text_len = len(case.get("text", ""))
            if case.get("split") in ("validation", "test") and text_len > max_issue_length:
                length_excluded[case["id"]] = f"issue text exceeds maximum length ({text_len} > {max_issue_length})"
            else:
                filtered_cases.append(case)
        cases = filtered_cases
    if length_excluded:
        print(f"Skipping {len(length_excluded)} issues exceeding length threshold of {max_issue_length} chars: {list(length_excluded.keys())}", flush=True)

    split = {name: [case for case in cases if case["split"] == name] for name in ("train", "validation", "test")}
    if not split["train"] or not split["validation"] or not split["test"]:
        raise ValueError("Choose dates and owners that retain issues in all three periods.")
    selected = {name: rows[:selected_per_period] if selected_per_period else rows for name, rows in split.items()}
    selected["train"] = split["train"]
    names = ownership.names
    rates = np.array([sum(index in case["owners"] for case in split["train"]) / len(split["train"])
                      for index in range(len(names))])
    descriptor = {"repository": scan["repository"], "revision": scan["revision"], "roots": scan["roots"],
                  "groups": ownership.groups, "cutoffs": list(cutoffs), "selection": selected_per_period,
                  "model": _model_details(provider), "protocol": 1}
    run_key = fingerprint(descriptor)
    store = Store(cache, scan["repository"])
    directory = store.directory / "direct" / run_key
    destination = directory / "inputs.json.gz"
    with store.active():
        if destination.exists():
            progress("Restoring saved direct probabilities")
            return destination
        source = Snapshot(Path(scan["source_git"])) if scan.get("source_git") else FileSnapshot(store.directory, scan["roots"])
        records, failures, features = {}, [], {}
        total_eval_cases = sum(len(selected[name]) for name in ("validation", "test"))
        total_steps = total_eval_cases * len(names)
        current_step = 0
        try:
            for period in ("validation", "test"):
                for case_number, case in enumerate(selected[period], 1):
                    progress(f"Reading source and estimating ownership {case_number}/{len(selected[period])} · {case['id']}")
                    print(f"\n[LLM] Issue {case_number}/{len(selected[period])} ({period}): {case['id']}", flush=True)
                    if on_progress:
                        on_progress({
                            "type": "issue_start",
                            "period": period,
                            "case_number": case_number,
                            "total_cases": len(selected[period]),
                            "case_id": case["id"],
                            "title": case.get("title", ""),
                            "agents": names,
                            "overall_step": current_step,
                            "total_steps": total_steps,
                        })
                    features[case["id"]] = extract_one(source, case, ownership)
                    for owner, name in enumerate(names):
                        current_step += 1
                        allowed = snippets(source, ownership, case, owner)
                        system, prompt = direct_prompt(case, name, float(rates[owner]), allowed)
                        signature = fingerprint({"descriptor": descriptor, "issue": case["id"], "base": case["base"],
                                                 "owner": owner, "prompt": prompt})
                        path = directory / "responses" / (signature + ".json")
                        if path.exists():
                            record = read(path)
                            prob = float(record.get("probability", 0.0))
                            print(f"  · {name} ({owner + 1}/{len(names)}): cached (p={prob:.2f})", flush=True)
                            if on_progress:
                                on_progress({
                                    "type": "agent_update",
                                    "period": period,
                                    "case_id": case["id"],
                                    "case_number": case_number,
                                    "total_cases": len(selected[period]),
                                    "owner_index": owner,
                                    "agent": name,
                                    "status": "cached",
                                    "probability": prob,
                                    "elapsed": float(record.get("elapsed_seconds", 0.0)),
                                    "overall_step": current_step,
                                    "total_steps": total_steps,
                                })
                        else:
                            if on_progress:
                                on_progress({
                                    "type": "agent_update",
                                    "period": period,
                                    "case_id": case["id"],
                                    "case_number": case_number,
                                    "total_cases": len(selected[period]),
                                    "owner_index": owner,
                                    "agent": name,
                                    "status": "calling",
                                    "probability": None,
                                    "elapsed": None,
                                    "overall_step": current_step,
                                    "total_steps": total_steps,
                                })
                            attempts = max(1, int(os.getenv("DIRECT_MODEL_ATTEMPTS", "3")))
                            started = time.perf_counter()
                            print(f"  · {name} ({owner + 1}/{len(names)}): calling model ...", end="", flush=True)
                            for attempt in range(1, attempts + 1):
                                try:
                                    temp = 0.0 if attempt == 1 else 0.2
                                    try:
                                        raw_response = provider.complete(system, prompt, temperature=temp)
                                    except TypeError:
                                        raw_response = provider.complete(system, prompt)
                                    response = validate(raw_response, allowed)
                                    elapsed = time.perf_counter() - started
                                    record = {**response, "signature": signature, "issue": case["id"], "owner": owner,
                                              "elapsed_seconds": elapsed, "prompt": prompt,
                                              "raw_response": raw_response,
                                              "attempt": attempt}
                                    store.json(path, record)
                                    print(f" p={response['probability']:.2f} ({elapsed:.2f}s)", flush=True)
                                    if on_progress:
                                        on_progress({
                                            "type": "agent_update",
                                            "period": period,
                                            "case_id": case["id"],
                                            "case_number": case_number,
                                            "total_cases": len(selected[period]),
                                            "owner_index": owner,
                                            "agent": name,
                                            "status": "done",
                                            "probability": response["probability"],
                                            "elapsed": elapsed,
                                            "overall_step": current_step,
                                            "total_steps": total_steps,
                                        })
                                    time.sleep(0.15)
                                    break
                                except Exception as error:
                                    if attempt < attempts:
                                        print(f" retry {attempt + 1}/{attempts} after {type(error).__name__} ...", end="", flush=True)
                                        time.sleep(0.5)
                                        continue
                                    print(f" ERROR: {type(error).__name__}: {error}", flush=True)
                                    failures.append({"issue": case["id"], "owner": name,
                                                     "error": f"{type(error).__name__}: {error}",
                                                     "attempts": attempt})
                                    record = None
                                    if on_progress:
                                        on_progress({
                                            "type": "agent_update",
                                            "period": period,
                                            "case_id": case["id"],
                                            "case_number": case_number,
                                            "total_cases": len(selected[period]),
                                            "owner_index": owner,
                                            "agent": name,
                                            "status": "error",
                                            "error": f"{type(error).__name__}: {error}",
                                            "overall_step": current_step,
                                            "total_steps": total_steps,
                                        })
                            if record is None:
                                continue
                        records[(case["id"], owner)] = record
        finally:
            source.close()
        expected = sum(len(selected[name]) for name in ("validation", "test")) * len(names)
        if len(records) != expected:
            store.json(directory / "failures.json", failures)
            raise ValueError(f"{expected - len(records)} ownership estimates are incomplete. Retry to resume.")
        validation, test = selected["validation"], selected["test"]
        def matrix(rows):
            return np.asarray([[records[(case["id"], owner)]["probability"] for owner in range(len(names))]
                               for case in rows])
        validation_beliefs, beliefs = matrix(validation), matrix(test)
        validation_y = np.asarray([[int(owner in case["owners"]) for owner in range(len(names))] for case in validation])
        outcomes = np.asarray([[int(owner in case["owners"]) for owner in range(len(names))] for case in test])
        degrees = lambda rows: np.asarray([[sum({edge[0], edge[1]}.__contains__(owner) for edge in features[case["id"]]["edges"])
                                             for owner in range(len(names))] for case in rows])
        zero_pool = {reward: [0] * len(test) for reward in (5, 8, 12)}
        direct_evidence = {case["id"]: {str(owner): records[(case["id"], owner)] for owner in range(len(names))}
                           for case in test}
        snapshots = {case["id"]: {"base": case["base"], "features": features[case["id"]]["features"],
                                      "edges": [edge[:4] for edge in features[case["id"]]["edges"]],
                                      "citations": {str(key): value[:3] for key, value in features[case["id"]]["citations"].items()},
                                      "direct_evidence": direct_evidence[case["id"]]} for case in test}
        metric = [{"agent": name, "training_positives": int(sum(owner in case["owners"] for case in split["train"])),
                   "training_total": len(split["train"]), "validation_positives": int(validation_y[:, owner].sum()),
                   "held_out_positives": int(outcomes[:, owner].sum()), "predictor": "direct model", "calibration": "not fitted",
                   "brier": float(np.mean((beliefs[:, owner] - outcomes[:, owner]) ** 2))}
                  for owner, name in enumerate(names)]
        manifest = {**descriptor, "preset": "public-python-direct", "modules": names,
                    "ownership": ownership.groups, "split_counts": {name: len(selected[name]) for name in selected},
                    "model_metrics": metric, "exclusions": {**unmapped, **unavailable, **length_excluded},
                    "max_issue_length": max_issue_length, "skipped_length_count": len(length_excluded),
                    "probability_source": "direct_model", "pooled_reference": False}
        result = {"format": 1, "manifest": manifest,
                  "issues": [{key: case[key] for key in ("id", "title", "issue", "pr", "base")} for case in test],
                  "prior": rates.tolist(), "validation_beliefs": validation_beliefs.tolist(),
                  "validation_degree": degrees(validation).tolist(), "validation_outcomes": validation_y.tolist(),
                  "beliefs": beliefs.tolist(), "degrees": degrees(test).tolist(), "outcomes": outcomes.tolist(),
                  "snapshots": snapshots, "pooled": zero_pool}
        store.write(destination, gzip.compress(json.dumps(result, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
        return destination
