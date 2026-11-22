"""Paired lexical and lexical-plus-semantic probability models for E2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.e1.dataset import MODULES
from src.e1.models import Cases, fit_probabilities


def select_cohort(issues: list[dict], per_split: int) -> list[dict]:
    """Choose the earliest fixed-size issue sample from every frozen partition."""
    selected = []
    for split_name in ("train", "validation", "test"):
        partition = [issue for issue in issues if issue["split"] == split_name]
        selected.extend(partition[:per_split])
    return selected


def semantic_features(directory: Path, ids: list[str]) -> np.ndarray:
    """Load source-matched LLM claim features; claim support is not verified."""
    cache = directory / "e2_evidence"
    features = np.zeros((len(ids), len(MODULES), 4), dtype=float)
    for issue_index, issue_id in enumerate(ids):
        for module_id in range(len(MODULES)):
            path = cache / issue_id / f"{module_id}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing E2 evidence for issue {issue_id}, module {module_id}"
                )
            record = json.loads(path.read_text(encoding="utf-8"))
            values = record.get("semantic_features")
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError(f"Invalid semantic feature vector in {path}")
            features[issue_index, module_id] = np.asarray(values, dtype=float)
    return features


def paired_predictors(
    directory: Path,
    issues_per_split: int = 50,
) -> tuple[Cases, Cases, np.ndarray]:
    """Fit paired local and pooled models on one identical frozen E2 cohort."""
    all_issues = json.loads((directory / "issues.json").read_text(encoding="utf-8"))
    issues = select_cohort(all_issues, issues_per_split)
    ids = [record["id"] for record in issues]
    split = np.asarray([record["split"] for record in issues])
    train, valid = split == "train", split == "validation"

    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    y = np.zeros((len(ids), len(MODULES)), dtype=int)
    for issue_index, issue_id in enumerate(ids):
        y[issue_index, labels[issue_id]["owners"]] = 1

    evidence_rows = [
        json.loads(
            (directory / "evidence" / f"{issue_id}.json").read_text(encoding="utf-8")
        )
        for issue_id in ids
    ]
    lexical = np.asarray([row["features"] for row in evidence_rows], dtype=float)
    semantic = semantic_features(directory, ids)
    degree = np.asarray(
        [
            [
                len(
                    {
                        b if a == module_id else a
                        for a, b, *_ in row["edges"]
                        if module_id in (a, b)
                    }
                )
                for module_id in range(len(MODULES))
            ]
            for row in evidence_rows
        ],
        dtype=int,
    )

    lexical_inputs = [
        StandardScaler().fit(lexical[train, module_id]).transform(
            lexical[:, module_id]
        )
        for module_id in range(len(MODULES))
    ]
    semantic_inputs = [
        np.hstack(
            [
                lexical_inputs[module_id],
                StandardScaler().fit(semantic[train, module_id]).transform(
                    semantic[:, module_id]
                ),
            ]
        )
        for module_id in range(len(MODULES))
    ]
    lexical_local = fit_probabilities(lexical_inputs, y, train, valid)
    semantic_local = fit_probabilities(semantic_inputs, y, train, valid)

    text = [record["text"] for record in issues]
    vocabulary = TfidfVectorizer(max_features=6000, sublinear_tf=True)
    vocabulary.fit([value for value, is_train in zip(text, train) if is_train])
    public = vocabulary.transform(text)
    central = fit_probabilities([public] * len(MODULES), y, train, valid)

    descriptions = json.loads(
        (directory / "descriptions.json").read_text(encoding="utf-8")
    )["docs"]
    description_similarity = cosine_similarity(
        public,
        vocabulary.transform(descriptions),
    )
    stale = fit_probabilities(
        [
            description_similarity[:, [module_id]] * 20
            for module_id in range(len(MODULES))
        ],
        y,
        train,
        valid,
    )

    priors = y[train].mean(axis=0)

    def joint_benchmark(
        inputs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        pooled = np.hstack(inputs)
        distances = (
            ((pooled[:, None, :] - pooled[train][None, :, :]) ** 2)
            .mean(axis=2)
        )
        distances += 1 - cosine_similarity(public, public[train])
        neighbor_count = min(25, int(train.sum()))
        neighbors = np.argsort(distances, axis=1, kind="stable")[
            :, :neighbor_count
        ]
        neighbor_distances = np.take_along_axis(distances, neighbors, axis=1)
        weights = np.exp(
            -(neighbor_distances - neighbor_distances.min(axis=1, keepdims=True))
        )
        weights /= weights.sum(axis=1, keepdims=True)
        owner_masks = y[train] @ (1 << np.arange(len(MODULES)))
        return owner_masks[neighbors], weights

    lexical_masks, lexical_weights = joint_benchmark(lexical_inputs)
    semantic_masks, semantic_weights = joint_benchmark(semantic_inputs)

    def make_cases(
        local: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
    ) -> Cases:
        return Cases(
            ids=ids,
            split=split,
            local=local,
            central=central,
            descriptions=stale,
            prior=priors,
            joint_masks=masks,
            joint_weights=weights,
            degree=degree,
        )

    return (
        make_cases(lexical_local, lexical_masks, lexical_weights),
        make_cases(semantic_local, semantic_masks, semantic_weights),
        y[valid],
    )
