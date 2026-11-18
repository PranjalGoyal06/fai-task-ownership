"""Fit module predictors and public comparators on chronological partitions."""
import json
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.e1.dataset import MODULES


@dataclass
class Cases:
    ids: list
    split: np.ndarray
    local: np.ndarray
    central: np.ndarray
    descriptions: np.ndarray
    prior: np.ndarray
    joint_masks: np.ndarray
    joint_weights: np.ndarray
    degree: np.ndarray


def fit_probabilities(x, y, train, valid):
    predictions = []
    for i in range(y.shape[1]):
        data = x[i]
        labels = y[train, i]
        if len(np.unique(labels)) < 2:
            predictions.append(np.full(len(y), (1 + labels.sum()) / (2 + len(labels))))
            continue
        model = LogisticRegression(max_iter=400).fit(data[train], labels)
        logits = model.decision_function(data).reshape(-1, 1)
        if len(np.unique(y[valid, i])) == 2:
            calibration = LogisticRegression(max_iter=400).fit(logits[valid], y[valid, i])
            predictions.append(calibration.predict_proba(logits)[:, 1])
        else:
            predictions.append(model.predict_proba(data)[:, 1])
    return np.array(predictions).T


def predictors(directory):
    issues = json.loads((directory / "issues.json").read_text())
    labels = json.loads((directory / "labels.json").read_text())
    ids = [r["id"] for r in issues]
    split = np.array([r["split"] for r in issues])
    train, valid = split == "train", split == "validation"
    y = np.zeros((len(ids), len(MODULES)), dtype=int)
    for j in np.flatnonzero(train | valid):
        y[j, labels[ids[j]]["owners"]] = 1
    rows = [json.loads((directory / "evidence" / (key + ".json")).read_text()) for key in ids]
    features = np.array([r["features"] for r in rows])
    degrees = [[len({b if a == i else a for a, b, *_ in r["edges"] if i in (a, b)})
                for i in range(len(MODULES))] for r in rows]
    text = [r["text"] for r in issues]
    vocabulary = TfidfVectorizer(max_features=6000, sublinear_tf=True)
    vocabulary.fit([s for s, earlier in zip(text, train) if earlier])
    public = vocabulary.transform(text)
    docs = json.loads((directory / "descriptions.json").read_text())["docs"]
    similarity = cosine_similarity(public, vocabulary.transform(docs))
    local_inputs = [StandardScaler().fit(features[train, i]).transform(features[:, i])
                    for i in range(len(MODULES))]
    local = fit_probabilities(local_inputs, y, train, valid)
    central = fit_probabilities([public] * len(MODULES), y, train, valid)
    stale = fit_probabilities([similarity[:, [i]] * 20 for i in range(len(MODULES))],
                              y, train, valid)

    # Estimate a joint distribution over whole historical owner sets. Nearby
    # training issues share public vocabulary and pooled module features.
    stacked = np.hstack(local_inputs)
    distances = ((stacked[:, None, :] - stacked[train][None, :, :]) ** 2).mean(axis=2)
    distances += 1 - cosine_similarity(public, public[train])
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :25]
    d = np.take_along_axis(distances, neighbors, axis=1)
    weights = np.exp(-(d - d.min(axis=1, keepdims=True)))
    weights /= weights.sum(axis=1, keepdims=True)
    masks = y[train] @ (1 << np.arange(len(MODULES)))
    return Cases(ids, split, local, central, stale, y[train].mean(axis=0),
                 masks[neighbors], weights, np.array(degrees)), y[valid]
