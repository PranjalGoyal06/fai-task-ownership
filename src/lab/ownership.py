"""Resolve experiment ownership and build label-availability-aware chronological splits."""
from collections import Counter
from pathlib import PurePosixPath

from src.lab.intake import eligible, git
from src.lab.layout import import_path, python_source, under


class Ownership:
    def __init__(self, groups, roots=None):
        self.roots = roots
        self.names = [g["name"].strip() for g in groups]
        if not 2 <= len(groups) <= 9 or not all(self.names) or len(set(self.names)) != len(groups):
            raise ValueError("Choose two to nine agents with distinct, nonempty names.")
        self.rules = {}
        self.groups = []
        for idx, group in enumerate(groups):
            paths = [p.strip().rstrip("/") for p in group["paths"]]
            if not paths:
                raise ValueError(f"Assign a path to {self.names[idx]}.")
            for path in paths:
                parts = path.split("/")
                if (roots is None and parts[0] != "sklearn") or any(p in {"", ".", ".."} for p in parts):
                    raise ValueError(f"Expected a repository-relative source path: {path}")
                if roots is not None and not under(path, roots):
                    raise ValueError(f"Rule outside confirmed source roots: {path}")
                if path in self.rules:
                    raise ValueError(f"Conflicting or repeated rule for {path}.")
                self.rules[path] = idx
            self.groups.append(dict(name=self.names[idx], paths=sorted(paths)))
        self.mentions = [[p.removeprefix("sklearn/").removesuffix(".py") for p in g["paths"] if p != "sklearn"]
                         for g in self.groups]
        if roots is not None:
            self.mentions = [[import_path(p, roots).removesuffix(".py").lower() for p in g["paths"]]
                             for g in self.groups]

    def owner(self, path):
        if self.roots is not None and not (python_source(path) and under(path, self.roots)):
            return None
        if self.roots is None and not eligible(path):
            return None
        while path:
            if path in self.rules:
                return self.rules[path]
            path = path.rpartition("/")[0]
        return None


def source_paths(repo, base):
    return [path for path in git(repo, "ls-tree", "--name-only", "-r", base, "sklearn").decode().splitlines()
            if eligible(path)]


def suggested_groups(paths):
    families = {"Preprocessing": ["preprocessing", "compose", "impute", "feature_extraction"],
                "Evaluation": ["metrics", "model_selection"], "Linear models": ["linear_model", "svm"],
                "Tree models": ["tree", "ensemble"], "Clustering": ["cluster", "neighbors"],
                "Utilities": ["utils"]}
    directories = {str(PurePosixPath(*p.split("/")[:2])) for p in paths if p.count("/") >= 2}
    groups = []
    for name, prefixes in families.items():
        matched = [f"sklearn/{p}" for p in prefixes if f"sklearn/{p}" in directories]
        if matched:
            groups.append(dict(name=name, paths=matched))
    groups.append(dict(name="Other production code", paths=["sklearn"]))
    return groups


def label_cases(scan, ownership):
    cases, excluded = [], {}
    keys = [(c["issue"], c["merge"], " ".join(c["text"].split()).casefold()) for c in scan["cases"]]
    counts = [Counter(key[i] for key in keys) for i in range(3)]
    for case, key in zip(scan["cases"], keys, strict=True):
        if any(counter[value] > 1 for counter, value in zip(counts, key, strict=True)):
            excluded[case["id"]] = "repeated issue, fix or report"
            continue
        owners = {ownership.owner(path) for path in case["paths"]}
        if None in owners:
            excluded[case["id"]] = "unmapped: " + ", ".join(p for p in case["paths"] if ownership.owner(p) is None)
            continue
        cases.append(dict(case, owners=sorted(owners)))
    return cases, excluded


def default_cutoffs(cases):
    if len(cases) < 3:
        raise ValueError("At least three distinct usable fixes are needed for the three periods.")
    return tuple(cases[int(f * len(cases))]["opened"][:10] for f in (.6, .8))


def partition(cases, cutoffs):
    train_end, valid_end = cutoffs
    if train_end >= valid_end:
        raise ValueError("The training cutoff must precede the validation cutoff.")
    kept, excluded = [], {}
    for case in cases:
        opened = case["opened"][:10]
        if opened < train_end:
            split, end = "train", train_end
        elif opened < valid_end:
            split, end = "validation", valid_end
        else:
            split, end = "test", None
        if end and case["merged"][:10] >= end:
            excluded[case["id"]] = f"fix unavailable by {end}"
        else:
            kept.append(dict(case, split=split))
    return kept, excluded


def rule_counts(mapping, first, history):
    """Counts reflect the winning path rule after more-specific overrides."""
    def count(paths):
        counts = Counter()
        for path in set(paths):
            rule = path
            while rule and rule not in mapping.rules:
                rule = rule.rpartition("/")[0]
            if rule:
                counts[rule] += 1
        return counts

    initial, all_paths = count(first), count(history)
    return [{"Agent": mapping.names[index], "Rule": rule, "First snapshot": initial[rule],
             "Across snapshots": all_paths[rule]} for rule, index in mapping.rules.items()]


def verify_rules(rows):
    missing = [row["Rule"] for row in rows if not row["Across snapshots"]]
    if missing:
        raise ValueError("No eligible files resolve to these rules in the selected history: " + ", ".join(missing))


def suggest_from_tree(paths, roots):
    paths = sorted(set(paths))
    if len(paths) < 2:
        raise ValueError("A multi-owner experiment needs at least two eligible Python files.")
    active = [r for r in roots if any(p == r or p.startswith(r + "/") for p in paths)]
    if len(active) > 1:
        candidates = [[r] for r in sorted(active)]
    else:
        prefix = active[0] + "/" if active and active[0] not in paths else ""
        while True:
            folders, loose = set(), []
            for path in paths:
                suffix = path.removeprefix(prefix)
                if "/" in suffix:
                    folders.add(prefix + suffix.split("/")[0])
                else:
                    loose.append(path)
            if len(folders) == 1 and not loose:
                prefix = folders.pop() + "/"
                continue
            candidates = [[p] for p in sorted(folders)]
            if loose and folders:
                candidates.append(loose)
            else:
                candidates.extend([p] for p in loose)
            break
    def size(group):
        return sum(any(p == r or p.startswith(r + "/") for r in group) for p in paths)
    candidates.sort(key=lambda g: (-size(g), g[0]))
    groups = []
    for group in candidates[:8] if len(candidates) > 9 else candidates:
        name = PurePosixPath(group[0]).stem if len(group) == 1 else "Package files"
        groups.append(dict(name=name, paths=group))
    if len(candidates) > 9:
        groups.append(dict(name="Other production code", paths=[p for group in candidates[8:] for p in group]))
    names = Counter(g["name"] for g in groups)
    for group in groups:
        if names[group["name"]] > 1:
            group["name"] += " (" + group["paths"][0] + ")"
    return groups
