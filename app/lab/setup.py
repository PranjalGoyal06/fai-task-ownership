"""Guided preparation, training and replay for historical repository experiments."""
from collections import Counter
from datetime import date
from pathlib import Path
import subprocess

import pandas as pd
import streamlit as st

from src.e1.evidence import Snapshot
from src.lab.intake import CACHE, read
from src.lab.ownership import (Ownership, default_cutoffs, label_cases, partition, rule_counts,
                               source_paths, suggest_from_tree, suggested_groups, verify_rules)
from src.lab.repository import FileSnapshot, Interrupted, check_period_support, inspect, GitHub, history_estimate, missing_bytes, preflight, prepare
from src.lab.storage import LAB_CACHE, MIB, Store, StorageLimit, usage
from src.lab.preset import load_prepared
from src.lab.replay import Settings
from src.lab.training import extract_one, train
from src.lab_ui.graph import graph


def stage(name):
    st.session_state["setup_stage"] = name


def open_replay(path):
    preset = load_prepared(path)
    st.session_state.update(active_prepared_path=str(path), historical_view="experiment",
                            historical_run=Settings(strategic_agents=round(len(preset.manifest["modules"]) * 2 / 3)))
    st.switch_page("app/lab/historical.py")


def saved_runs():
    paths = sorted([*(CACHE / "runs").glob("*/inputs.json.gz"),
                    *LAB_CACHE.glob("*/*/runs/*/inputs.json.gz")], key=lambda p: p.stat().st_mtime, reverse=True)
    if paths:
        with st.expander("Open a saved experiment"):
            path = st.selectbox("Prepared run", paths, format_func=lambda p: f"{p.parents[2].name} · {p.parent.name[:12]}")
            manifest = load_prepared(path).manifest
            st.caption(" · ".join(manifest["modules"]) + f" · {manifest['split_counts']['test']} held-out changes")
            if st.button("Open saved replay"):
                open_replay(path)


def repository():
    st.header("Choose a repository")
    url = st.text_input("Public GitHub URL", placeholder="https://github.com/owner/project")
    year = date.today().year - 1
    left, right = st.columns(2)
    start = left.date_input("Merged from", date(year, 1, 1), min_value=date(2008, 1, 1), max_value=date.today())
    end = right.date_input("Merged through", date(year, 12, 31), min_value=date(2008, 1, 1), max_value=date.today())
    if st.button("Check repository", type="primary"):
        with st.spinner("Checking public repository metadata"):
            info = inspect(url, start, end)
        reset_setup()
        st.session_state["setup_info"] = info
        stage("roots")
        st.rerun()
    if (CACHE / "scan.json").exists() and st.button("Review scikit-learn pilot"):
        reset_setup()
        st.session_state["setup_scan"] = read(CACHE / "scan.json")
        stage("suitability")
        st.rerun()
    saved_runs()
    directories = sorted(p.parent for p in LAB_CACHE.glob("*/*/.active"))
    if directories:
        with st.expander("Manage downloaded data"):
            directory = st.selectbox("Repository cache", directories,
                                     format_func=lambda p: str(p.relative_to(LAB_CACHE)))
            st.caption(f"{usage(directory) / MIB:.2f} MiB including saved runs.")
            if st.button("Remove downloads and keep saved replays"):
                Store(LAB_CACHE, str(directory.relative_to(LAB_CACHE))).clear_downloads()
                st.success("Downloaded history, source and evidence cleared.")


def reset_setup():
    for key in list(st.session_state):
        if key.startswith("setup_") or key == "owners_editor":
            del st.session_state[key]


def roots_page():
    info = st.session_state["setup_info"]
    st.header("Check production roots")
    st.caption(f"{info['repository']} · revision {info['revision'][:12]} · "
               f"{info['source_files']} Python files · {info['source_bytes'] / MIB:.2f} MiB")
    if unsupported := info.get("unsupported_examples"):
        st.warning("This repository contains production source outside the Python analysis. "
                   "Fixes that touch unsupported production files are excluded in full. "
                   f"Examples include {', '.join(unsupported)}.")
    roots = st.multiselect("Production paths to include", info["roots"], default=info["roots"])
    store = Store(LAB_CACHE, info["repository"])
    estimate = history_estimate(store, GitHub(info["repository"]))
    st.write(f"Estimated additional bare Git history: **{estimate / MIB:.1f} MiB** · "
             f"available within disk limits: **{max(0, store.space()[2]) / MIB:.1f} MiB**.")
    st.caption("The estimate uses repository storage reported by GitHub. Actual packed size can differ. "
               "History that fits is downloaded without a working checkout. Larger repositories use bounded Git packs for selected revisions.")
    if st.button("Check history and storage", type="primary", disabled=not roots):
        with st.status("Checking linked changes and historical file sizes", expanded=True) as status:
            scan = preflight(info, roots, progress=st.empty().write)
            status.update(label="Metadata complete", state="complete")
        if st.session_state.get("setup_scan", {}).get("roots") != roots:
            for key in ("setup_groups", "setup_cutoffs", "owners_editor"):
                st.session_state.pop(key, None)
        st.session_state["setup_scan"] = scan
        stage("suitability")
        st.rerun()


def suitability(scan):
    counts = scan["counts"]
    st.header("Repository suitability")
    st.table(pd.Series({"Merged pull requests": counts["merged"], "One same-repository closing link": counts["linked"],
                        "Recoverable pre-fix parent": counts["recoverable"], "Verified production Python change": counts["python"],
                        "Available issue text and fix chronology": len(scan["cases"])}, name="Cases"))
    if "roots" in scan:
        st.caption(f"{counts.get('unsupported', 0)} fixes excluded for changes to unsupported production source.")
        store = Store(LAB_CACHE, scan["repository"])
        needed = history_estimate(store, GitHub(scan["repository"])) if missing_bytes(scan, store) else 0
        used, total, available = store.space()
        st.write(f"Estimated additional Git history: **{needed:,} bytes**. "
                 f"Available within storage limits: **{max(0, available):,} bytes**.")
        st.caption(f"Repository cache {used / MIB:.2f} MiB · lab cache {total / MIB:.2f} MiB · "
                   f"limits {store.limits.snapshot // MIB} MiB per snapshot, "
                   f"{store.limits.repository // MIB} MiB per repository, {store.limits.total // MIB} MiB total, "
                   f"{store.limits.reserve // MIB} MiB kept free.")
        if needed > available or (store.directory / "selected.git").exists():
            needed = missing_bytes(scan, store)
            st.info(f"Using selected Git packs. Remaining source content is {needed / MIB:.1f} MiB "
                    "before compression and tree overhead. Actual storage is monitored.")
        store.check(needed)
    else:
        st.caption(f"{counts.get('compiled', 0)} pilot patches included compiled-source changes.")
    with st.expander("Excluded pull requests and reasons"):
        st.dataframe([{"PR": n, "Reason": reason} for n, reason in scan["excluded"].items()], hide_index=True)
    if "roots" in scan:
        try:
            check_period_support(scan["cases"])
        except ValueError as error:
            st.error(str(error))
            return
    if len(scan["cases"]) < 3:
        st.error("Unsuitable for a three-period replay. Too few usable changes survived the scan.")
        return
    st.success("Ready for ownership mapping. The next checks measure coverage and time-period support for your map.")
    if "roots" in scan:
        if st.button("Prepare experiment", type="primary"):
            with st.status("Preparing verified Git history", expanded=True):
                prepare(scan, progress=st.empty().write)
            stage("ownership")
            st.rerun()
    else:
        st.button("Define agents", on_click=stage, args=("ownership",), type="primary")


@st.cache_data(show_spinner=False)
def tree(base, repository, roots):
    if roots is not None:
        source = FileSnapshot(Store(LAB_CACHE, repository).directory, roots)
        return [p for _, p in source.files(base, roots)]
    return source_paths(CACHE / "source.git", base)


@st.cache_data(show_spinner=False)
def preview(case, groups, repository, roots):
    source = FileSnapshot(Store(LAB_CACHE, repository).directory, roots) if roots else Snapshot(CACHE / "source.git")
    try:
        snapshot = extract_one(source, case, Ownership(groups, roots))
        return {"base": snapshot["base"], "edges": [edge[:4] for edge in snapshot["edges"]]}
    finally:
        source.close()


def mapping(scan):
    st.header("Define module owners")
    repository, roots = scan["repository"], scan.get("roots")
    paths = tree(scan["cases"][0]["base"], repository, roots)
    suggestions = suggest_from_tree(paths, roots) if roots else suggested_groups(paths)
    initial = st.session_state.get("setup_groups", suggestions)
    st.write("Edit the suggested groups, add an agent or remove a row to merge agents. "
             "Give each agent one or more comma-separated paths. The most specific file or folder wins.")
    edited = st.data_editor([{"Agent": g["name"], "Paths": ", ".join(g["paths"])} for g in initial],
                            num_rows="dynamic", hide_index=True, key="owners_editor", width="stretch")
    try:
        mapping = Ownership([{"name": (r["Agent"] or "").strip(),
                              "paths": [p.strip() for p in (r["Paths"] or "").split(",") if p.strip()]} for r in edited], roots)
    except ValueError as error:
        st.error(str(error))
        return
    history = {p for base in {c["base"] for c in scan["cases"]} for p in tree(base, repository, roots)}
    rows = rule_counts(mapping, paths, history)
    st.dataframe(rows, hide_index=True)
    try:
        verify_rules(rows)
    except ValueError as error:
        st.error(str(error))
        return
    if any(not row["First snapshot"] for row in rows):
        st.info("Some rules first match a file in a later snapshot.")
    cases, excluded = label_cases(scan, mapping)
    assigned = [mapping.owner(p) for p in paths]
    st.caption(f"{len(mapping.names)} agents · {sum(i is None for i in assigned)} unmapped files at the first snapshot · "
               f"{len(cases)} eligible changes · {len(excluded)} excluded")
    with st.expander("Preview file ownership and excluded changes"):
        st.dataframe([{"Path": p, "Owner": mapping.names[i] if i is not None else "Unmapped"}
                      for p, i in zip(paths, assigned)], hide_index=True)
        if excluded:
            st.dataframe([{"Fix": k, "Reason": v} for k, v in excluded.items()], hide_index=True)
    if len(cases) < 3:
        st.warning("This map leaves fewer than three changes. Assign more production paths.")
        return
    example = cases[0]
    st.subheader("Preview an early change")
    st.write(f"[#{example['pr']} · {example['title']}](https://github.com/{repository}/pull/{example['pr']})")
    st.dataframe([{"Changed path": p, "Required agent": mapping.names[mapping.owner(p)]} for p in example["paths"]],
                 hide_index=True)
    if roots:
        st.caption("The preview graph shows imports resolved within eligible Python source at the pre-fix revision.")
    graph(preview(example, mapping.groups, repository, roots), mapping.names, repository)
    if st.button("Use this ownership map", type="primary"):
        if st.session_state.get("setup_groups") != mapping.groups:
            st.session_state.pop("setup_cutoffs", None)
        st.session_state["setup_groups"] = mapping.groups
        stage("timeline")
        st.rerun()


def timeline(scan):
    st.header("Choose the issue timeline")
    groups = st.session_state["setup_groups"]
    cases, _ = label_cases(scan, Ownership(groups, scan.get("roots")))
    proposed = st.session_state.get("setup_cutoffs", default_cutoffs(cases))
    st.write("Start with approximately 60% for learning, 20% for calibration and reporting strategies, "
             "and 20% for evaluation. Earlier periods can only use changes merged before the next cutoff.")
    st.caption("Cases are dated when both the issue and the first parent of its merge exist. "
               "For a long-running issue this can be much later than the filing date.")
    first, second = st.columns(2)
    cutoffs = [first.date_input("Validation starts (UTC)", date.fromisoformat(proposed[0])).isoformat(),
               second.date_input("Held-out evaluation starts (UTC)", date.fromisoformat(proposed[1])).isoformat()]
    try:
        retained, excluded = partition(cases, cutoffs)
    except ValueError as error:
        st.error(str(error))
        return
    counts = Counter(c["split"] for c in retained)
    st.table([{"Period": split, "Retained changes": counts[split]} for split in ("train", "validation", "test")])
    st.caption(f"{len(excluded)} changes crossed their label-availability cutoff.")
    if excluded:
        with st.expander("Excluded by cutoff"):
            st.dataframe([{"Fix": k, "Reason": v} for k, v in excluded.items()], hide_index=True)
    if len(counts) < 3:
        st.warning("Each period needs at least one usable change. Adjust the cutoffs or ownership map.")
    elif st.button("Accept dates and continue", type="primary"):
        st.session_state["setup_cutoffs"] = cutoffs
        stage("training")
        st.rerun()


def training(scan):
    st.header("Train ownership models")
    st.write("Each agent learns from words, symbols and paths in its own pre-fix source. "
             "Prepared evidence is saved after each issue so interrupted extraction can resume.")
    groups, cutoffs = st.session_state["setup_groups"], st.session_state["setup_cutoffs"]
    st.caption(f"{len(groups)} agents · validation starts {cutoffs[0]} · evaluation starts {cutoffs[1]}")
    if st.button("Train models", type="primary"):
        with st.status("Preparing repository-local predictors", expanded=True) as status:
            message = st.empty()
            try:
                path = train(scan, groups, cutoffs, cache=CACHE, progress=message.write)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                status.update(label="Training incomplete", state="error")
                st.error(f"{error}. Completed issue evidence is available for a retry.")
                return
            status.update(label="Training complete", state="complete")
        st.session_state["setup_result"] = str(path)
        stage("results")
        st.rerun()


def results():
    path = Path(st.session_state["setup_result"])
    preset = load_prepared(path)
    st.header("Ownership model results")
    counts = preset.manifest["split_counts"]
    st.write(f"Learned from {counts['train']} changes, calibrated on {counts['validation']}, "
             f"and evaluated on {counts['test']} later changes.")
    st.write("Brier loss is the average squared error of the ownership probability. "
             "Calibration error compares predicted and observed frequencies in ten probability bins. Lower is better for both.")
    st.dataframe(preset.manifest["model_metrics"], hide_index=True, width="stretch")
    st.caption(f"{len(preset.issues)} held-out changes · {sum(preset.outcomes.sum(axis=1) > 1)} require multiple agents · "
               f"run {path.parent.name[:12]}")
    if st.button("Open reporting game", type="primary"):
        open_replay(path)


def main():
    current = st.session_state.get("setup_stage", "repository")
    steps = ["repository", "suitability", "ownership", "timeline", "training", "results"]
    if "setup_info" in st.session_state:
        steps.insert(1, "roots")
    st.caption(f"New repository experiment · {steps.index(current) + 1} of {len(steps)}")
    if current != "repository":
        st.button("← Back", on_click=stage, args=(steps[steps.index(current) - 1],))
    if current == "repository":
        repository()
    elif current == "roots":
        roots_page()
    elif current == "results":
        results()
    else:
        scan = st.session_state["setup_scan"]
        {"suitability": suitability, "ownership": mapping, "timeline": timeline, "training": training}[current](scan)


try:
    main()
except Interrupted as error:
    st.warning(f"Scan or download interrupted. Retry the current action to resume. {error}")
except (StorageLimit, OSError, ValueError, subprocess.SubprocessError) as error:
    st.error(str(error))
