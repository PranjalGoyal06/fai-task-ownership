"""Inspect and compare policies on frozen historical replay inputs."""
from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.lab.analysis import compare
from src.lab_ui.graph import graph
from src.lab.preset import load_prepared, load_preset
from src.lab.replay import Settings, export, read_recipe, recipe, replay
from src.e2.research_grid import (
    sweep_rq1,
    sweep_rq2,
    sweep_rq3,
    generate_methods_benchmark,
    export_full_research_suite,
)


@st.cache_data(max_entries=16, show_spinner=False)
def run_cached(digest, settings, _preset):
    return replay(_preset, settings)


def configure(preset):
    count_agents = len(preset.manifest["modules"])
    active = st.session_state.get("historical_run", Settings(strategic_agents=round(count_agents * 2 / 3)))
    with st.form("historical_parameters"):
        scale = st.number_input("Scoring strength", min_value=0., value=float(active.scoring), step=.5,
                                help="Higher values give agents a stronger reward for accurate probabilities.")
        with st.expander("Advanced settings"):
            count = st.slider("Agents with a private incentive", 0, count_agents, active.strategic_agents)
            cost = st.number_input("Private cost of assignment", value=float(active.private_cost), step=.25,
                                   help="A negative cost rewards the agent for being assigned.")
            audit = st.slider("Chance of an audit", 0., 1., float(active.audit), step=.05)
            rules = ["off", "equal", "shapley"]
            credit = st.selectbox("Resolution credit", rules, index=rules.index(active.credit),
                                  format_func={"off": "Off", "equal": "Equal split", "shapley": "Shapley"}.get)
            reward = st.select_slider("Reward for a covered fix", (5, 8, 12), value=active.reward)
            detection = st.slider("Detection chance per neighbor", 0., 1., float(active.detection), step=.05)
            fine = st.number_input("Fine for a contradicted payload", min_value=0., value=float(active.fine), step=.5)
            seed = st.number_input("Random seed", 0, 2**32 - 1, active.seed)
        if st.form_submit_button("Run replay", type="primary"):
            active = Settings(private_cost=cost, strategic_agents=count, scoring=scale, audit=audit,
                              credit=credit, seed=seed, reward=reward, detection=detection, fine=fine)
            st.session_state["historical_run"] = active
    with st.expander("Restore a replay recipe"):
        upload = st.file_uploader("Replay recipe", type="json")
        if st.button("Apply recipe", disabled=upload is None):
            try:
                st.session_state["historical_run"] = read_recipe(upload.getvalue(), preset)
            except (ValueError, TypeError, OverflowError) as error:
                st.error(f"Could not restore settings. {error}")
            else:
                st.rerun()
    return active


def diagnostics(run, name):
    d = run.diagnostic
    message = (f"{name} solver · {d['status']} after {d['iterations']} passes. "
               f"Largest gain available by changing one reporting decision was {d['residual']:.3g}.")
    if d["status"] == "converged":
        st.caption(message)
    else:
        st.warning(message + " Treat these numbers as the final iteration of an unfinished search.")


def comparison_view(run, polling, ids):
    comparison = compare(run.issues.loc[ids], polling.issues)
    table = comparison.totals
    st.dataframe(table.style.format("{:.3f}", subset=pd.IndexSlice[table.index[-2:], :])
                 .format("{:.0f}", subset=pd.IndexSlice[table.index[:3], :]), width="stretch")
    newly, lost = comparison.newly_covered, comparison.newly_missed
    gain = table.at["Welfare per issue", "Change"]
    low, high = comparison.welfare_gain_interval
    st.write(f"Of these **{len(ids)} issues**, the selected settings cover **{newly}** fixes that "
             f"polling missed and miss **{lost}** that polling covered. Each covered fix earns {run.settings.reward} units. "
             "Each assignment costs 1 unit, including assignments to modules outside the fix.")
    st.caption(f"Mean welfare change {gain:+.2f} units per issue. "
               f"95% paired bootstrap interval [{low:+.2f}, {high:+.2f}]. "
               "The interval resamples issues with the fitted policies and seed held fixed. "
               "Incentive cost counts centered score payments and expected audit spending separately from welfare.")


def evidence(snapshot, frame, modules, repository):
    module = st.selectbox("Inspect local evidence for module", modules)
    idx = modules.index(module)
    if direct := snapshot.get("direct_evidence", {}).get(str(idx)):
        st.write(f"The local model estimated **{frame.at[module, 'belief']:.1%}** ownership from this issue and "
                 "the supplied pre-fix source excerpts.")
        if direct["evidence"]:
            st.write(" ".join(direct["evidence"]))
        if direct["citations"]:
            st.caption("Verified excerpts supplied to the model")
            for citation in direct["citations"]:
                url = f"https://github.com/{repository}/blob/{snapshot['base']}/{citation['path']}#L{citation['start_line']}"
                st.markdown(f"[{citation['path']} lines {citation['start_line']}–{citation['end_line']}]({url})")
                st.code(citation["snippet"], language="python")
        else:
            st.caption("The model returned no verified source citation for this owner.")
        return
    overlap, symbols, paths, mentioned = snapshot["features"][idx]
    st.write(f"The local model estimated **{frame.at[module, 'belief']:.1%}** ownership. "
             "It used four measurements from this module at the pre-fix revision.")
    st.table(pd.DataFrame({
        "Local signal": ["Highest normalized word overlap", "Most matching symbols in one file",
                         "Files named by path or dotted name", "Module prefix mentioned"],
        "Value": [f"{overlap:.3f}", str(int(symbols)), str(int(paths)), "Yes" if mentioned else "No"],
    }).set_index("Local signal"))
    citations = snapshot["citations"].get(str(idx), [])
    if citations:
        st.caption("Up to three matching files from the extractor. A matching name shows lexical overlap.")
        for path, names, lines in citations:
            url = f"https://github.com/{repository}/blob/{snapshot['base']}/{path}"
            if lines:
                url += f"#L{lines[0]}"
            st.markdown(f"[{path}]({url})" + (" · " + ", ".join(names) if names else ""))
    else:
        st.caption("The extractor recorded no matching path or symbol citation for this module.")


def switch_view(view):
    st.session_state["historical_view"] = view


def open_inspector(settings, subset):
    st.session_state["inspection_settings"] = settings
    st.session_state["inspection_subset"] = subset
    switch_view("inspect")


def paired_runs(preset, settings):
    return (run_cached(preset.digest, settings, preset),
            run_cached(preset.digest, settings.polling(), preset))


def solver_warnings(run, polling):
    for name, result in [("Selected policy", run), ("Polling", polling)]:
        if result.diagnostic["status"] != "converged":
            diagnostics(result, name)


def issue_ids(run, subset):
    sizes = run.issues.k
    if subset == "Multi-module":
        return sizes.index[sizes > 1]
    if subset == "Single-module":
        return sizes.index[sizes == 1]
    return sizes.index


def overview(preset):
    run, polling = paired_runs(preset, Settings())
    st.caption("Django · Published historical replay")
    st.title("When a module owner benefits from being assigned, can rewarding accurate reports reduce unnecessary assignments?")
    solver_warnings(run, polling)
    story, diagram = st.columns([1, 1.15], gap="large")
    with story:
        st.write("Six of nine module owners get a private benefit from being picked. "
                 "Polling asks each owner how likely it is to be needed. Scoring also rewards "
                 "an accurate probability after the fix is revealed.")
        multi = run.issues.k > 1
        st.metric(f"Assignments across {len(run.issues)} fixes · polling → scoring",
                  f"{polling.issues.assignments.sum()} → {run.issues.assignments.sum()}")
        st.write(f"Scoring covered **{run.issues.coverage.sum()} fixes**, up from **{polling.issues.coverage.sum()}**. "
                 "A fix is covered when every module changed by its patch is assigned.")
        st.write(f"Among the **{multi.sum()} multi-module fixes**, coverage fell from "
                 f"**{polling.issues.loc[multi, 'coverage'].sum()} to {run.issues.loc[multi, 'coverage'].sum()}**.")
        with st.container(horizontal=True):
            st.button("Inspect a fix", type="primary", on_click=open_inspector, args=(Settings(), "All fixes"))
            st.button("Change the experiment", on_click=switch_view, args=("experiment",))
    with diagram:
        graph(preset.snapshots[preset.issues.index[0]], preset.manifest["modules"], preset.manifest["repository"])


def inspector(preset):
    settings = st.session_state["inspection_settings"]
    subset = st.session_state["inspection_subset"]
    run, polling = paired_runs(preset, settings)
    ids = issue_ids(run, subset)
    selected, base = run.issues.loc[ids], polling.issues.loc[ids]
    st.title("Inspect a fix")
    repository = preset.manifest["repository"]
    st.caption(f"{repository} · {subset} · Scoring strength {settings.scoring:g}")
    solver_warnings(run, polling)
    gained = selected.coverage > base.coverage
    lost = selected.coverage < base.coverage
    different = selected.assigned != base.assigned
    groups = {"Newly covered": ids[gained], "Newly missed": ids[lost],
              "Different assignments": ids[different], "All fixes": ids}
    options = [label for label, matches in groups.items() if len(matches)]
    group = st.radio("Show fixes", options, horizontal=True)
    key = st.selectbox("Historical fix", list(groups[group]),
                       format_func=lambda k: f"#{k.rsplit('-', 1)[1]} · {preset.issues.at[k, 'title']}")
    before, after = polling.agents.loc[key], run.agents.loc[key]
    title = preset.issues.at[key, "title"]
    st.subheader(title)
    pr = key.rsplit("-", 1)[1]
    links = f"[Resolving pull request](https://github.com/{repository}/pull/{pr})"
    if "issue" in preset.issues:
        number = preset.issues.at[key, "issue"]
        links += f" · [Original issue](https://github.com/{repository}/issues/{number})"
    st.caption(links)
    outcomes = ["missed", "covered"]
    st.write(f"Polling **{outcomes[int(base.at[key, 'coverage'])]}** this fix with "
             f"{int(base.at[key, 'assignments'])} assignments. The selected policy "
             f"**{outcomes[int(selected.at[key, 'coverage'])]}** it with {int(selected.at[key, 'assignments'])}.")
    st.write("Each agent starts with a local estimate. Its incentives can change the probability it reports. "
             "The last column shows which modules the eventual patch changed.")
    ledger = pd.DataFrame({"Local estimate": after.belief, "Polling report": before.report,
                           "Selected report": after.report, "Polling assigned": before.assigned,
                           "Selected assigned": after.assigned, "Patch owner": after.patch_owner})
    st.dataframe(ledger.style.format({name: "{:.1%}" for name in ledger.columns[:3]}), width="stretch")
    if "roots" in preset.manifest:
        st.caption("The graph shows imports resolved within eligible Python source at this pre-fix revision.")
    graph(preset.snapshots[key], preset.manifest["modules"], repository, after)
    with st.expander("Evidence, audits and payouts"):
        evidence(preset.snapshots[key], after, preset.manifest["modules"], repository)
        st.dataframe(after[["private_cost", "altered_payload", "audited", "contradicted",
                            "selection_input", "score", "credit"]], width="stretch")
    st.button("Change the experiment", on_click=switch_view, args=("experiment",))


def experiment(preset):
    st.title("Change the experiment")
    st.caption(preset.manifest["repository"])
    st.write("Try changing the reward for an accurate report. Each run compares your settings with "
             "polling on the same fixes, with the same private incentives.")
    settings = configure(preset)
    run, polling = paired_runs(preset, settings)
    solver_warnings(run, polling)
    st.subheader("Replay results")
    if "roots" in preset.manifest:
        st.caption("Results cover eligible production Python fixes in the selected roots.")
    st.caption(f"Applied scoring {settings.scoring:g} · {settings.strategic_agents} agents with a private incentive")
    slices = [label for label in ("All fixes", "Single-module", "Multi-module") if len(issue_ids(run, label))]
    slice_name = st.radio("Compare fixes", slices, horizontal=True)
    comparison_view(run, polling, issue_ids(run, slice_name))
    st.button("Inspect a fix", on_click=open_inspector, args=(settings, slice_name))
    with st.expander("Applied settings and solver details"):
        st.json(asdict(settings))
        for name, result in [("Selected policy", run), ("Polling", polling)]:
            if result.diagnostic["status"] == "converged":
                diagnostics(result, name)
    if not preset.methods.empty:
        with st.expander("Published benchmarks"):
            st.write("Published results at reward 8 on this slice. The pooled benchmark estimates likely owner sets from earlier "
                     "fixes and pooled pre-fix evidence. A fix counts as covered when every changed module is assigned.")
            labels = {"B4": "Pooled benchmark", "central": "Issue-text classifier", "B1": "Description router", "B3": "Voting"}
            subset = {"All fixes": "all", "Single-module": "single", "Multi-module": "multi"}[slice_name]
            rows = preset.methods[preset.methods.config.isin(labels) & (preset.methods["slice"] == subset)]
            st.dataframe(rows.assign(Method=rows.config.map(labels)).set_index("Method")
                         [["n", "coverage", "padding", "welfare"]], width="stretch")
    with st.expander("Reproduce this run"):
        st.write("The export includes the applied settings, input digest, solver status, issue totals and "
                 "agent reports for both policies across the full held-out set.")
        st.json(asdict(settings))
        st.download_button("Download settings", data=json.dumps(recipe(preset, settings), indent=2),
                           file_name="recipe.json", mime="application/json")
        st.download_button("Download replay ZIP", data=export(preset, run, polling),
                           file_name=preset.manifest["repository"].split("/")[-1] + "-replay.zip", mime="application/zip")

    with st.expander("📊 Paper Research Questions (Grid Replay & Tables)", expanded=True):
        st.markdown(
            "Run the full parameter sweeps across **RQ1** (misalignment scaling), **RQ2** (scoring & audit recovery), "
            "and **RQ3** (coalition credit sharing) on these held-out issues to reproduce the results for the paper."
        )
        is_direct = preset.manifest.get("probability_source") == "direct_model"
        reports_dir = Path("reports/e2" if is_direct else "reports/e1")
        if st.button("🚀 Run Full Research Grid Sweep", type="secondary"):
            with st.spinner("Executing grid sweeps across all game-theoretic conditions..."):
                export_full_research_suite(preset, output_dir=reports_dir)
                st.session_state["research_grid_done"] = True
                st.success(f"Grid sweep complete! Artifacts exported to `{reports_dir}/`.")

        out_dir = reports_dir
        if (out_dir / "methods_benchmark.csv").exists():
            t1, t2, t3, t4 = st.tabs(["Methods Benchmark (Table 4)", "RQ1: Misalignment", "RQ2: Scoring & Audit", "RQ3: Coalition Credit"])
            with t1:
                st.subheader(f"Benchmark Comparison (Table 4) · {'Direct LLM (E2)' if is_direct else 'Trained Classifier (E1)'}")
                df_meth = pd.read_csv(out_dir / "methods_benchmark.csv")
                st.dataframe(df_meth, hide_index=True, width="stretch")
            with t2:
                st.subheader("RQ1: Scaling with Misalignment (ell, phi)")
                df_rq1 = pd.read_csv(out_dir / "rq1_misalignment.csv")
                st.dataframe(df_rq1[["regime", "private_cost_ell", "strategic_fraction_phi", "welfare", "miss_loss", "excess_cost", "padding", "coverage"]], hide_index=True, width="stretch")
            with t3:
                st.subheader("RQ2: Scoring & Audit Recovery (lambda, alpha)")
                df_rq2 = pd.read_csv(out_dir / "rq2_scoring_audit.csv")
                st.dataframe(df_rq2[["regime", "scoring_lambda", "audit_rate_alpha", "welfare", "net_welfare", "padding", "score_overhead", "net_overhead", "audited_count", "contradicted_count"]], hide_index=True, width="stretch")
            with t4:
                st.subheader("RQ3: Coalition Credit (Equal Split vs. Shapley Value)")
                df_rq3 = pd.read_csv(out_dir / "rq3_credit_coalitions.csv")
                st.dataframe(df_rq3[["credit_rule", "reward_R", "scoring_lambda", "welfare", "coverage", "padding", "iterations", "converged"]], hide_index=True, width="stretch")


def main():
    try:
        prepared = st.session_state.get("active_prepared_path")
        preset = load_prepared(Path(prepared)) if prepared else load_preset()
    except (ValueError, OSError, KeyError) as error:
        st.error(f"Could not load the replay. Check the preset and report files. {error}")
        return
    if prepared and st.button("Open published Django replay"):
        for key in ("active_prepared_path", "historical_run", "historical_view", "inspection_settings", "inspection_subset"):
            st.session_state.pop(key, None)
        st.rerun()
    view = st.session_state.get("historical_view", "experiment" if prepared else "overview")
    if view == "overview":
        overview(preset)
        return
    if not prepared:
        st.button("← Published replay", on_click=switch_view, args=("overview",))
    if view == "inspect":
        inspector(preset)
    else:
        experiment(preset)


main()
