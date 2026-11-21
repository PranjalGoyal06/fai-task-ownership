"""Streamlit presentation for the frozen Django experiment."""
from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.e1.dataset import MODULES
from src.lab.analysis import compare
from src.lab.imports import import_graph
from src.lab.preset import load_preset
from src.lab.replay import Settings, export, read_recipe, recipe, replay


@st.cache_data(max_entries=16, show_spinner=False)
def run_cached(digest, settings, _preset):
    return replay(_preset, settings)


def configure(preset):
    active = st.session_state.get("historical_run", Settings())
    with st.form("historical_parameters"):
        scale = st.number_input("Scoring strength", min_value=0., value=float(active.scoring), step=.5,
                                help="Higher values give agents a stronger reward for accurate probabilities.")
        with st.expander("Advanced settings"):
            count = st.slider("Agents with a private incentive", 0, 9, active.strategic_agents)
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


ASSETS = Path(__file__).parent
render_graph = st.components.v2.component(
    "ownership_import_graph",
    html='<div class="imports"></div>',
    css=(ASSETS / "import_graph.css").read_text(),
    js=(ASSETS / "import_graph.js").read_text(),
)


def graph(preset, key, frame=None):
    snapshot = preset.snapshots[key]
    assigned = [] if frame is None else frame.index[frame.assigned]
    owners = [] if frame is None else frame.index[frame.patch_owner]
    data = import_graph(preset.manifest["modules"], snapshot, preset.manifest["repository"], assigned, owners)
    st.subheader("Project import graph")
    st.caption("An arrow points to the imported module. A double arrow means both modules import each other.")
    if frame is not None:
        st.caption("Teal fill = assigned by the selected policy · diamond = changed in the patch.")
    render_graph(data=data, key="historical-import-graph")
    st.caption(f"Pre-fix revision `{snapshot['base'][:12]}` · {len(data['nodes'])} modules · "
               f"{len(snapshot['edges'])} directed imports")


def evidence(snapshot, frame):
    module = st.selectbox("Inspect local evidence for module", MODULES)
    idx = MODULES.index(module)
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
            url = f"https://github.com/django/django/blob/{snapshot['base']}/{path}"
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
        graph(preset, preset.issues.index[0])


def inspector(preset):
    settings = st.session_state["inspection_settings"]
    subset = st.session_state["inspection_subset"]
    run, polling = paired_runs(preset, settings)
    ids = issue_ids(run, subset)
    selected, base = run.issues.loc[ids], polling.issues.loc[ids]
    st.title("Inspect a fix")
    st.caption(f"Django · {subset} · Scoring strength {settings.scoring:g}")
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
    st.caption(f"[Resolving pull request](https://github.com/django/django/pull/{key.rsplit('-', 1)[1]})")
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
    graph(preset, key, after)
    with st.expander("Evidence, audits and payouts"):
        evidence(preset.snapshots[key], after)
        st.dataframe(after[["private_cost", "altered_payload", "audited", "contradicted",
                            "selection_input", "score", "credit"]], width="stretch")
    st.button("Change the experiment", on_click=switch_view, args=("experiment",))


def experiment(preset):
    st.title("Change the experiment")
    st.write("Try changing the reward for an accurate report. Each run compares your settings with "
             "polling on the same fixes, with the same private incentives.")
    settings = configure(preset)
    run, polling = paired_runs(preset, settings)
    solver_warnings(run, polling)
    st.subheader("Replay results")
    st.caption(f"Applied scoring {settings.scoring:g} · {settings.strategic_agents} agents with a private incentive")
    slice_name = st.radio("Compare fixes", ["All fixes", "Single-module", "Multi-module"], horizontal=True)
    comparison_view(run, polling, issue_ids(run, slice_name))
    st.button("Inspect a fix", on_click=open_inspector, args=(settings, slice_name))
    with st.expander("Applied settings and solver details"):
        st.json(asdict(settings))
        for name, result in [("Selected policy", run), ("Polling", polling)]:
            if result.diagnostic["status"] == "converged":
                diagnostics(result, name)
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
                           file_name="django-replay.zip", mime="application/zip")



def main():
    try:
        preset = load_preset()
    except (ValueError, OSError, KeyError) as error:
        st.error(f"Could not load the Django preset. Check the preset and report files. {error}")
        return
    view = st.session_state.get("historical_view", "overview")
    if view == "overview":
        overview(preset)
        return
    st.button("← Published replay", on_click=switch_view, args=("overview",))
    if view == "inspect":
        inspector(preset)
    else:
        experiment(preset)


main()
