"""Streamlit presentation for the frozen Django experiment."""
from dataclasses import asdict

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.e1.dataset import MODULES
from src.lab.analysis import compare
from src.lab.preset import load_preset
from src.lab.replay import Settings, export, read_recipe, recipe, replay


@st.cache_data(max_entries=16, show_spinner=False)
def run_cached(digest, settings, _preset):
    return replay(_preset, settings)


def configure(preset):
    active = st.session_state.get("historical_run", Settings())
    with st.sidebar.form("historical_parameters"):
        st.subheader("Reporting game")
        count = st.slider("Agents with a private incentive", 0, 9, active.strategic_agents,
                          help="Each issue has nine module owners. The seed picks which owners get this cost.")
        cost = st.number_input("Private cost of assignment", value=float(active.private_cost), step=.25,
                               help="A negative cost rewards the agent for being assigned.")
        scale = st.number_input("Scoring strength", min_value=0., value=float(active.scoring), step=.5,
                                help="How strongly the outcome rewards an accurate probability.")
        audit = st.slider("Chance of an audit", 0., 1., float(active.audit), step=.05)
        rules = ["off", "equal", "shapley"]
        credit = st.selectbox("Resolution credit", rules, index=rules.index(active.credit),
                              format_func={"off": "Off", "equal": "Equal split", "shapley": "Shapley"}.get)
        with st.expander("Reward and audit model"):
            reward = st.select_slider("Reward for a covered fix", (5, 8, 12), value=active.reward)
            detection = st.slider("Detection chance per neighbor", 0., 1., float(active.detection), step=.05)
            fine = st.number_input("Fine for a contradicted payload", min_value=0., value=float(active.fine), step=.5)
        seed = st.number_input("Random seed", 0, 2**32 - 1, active.seed)
        if st.form_submit_button("Run replay", type="primary"):
            active = Settings(private_cost=cost, strategic_agents=count, scoring=scale, audit=audit,
                              credit=credit, seed=seed, reward=reward, detection=detection, fine=fine)
            st.session_state["historical_run"] = active
    with st.sidebar.expander("Restore saved settings"):
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


def belief_plot(frame):
    chart = go.Figure()
    for name, row in frame.iterrows():
        chart.add_scatter(x=[row.belief, row.report], y=[name, name], mode="lines",
                          line=dict(color="#AEC1CA", width=3), showlegend=False, hoverinfo="skip")
    for name, column, color, symbol in [("Local estimate", "belief", "#146D75", "circle"),
                                        ("Submitted report", "report", "#B65430", "diamond-open")]:
        chart.add_scatter(name=name, x=frame[column], y=frame.index, mode="markers",
                          marker=dict(color=color, size=12, symbol=symbol, line=dict(width=2)),
                          hovertemplate="%{y} · %{x:.3f}<extra>%{fullData.name}</extra>")
    chart.update_layout(xaxis=dict(range=[-.04, 1.04], tickformat=".0%", title="Probability of ownership"),
                        yaxis=dict(autorange="reversed"), height=365,
                        margin=dict(l=0, r=16, t=0, b=0), legend=dict(orientation="h"))
    st.plotly_chart(chart, width="stretch")


def graph(snapshot):
    focus = st.selectbox("Inspect imports for module", MODULES)
    i = MODULES.index(focus)
    edges = [edge for edge in snapshot["edges"] if i in edge[:2]]
    nodes = {i} | {n for edge in edges for n in edge[:2]}
    dot = ['digraph imports {', 'rankdir=LR;', 'node [shape=box style="rounded,filled"];']
    for node in sorted(nodes):
        color = "#AFDAD8" if node == i else "#E7EEF2"
        dot.append(f'n{node} [label="{MODULES[node]}" fillcolor="{color}" fontcolor="#173641"];')
    rows = []
    for source, target, path, line in edges:
        dot.append(f"n{source} -> n{target};")
        rows.append({"Importing module": MODULES[source], "Imports module": MODULES[target],
                     "Pre-fix source": f"https://github.com/django/django/blob/{snapshot['base']}/{path}#L{line}"})
    st.caption("An arrow points from the importing module to its dependency. Each row links one "
               "source line for that direction. A module connected in both directions counts as one audit neighbor.")
    st.graphviz_chart("\n".join([*dot, "}"]))
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Pre-fix source": st.column_config.LinkColumn(display_text="View import")})


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


def inspect_issue(preset, run, polling, ids):
    key = st.selectbox("Historical fix", list(ids),
                       format_func=lambda k: f"#{k.rsplit('-', 1)[1]} · {preset.issues.at[k, 'title']}")
    snapshot = preset.snapshots[key]
    frame = run.agents.loc[key]
    url = "https://github.com/django/django/pull/" + key.rsplit("-", 1)[1]
    st.caption(f"Pre-fix revision {snapshot['base'][:12]} · [Resolving pull request]({url})")
    step = st.radio("Issue inspector", ["Evidence", "Reports", "Assignment", "Settlement", "Import graph"],
                    index=1, horizontal=True)
    if step == "Evidence":
        evidence(snapshot, frame)
    elif step == "Reports":
        st.write("Each dot is a module’s estimate from pre-fix evidence. The diamond is the "
                 "probability its agent submitted after accounting for incentives.")
        belief_plot(frame)
        st.dataframe(frame[["private_cost", "belief", "report", "altered_payload", "audited",
                            "contradicted", "selection_input"]], width="stretch")
    elif step == "Assignment":
        assigned = frame.index[frame.assigned].tolist()
        st.write("**Selected modules**", ", ".join(assigned) or "Empty set")
        st.caption("The mechanism ranks the probabilities available after audit checks and chooses "
                   "the set with the highest expected reward minus assignment costs.")
        st.dataframe(pd.DataFrame({"Selection probability": frame.selection_input,
                                   "Assigned by polling": polling.agents.loc[key].assigned,
                                   "Assigned by selected settings": frame.assigned}), width="stretch")
    elif step == "Settlement":
        st.write("**Modules changed in the historical fix**", ", ".join(frame.index[frame.patch_owner]))
        row = run.issues.loc[key]
        st.write(f"{'All required modules covered' if row.coverage else 'Fix missed'} · "
                 f"{row.assignments:.0f} assignments · {row.padding:.0f} extra · {row.welfare:g} units of welfare.")
        st.dataframe(frame[["assigned", "patch_owner", "score", "credit"]], width="stretch")
    else:
        graph(snapshot)


def main():
    st.title("Task ownership lab")
    st.subheader("Django historical replay")
    try:
        preset = load_preset()
    except (ValueError, OSError, KeyError) as error:
        st.error(f"Could not load the Django preset. Check the preset and report files. {error}")
        return
    counts = preset.manifest["split_counts"]
    st.write(f"Replay {counts['test']} fixes with nine module owners. "
             f"The ownership models learned from {counts['train']} earlier fixes. "
             f"Another {counts['validation']} supplied calibration data and examples for fitting reports.")
    st.caption("Ownership refers to modules changed in the historical patch. Polling uses the same "
               "private incentives and random draws, with scoring, audits and credit switched off.")
    settings = configure(preset)
    with st.spinner("Solving the reporting game and its paired polling control…"):
        run = run_cached(preset.digest, settings, preset)
        polling = run_cached(preset.digest, settings.polling(), preset)
    st.caption(f"Applied settings · {settings.strategic_agents} agents at private cost {settings.private_cost:g} · "
               f"scoring {settings.scoring:g} · audit {settings.audit:.0%} · detection {settings.detection:.0%} · "
               f"fine {settings.fine:g} · reward {settings.reward} · credit {settings.credit} · seed {settings.seed}")
    slice_name = st.radio("Compare the same issues under both policies", ["All fixes", "Single-module", "Multi-module"],
                          horizontal=True)
    sizes = run.issues.k
    ids = sizes.index
    if slice_name == "Single-module":
        ids = sizes.index[sizes == 1]
    elif slice_name == "Multi-module":
        ids = sizes.index[sizes > 1]
    comparison_view(run, polling, ids)
    diagnostics(run, "Selected policy")
    diagnostics(polling, "Polling")
    st.divider()
    inspect_issue(preset, run, polling, ids)
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


main()
