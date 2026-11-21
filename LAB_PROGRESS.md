# Task ownership lab progress

Branch `feat/task-ownership-lab` starts at E1 revision `0d15b6a`. Its worktree is separate from the E1 and E2 development lines.

| Local commit time | Commit subject | Completed work |
| --- | --- | --- |
| 2026-11-19T23:48:30+05:30 | Merge remote-tracking branch 'origin/main' into feat/task-ownership-lab | Integrated the simulation and interpretability changes from main at `279cf29`. |
| 2026-11-19T23:57:33+0530 | feat: add reproducible historical replay runner | Prepared the portable Django inputs. Added policy replay, paired analysis, per-agent traces, recipe export and issue-level parity tests for all 44 E1 conditions. |
| 2026-11-19T23:58:07+0530 | feat: add historical inspector to the task ownership lab | Integrated the synthetic simulator, reporting controls, comparisons, evidence, import graph, settings restore and local downloads. |
| 2026-11-21T12:00:27+0530 | feat: guide the Django replay and show project-wide imports | Added the published comparison, paired fix inspector, progressive settings and interactive directed graph. |

## Guided replay checkpoint

- The app opens on the published result. Inspect a fix opens a coverage-changing example. Change the experiment exposes scoring first and retains recipe restore and export. Warnings follow unfinished solver runs into the inspector.
- The graph displays all nine owners. The 46 directed imports form 32 selectable connections, 14 with arrows at both ends. Issue-specific citations use the selected pre-fix revision.
- All 105 tests passed, including the 44 archived E1 conditions. Browser verification covered node highlighting with stable positions, both source links on a reciprocal connection, a newly missed fix, changing scoring, and navigation to the synthetic simulator.
- No open blocker at this review point. Next milestone is an end-to-end run on a second curated repository.

## Initial replay checkpoint

- Start the unified app with `python -m streamlit run lab.py`. Choose **Historical replay** or **Synthetic simulation** in the sidebar.
- The full suite passed 102 tests. Across all 44 archived conditions, the lab matched the original per-issue assignments, welfare, coverage, padding, loss decomposition, payments, audit and message counts on all 234 held-out issues. Solver status, iteration count and residual also matched.
- Default polling covered 110 fixes with 792 assignments. Scoring covered 114 with 610 assignments. Across the 27 multi-module fixes, coverage fell from nine to six. These are results of the original fitted Django experiment.
- Rebuilding the compressed supplement from the local E1 cache produced identical bytes. Downloaded-format settings reproduced identical ZIP bytes in a separate command-line run. The UI tests cover an edited run and settings restoration.
- A browser check opened both pages and rendered the directed pre-fix graph for the selected module. Local verification used the E1 reproduction environment.
- No outstanding blocker for this milestone. The preset fixes the nine Django owners and trains on this repository’s history. The lab recomputes strategic policies for each chosen configuration.

## Next steps

1. Review the Django replay and issue inspector against the published result.
2. Add an editable ownership map and a second curated repository. Regenerate ownership labels, local evidence, graph, training splits and probabilities for each map. Verify that a fresh export can reproduce the frozen Django map.
3. Add repository suitability checks before arbitrary repository intake. Record issue-to-fix linkage, eligible paths, ambiguous or unmapped patches, chronological split support, source size and import coverage.
4. Integrate the real E2 artifacts through a paired evidence comparison once available. Coordinate physical moves of shared model code after the E2 merge.
