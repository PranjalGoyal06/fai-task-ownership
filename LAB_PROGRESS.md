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

## scikit-learn pilot checkpoint · 21 November 2026

The pilot is ready for local review on `feat/task-ownership-lab` at base `8e6d17c`. All milestone changes are unstaged and uncommitted. The setup page, headless intake and training commands, replay exports and recovery instructions are described in the [lab guide](README.md#run-the-scikit-learn-pilot).

| Intake or training stage | Count |
| --- | ---: |
| Pull requests merged in 2022 | 1,734 |
| One same-repository closing issue | 320 |
| Recoverable pre-fix parent | 320 |
| Verified patch with a production Python change | 217 |
| Usable after text and timing checks | 207 |
| Training / validation / held out | 124 / 41 / 42 |
| Held-out fixes requiring multiple owners | 8 |

The scan excluded 1,401 PRs without a closing link, 13 with ambiguous links, one whose path list could not be verified, 102 with no eligible Python change, and ten with timing or issue-text availability problems. Two merge objects required fetching by exact revision. Fourteen retained fixes also touched compiled sources. The separately cached SWE-bench reference contained 229 scikit-learn cases and shared no PR IDs with this scan.

The suggested map has seven owners. Validation starts on July 25 and evaluation starts on October 11, 2022 (UTC). The agents had 9, 18, 13, 6, 16, 20 and 55 positive training examples in map order. Each had both label classes in validation and received sigmoid calibration. The counts per owner can sum to more than the number of issues because a patch can touch several owners.

| Default replay, seed 331, five strategic agents | Polling | Scoring at strength 8 |
| --- | ---: | ---: |
| Owner assignments over 42 fixes | 239 | 188 |
| Fixes with every required owner assigned | 37 | 29 |
| Multi-owner fixes fully covered, out of 8 | 4 | 3 |

Both reporting policies converged. Scoring saved 51 assignments and covered eight fewer fixes on this particular cohort. Only eight held-out fixes required multiple owners, and the coverage measure observes Python changes only.

**Reproduction checks**

- Two independent clean-cache runs produced identical prepared inputs with SHA-256 `13140428aec0cd90b6a1e3852a6c02d563d68bfe0e82206b885debf558249a48`. The recorded repository revision is `f85f0584d5d7a66068bfe50b395ac855b2289dc2`.
- Replaying the default ZIP using only its settings and frozen inputs reproduced every byte, including per-issue assignments, per-agent reports and solver diagnostics. The local export is `results/sklearn-2022/checks/default.zip`.
- Merged *Utilities* and *Other production code*, held both dates fixed and kept four strategic agents on each map for this separate check. The same 42 issues remained. Three needed fewer owners, the directed edge count changed on all 42, and the merged agent’s local features differed from the former catch-all agent’s on 25. Both trained maps reproduced their exported ZIPs. Editing scoring alone reused the identical prepared probabilities.
- All 111 tests passed, including interrupted monthly intake, late and duplicate labels, unmapped paths, sparse predictors, variable agent counts and all 44 original Django replay conditions. A browser walkthrough reached the reporting game from setup, including the graph, dates, training table and issue inspector.
- No outstanding blocker. The local GitHub history, month batches, evidence, prepared runs and comparison exports are in ignored storage.

## Public Python repository checkpoint · 21 November 2026

Branch `feat/task-ownership-lab`, base `8e6d17c`. The pilot correction and generic workflow are unstaged and uncommitted.

- The pilot ownership editor now counts files per rule in the first snapshot and across the scan. A misspelled `sklearn/utlis` rule is blocked even when a catch-all owner exists. A path introduced later is accepted. Offline replay of the saved scikit-learn export still produces SHA-256 `796dd1423f86da2d8dc347f7f20fcfad81a2bf15a78ceec699f97dfcec317a16`.
- The generic workflow pins a public repository, checks the file trees and scan window, and presents a source-byte estimate before downloading. It verifies Git object digests and uses a shared disk-budget lock for metadata, source, evidence and prepared inputs. Cache cleanup retains trained runs.
- The browser acceptance run used `pallets/click` with its `src/click/` layout and the 2025 merge window. From an empty repository cache, the lab scanned 124 merged PRs, found 33 explicit same-repository links with recoverable parents, verified 20 production Python changes and retained 19 after the issue-text and chronology checks. One training label missed its cutoff. The final split was **10 training, 4 validation and 4 held-out changes**, including **0 multi-owner held-out changes**. This exercises the software flow. The cohort is too small for a conclusion about scoring.
- Preflight showed **2,562,940 bytes** across 59 distinct source versions. Two independently acquired caches produced identical prepared inputs (SHA-256 `4479f69c6814a9630be5e0be58860d40513bab4c700d65136cb37add758ab29d`). The verified repository cache occupied approximately 3.6 MiB, including metadata, evidence and the prepared run.
- In the inspector, Click PR **#3152** linked to issue **#3084**. GitHub’s patch list confirmed that `src/click/core.py` was the sole changed production file. The directed edge from `__init__` to `core` cited `src/click/__init__.py` line 10 at revision `7f7bbe4569ea68e8dabee232eade069ef3310aea`. The cited line imports `Argument` from `.core`.
- Exporting and replaying from only the ZIP’s inputs and recipe reproduced the complete Click ZIP byte-for-byte. The local artifact is `results/e1-acceptance/click-replay.zip` (SHA-256 `dc5a2549555183e45d34db3f91d88710f5dc6664c695cc911a48e455d4698a93`).
- All 119 tests passed, including all 44 Django parity cases. Regression coverage includes pre-download size rejection, concurrent writers, low free space, interrupted scans, invalid downloaded objects, late ownership paths, chronological support, unsupported projects, and preserved replay inputs after cache cleanup.

## Next steps

Review the generic setup page and saved results. E2 remains a separate evidence integration once its artifacts are available.

## Bulk Git history · 22 November 2026

Generic repository preparation now estimates compressed history using GitHub's repository storage figure and checks the remaining cache and disk budgets. The standard path acquires a bare clone. Larger histories use shallow pre-fix trees and batched Git blob transfers, with implicit lazy fetch disabled during availability checks. Existing per-file caches remain readable.

Verified against the interrupted 2025 LangChain scan. GitHub reported 644,590 KiB (approximately 629 MiB) for the repository. The selected Git cache occupied roughly 7.5 MiB in addition to the existing scan and files. All 7,844 required source versions for 107 candidate changes were available after preparation. A repeated preparation completed from local storage. The full suite passed 121 tests, including real local Git transfers for the full and filtered paths.
