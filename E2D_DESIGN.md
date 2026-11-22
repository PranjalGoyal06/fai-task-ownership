# E2-D: Direct verbalized-confidence experiment

## Status and research question

E2-D tests whether an LLM's direct verbalized confidence can rank and estimate
which Django module a historical issue's production fix changed. The LLM supplies
the probability itself; there is no fitted probability head or fine-tuning.
Mechanism policies are fitted separately on validation outcomes and evaluated on
held-out test issues.

E2-D is distinct from [E2_HANDOFF.md](./E2_HANDOFF.md). That handoff specifies an
LLM semantic-evidence extractor followed by a calibrated predictor, paired against
lexical features. E2-D is a direct-belief experiment and does not establish that
semantic evidence improves a learned predictor.

## Experimental mapping

| Model object | E2-D observation |
| --- | --- |
| Task | A frozen Django issue, its text, and the pre-fix base revision |
| Agent | One module-specific probability request |
| Local information | Issue text, module name, and the E1 training-only module base rate; optionally, bounded source snippets in the separate `snippets` mode |
| Belief estimate | The JSON probability that the issue's production fix touches that module |
| Outcome | E1's frozen production-module label, revealed only for evaluation and mechanism settlement |
| Public graph and assignment mechanism | The graph, costs, reporting game, and conjunctive assignment from E1 |

The `no_tools` prompt provides no source code; it is an issue-only control. The
`snippets` mode supplies bounded pre-fix snippets from only the queried module.
Citation validation checks exact source text, module scope, and (in snippets mode)
the retrieved range. It does **not** determine whether the natural-language claim
is entailed by or relevant to the cited text. Therefore the output establishes
source-matched citations, not independently verified semantic evidence. Neither
mode provides the gold patch, later source, or another module's source.

## Prediction and data protocol

1. Reuse E1's frozen issues, labels, module map, graph, and chronological splits.
   The cohort is ordered chronologically; `--max-issues 50` takes the first 50
   issues independently from each selected split. This is a deterministic pilot
   subset, not a random or necessarily representative sample.
2. Use the unsmoothed module ownership rates from E1's training issues as the
   public prior for both validation and test prompts. This matches E1's reference
   prior and avoids using validation outcomes to form validation predictions.
3. Make one call per selected issue/module. The current local configuration is
   Ollama with `qwen3:4b`, temperature zero, and an unknown documented cutoff.
   The provider, model tag, endpoint, cutoff, prompt signature, and per-call raw
   response are cached. Successful calls can be reused after interruption.
4. Require a finite probability in `[0, 1]`. Keep the raw response and parsed
   evidence in the per-call cache. Record failures rather than silently replacing
   missing probabilities. Only complete nine-module matrices enter mechanism
   replay; individual successful predictions remain available for calibration
   diagnostics.
5. Replay the cached matrix without making more model calls. The default CLI
   command selects up to 50 validation and 50 test issues, or 900 module calls
   per mode.
6. To compare issue-only and source-grounded prompts, run both modes into separate
   output directories with identical split limits, replay both, and use
   `--compare-mode` to calculate paired differences on their shared complete issue
   matrices.

The current model's documented cutoff is unknown and must remain recorded as
unknown unless an authoritative source establishes it. SWE-bench contamination
cannot be ruled out by local inference or citation validation.

## Evaluation

### Belief quality

Measure Brier loss, ten-bin ECE, and AUC against revealed module labels, overall
and per module. These are diagnostics of the raw verbalized probabilities; E2-D
does not fit a calibration map. Small pilot counts and modules with one outcome
class cannot support stable AUC or calibration conclusions. Do not treat
validation diagnostics as held-out results.

### Assignment and incentives

The current replay fits E1's finite reporting policy separately to direct LLM
forecasts and E1's lexical forecasts on the exact same complete validation
issues. It evaluates both fitted policies on the exact same complete test issues,
using matched private-type and audit random seeds. It includes honest routing,
strategic B2 (credit, score, and audit off), the aligned controls, acquisition
types at `phi = 1/3` and `2/3`, score scales 0/2/8, polling, scoring-only,
audit-only (`lambda = 0, alpha = 0.25`), combined scoring and audits, and the
configured equal-split/Shapley credit cases. The misalignment, scoring, and audit
sweeps reuse cached beliefs.

The replay reports per-issue welfare, coverage, padding, incentive score, audit
spend, fines, and a miss-loss/excess-cost decomposition. Paired summaries report
direct-minus-lexical Brier loss, welfare, coverage, and padding on matched issues.
For acquisition settings, gross recovery is welfare above strategic B2; net
recovery subtracts centered scoring and audit expenditures, matching E1's budget
accounting. The aggregate regret-like decomposition uses E1's lexical B4 estimate.
Neither it nor efficiency is an oracle built from direct LLM evidence; identify it
as the E1 lexical reference.

### Baselines and limitations

- **B1:** E1 module-description router.
- **B2-direct:** honest equation-2 routing from direct probabilities. The separately
  fitted credit-off, score-off, audit-off acquisition policy is labeled strategic
  B2.
- **B3:** plurality over module-level choices derived from the direct
  probabilities.
- **B4-E1-lexical:** E1's pooled lexical estimate, retained as a comparator and
  reference only; it is not the full-information oracle for E2-D.
- **Central:** E1's issue-text-only comparator.

An estimated full-information oracle based on all modules' cached evidence would
require an additional model stage and is not implemented. Calibration distortion
maps such as `g_i`, calibration fitting, and repeated generations for variance
are outside the current E2-D implementation. Do not claim those analyses from its
outputs.

For local inference, API usage cost is zero; Ollama token counts are recorded when
the local endpoint returns usage metadata. Inference time is recorded separately.
These operational measurements do not measure the model's training cost or
electricity use.

## Readiness criteria before reporting findings

- Complete validation and test matrices are available for the declared cohort.
- The versioned manifest identifies the exact issue IDs, provider/model, and
  settings; the failure file records call errors. No incomplete issue is silently
  treated as a full matrix.
- Test Brier/ECE/AUC and class counts are reported before mechanism outcomes.
- Mechanism results identify the E1 lexical B4 reference and the subset used to
  fit policy.
- Contamination, chronological-subset selection, label limitations, and the
  absence of a direct-evidence oracle remain explicit in the report.
