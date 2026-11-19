# E2 handoff — semantic evidence for module ownership

Prepared on 19 November 2026 for the teammate implementing E2. This document specifies the experiment's meaning, mathematical contract, comparisons, and acceptance criteria. E2 is proposed work. The idea review defines the model, and the existing E1 report records the historical replay E2 extends. [Idea review](FAI_Project_Idea_Review.md) · [E1 report](reports/e1/REPORT.md)

**The LLM operates before strategic reporting.** It reads the issue and its module's pre-fix code, extracts relevant semantic evidence, and supplies that evidence to a calibrated predictor. The resulting probability estimates whether the historical fix touches that module. After the evidence and estimate are frozen, numerical policies choose reports and the mechanism performs audits, assignment, scoring, and credit settlement.

The research question is whether semantic evidence improves ownership prediction over lexical evidence, and whether those improvements increase welfare under the same incentive mechanisms. More informed probabilities and higher welfare are hypotheses to test. The experiment can produce a null or negative result.

**Branch workflow**

Develop E2 on a new branch named `feat/llm-evidence-experiment`, starting from the current E1 branch, `feat/github-issue-experiment`. This gives the E2 teammate the dataset, calibrated predictors, reporting game, and evaluation already developed for E1.

1. Commit this handoff on the E1 branch and make that commit available to the teammate before they create the E2 branch.
2. Continue E1 development on `feat/github-issue-experiment` and E2 development on `feat/llm-evidence-experiment`. Incorporate subsequent E1 fixes into E2 as needed, recording the E1 revision used for each comparison.
3. Merge E1 into `main` first. Then update E2 against `main` so its review shows the E2 additions, and merge E2 through its own pull request.

**1. Relationship to the idea review**

The paper allows private evidence and machine-learned estimates whose calibration can vary. E2 instantiates that evidence using an LLM's reading of source code. It retains the issue-to-module setting in Table 3, conjunctive coverage, signed private assignment incentives, and the utility in Equation 4. E2 adds an evidence comparison within the grounded experiment promised in Section 4.2. The separate name E2 and the use of an LLM as an extractor are choices made for this extension. [Idea review, Sections 3–4](FAI_Project_Idea_Review.md#3-problem-formulation)

| Paper quantity | Meaning in E2 |
| --- | --- |
| Task | A historical GitHub issue, including its shared issue text and recorded pre-fix revision. |
| Agent $i$ | A representative of a fixed source-code module, comprising local evidence gathering, a learned probability predictor, and a strategic reporting policy. |
| Private evidence $e_i$ | The shared issue context together with findings from the agent's own pre-fix module. Semantic findings have supporting source references. |
| Local belief $\pi_i$ | The ideal conditional probability $\mathbb P(z_i=1\mid e_i)$. It is generally unknown in real data. |
| Estimated belief $\hat\pi_i$ | The learned and calibrated estimate of local ownership probability supplied to the reporting game. |
| Report $r_i$ | The probability chosen by the numerical reporting policy after accounting for private incentives and the mechanism. |
| Required set $S^\star$ | The modules changed in the historical production fix, using E1's frozen labeling convention. These are proxy ownership labels. |
| Graph $G$ | The pre-fix import graph used in E1. Distinct neighboring modules determine the modeled audit degree. |
| Assigned set $A$ | The modules chosen by the mechanism from reports after any audit correction. |
| Outcome $z_i$ | One if module $i$ belongs to $S^\star$, revealed to settlement after assignment. |

**2. What the LLM contributes**

The LLM may identify that a local function implements behavior described by the issue, connect a traceback to a local call path, or explain why a referenced symbol is relevant to the suspected fault. Each retained factual finding must be supported by the allowed snapshot. A reference to an existing file alone does not establish that the explanation follows from its contents.

The primary comparison augments E1's lexical evidence with these findings. It uses the same predictor family and calibration procedure in both conditions, with each predictor fitted on its own inputs. LLM evidence is generated for the earlier training and calibration issues as well as the evaluation issues. This allows the final head to learn how predictive the semantic findings actually are.

For module i and issue t, write the two estimates as

$$
\hat\pi_{it}^{\mathrm{lex}} = C_i^{\mathrm{lex}}\big(f_i^{\mathrm{lex}}(x_{it}^{\mathrm{lex}})\big),
\qquad
\hat\pi_{it}^{\mathrm{sem}} = C_i^{\mathrm{sem}}\big(f_i^{\mathrm{sem}}(x_{it}^{\mathrm{lex}},x_{it}^{\mathrm{LLM}})\big).
$$

Here $x$ denotes features of the local evidence, $f$ the fitted predictor, and $C$ the calibration map. Both outputs lie in $[0,1]$ and refer to the same event $z_i=1$. Matching the predictor and calibration methods controls the comparison procedure. Calibration quality still has to be measured on held-out data. Even good marginal calibration does not establish that an estimate equals the full conditional posterior or contains all information relevant to ownership.

If the LLM also emits a confidence number, treat that number as an additional candidate feature. The handoff's primary definition of $\hat\pi_i$ is the calibrated head's output. A direct verbalized-confidence experiment would be a separate comparison with its own calibration assessment.

**3. Information available before assignment**

Every agent receives the shared issue text and access to its own module at the recorded base commit. Local pre-existing tests may be used if the same evidence scope is made available to both conditions. A cross-module import written inside the module is observable local evidence. Reading the imported module's implementation belongs to the neighbor's information set or an explicitly authorized audit. Each issue's evidence is collected under a fixed extraction procedure before private types and mechanism settings are supplied to reporting policies.

Keep resolving patches, future source, resolving PR discussions, later tests, and other agents' private findings outside the prediction inputs. Public module priors and the public graph may be used where the E1 protocol permits them. Full evidence is available to the pooled benchmark, and held-out ownership is available to settlement. These roles follow the paper's information structure. [Idea review, Table 2](FAI_Project_Idea_Review.md#3-problem-formulation)

Freeze the extraction procedure, probability models, calibration maps, and mechanism selection using earlier partitions. Historical labels may support training and empirical expected-utility estimation once their fixes were available. Retain the same historical reference priors $\pi_i^0$ for paired scoring comparisons. A held-out patch becomes visible only after that issue's reports and assignment are determined. Cache the resulting evidence and estimates so each incentive condition replays the same observations. Re-estimate policies for each evidence condition because the distribution of local beliefs has changed.

The learned predictors and policy estimation use information from earlier outcomes. The real-data replay approximates the paper's Bayesian game through its chosen features, belief bins, finite actions, and empirical expectations. Numerical convergence supports an approximate equilibrium in that declared empirical game. It does not by itself prove a Bayesian Nash equilibrium in the unrestricted continuous-information model.

**4. One assignment round and its mathematical contract**

1. Read the issue and local pre-fix snapshot. Extract lexical and semantic evidence and obtain $\hat\pi_i$ for every module. Freeze the estimates and supporting artifacts.
2. Give each reporting policy its permitted local information and private type $\ell_i$. Acquisition, $\ell_i<0$, and the zero-private-incentive control lead the experiment. Deflection is a robustness condition. A zero private type can still respond to assignment-dependent credit.
3. Compute reports and payload actions numerically using expected utility conditional on the agent's information. The objective is

$$
\mathbb E\left[-\ell_i\mathbf 1\{i\in A\}
+\lambda\big(s(r_i,z_i)-s(\pi_i^0,z_i)\big)
-F\mathbf 1\{\text{payload contradicted}\}+\chi_i
\mid e_i,\ell_i\right],
\qquad s(r,z)=1-(z-r)^2.
$$

The reporting solver maximizes the complete utility, including conditional credit and audit effects. Proper scoring alone elicits a subjective belief. Assignment incentives and credit can make the optimal report differ from that belief. The paper's local-linear shading result is an analytical benchmark under its assumptions. The discrete set-selection mechanism requires numerical responses. E1 uses each agent's own estimate for its subjective expected scoring term and earlier cases for assignment and credit expectations. Retain and disclose that empirical convention.

4. Apply the E1 audit protocol before selection. It uses the real graph and modeled contradiction probability

$$q_i=\alpha\left[1-(1-\delta)^{m_i}\right].$$

In E1 a contradicted altered payload incurs a fine and substitutes the public prior for selection. The original submitted report is used for ex-post scoring. Semantic relevance alone supplies no factual contradiction of a probability. E1 found that acquirers could shade reports while retaining genuine payloads, so audits had no deterrence effect in those conditions. Richer evidence may or may not change this outcome. A new rule requiring evidence for high reports would change the mechanism and needs a separate mathematical specification. [E1 reporting and audit findings](reports/e1/REPORT.md#reporting-game-and-interpretation)

5. Select A using the paper's conjunctive rule for every issue

$$
A\in\arg\max_{A\subseteq[N]}
\left\{R\prod_{i\notin A}(1-r_i)-\sum_{i\in A}c_i\right\}.
$$

Use the audit-corrected input where applicable. This rule estimates joint coverage through a product of marginals. Real module changes can be correlated, so the product is a decision approximation. Better local estimates do not make conditional independence true. Keep the selection policy fixed before seeing S* or its size k.

6. Reveal the historical ownership set and compute V(A; S*) = 1{S* is a subset of A}. Apply centered scoring to submitted reports. Settle equal-split or assigned-set Shapley credit. For a successful conjunctive assignment, Shapley pays R/k to each required owner and zero to extra assignees. On failure, the conjunctive game has zero value for every subset of A and pays zero coalition credit.
7. Record assignment welfare R V minus assignment costs, incentive expenditure, prediction errors, coverage, and padding. Preserve solver residuals and non-convergence outcomes alongside the results.

The LLM has completed its role at step 1. The reporting policy represents the agent's strategic behavior through the remaining stages.

**5. Paired experiment and baselines**

Use the existing E1 Django cohort, nine module groups, ownership convention, chronological partitions, and per-issue snapshots. The recorded cohort has 429 training, 181 validation, and 234 held-out issues, including 27 held-out multi-module fixes. These counts come from the E1 report. A smaller E2 cohort is acceptable if selected before examining E2 outcomes and if both evidence conditions are fitted and evaluated on the same declared subset. Report extraction failures and missing evidence under a predeclared handling rule. [E1 cohort](reports/e1/REPORT.md#cohort-and-information-available-to-each-agent)

Compare lexical evidence alone with lexical plus LLM evidence under the same source-access scope. This comparison measures the practical benefit of the LLM evidence stage, including its additional search effort. A claim specifically about semantic understanding would additionally require matching retrieval effort and source exposure.

For both evidence conditions, include honest reporting as a diagnostic and the same strategic polling, scoring-only, audit-only, combined, and credit-rule comparisons. Match private type draws, mechanism settings, assignment costs, audit randomness, and evaluated issues across the pair. The primary comparison uses the same E1 mechanism configurations in both conditions. When applying a budget-selected E1 configuration, check its expenditure again under semantic beliefs because payments depend on reports. A secondary budget frontier may select mechanisms separately for each evidence condition using earlier partitions and evaluate those choices on the test set.

Retain B1–B4 and the issue-text central classifier. B1 uses fixed historical module descriptions. B2 is strategic polling with the declared incentive tools off. B3 uses the same local-vote convention as E1, with beliefs from the relevant evidence condition. The issue-text classifier gives the same public-information comparator in both conditions. For B4's practical approximation, fit a pooled benchmark using the evidence available in each condition, including semantic features for the E2 condition. E1 estimates joint coverage from whole historical owner sets. Comparing E2 only with the earlier lexical pooled benchmark would give the methods different evidence. A learned pooled benchmark can be outperformed on held-out issues and should be identified as an estimate of the theoretical oracle.

**6. Evidence needed to answer the research question**

- Assess the ownership estimates before strategic reporting using held-out Brier loss, calibration, and discrimination, with per-module results and the actual test counts. Brier loss uses (z_i minus hat-pi_i) squared. It is distinct from the centered score paid on r_i.
- Compare paired welfare, conjunctive coverage, mean extra assignees, probability of any extra assignee, and report deviation from the estimated belief. Show the multi-module slice as well as the natural test mix. Small coalition counts constrain the precision of that comparison.
- Measure welfare gaps against each condition's pooled benchmark, with the miss-loss and excess-cost decomposition from Equation 6. Also show the direct welfare difference between semantic and lexical evidence under matched mechanism settings, since their benchmark values can differ. Report efficiency only when the reference mean welfare is positive.
- Report centered score transfers, expected audit costs, credit, fines, and convergence diagnostics using E1's accounting. Evidence extraction consumes resources before assignment for all queried modules. Record LLM expenditure separately and, when feasible, show welfare after evidence-acquisition cost in common declared units. Putting all extraction cost into c_i would charge only selected modules and misrepresent this workflow.
- Use paired uncertainty estimates across issues. Record LLM model/version, extraction procedure, source revisions, and randomness so repeated evidence generation can be interpreted. Claim improved evidence only where the measured estimates support it, and improved mechanism performance only where the assignment comparison supports it.

**7. Threats to the interpretation and completion conditions**

Public SWE-bench issues and fixes may have appeared in the LLM's training data. Local tools restrict retrieved information but do not remove memorized knowledge from model weights. A supported citation proves that evidence exists in the snapshot, while the reason the model selected that evidence can remain uncertain. Record the chosen model's documented cutoff when available and keep this limitation attached to E2 results. An additional cohort of issues created after a documented training cutoff can strengthen the interpretation if collected with the same labeling and information rules. Static benchmark contamination is a documented concern. [SWE-bench Goes Live!](https://arxiv.org/abs/2505.23419)

The historical changed-module label reflects one observed fix and can differ from other valid fixes. E1 also records that the supplied issue text has no recoverable edit history, so visibility of every sentence at replay time is uncertain. Retain these limitations when interpreting semantic gains. The paper assumes ex-post revelation of S* and partially verifiable artifacts. E2 must assess its evidence within those assumptions and document factual extraction errors, which can occur even without strategic manipulation. Such errors alone do not establish strategic dishonesty. [E1 information limitations](reports/e1/REPORT.md#cohort-and-information-available-to-each-agent)

The handoff is satisfied when the teammate supplies a paired lexical/semantic evaluation, evidence provenance and uncertainty, the calibrated local estimates, and the numerical mechanism comparisons with their solver qualifications. The account of E2 must make clear which probability the agent privately estimates, which report it strategically submits, what the mechanism can observe, and when the historical patch is revealed. Successful replication of the comparison, including a null or negative finding, is a valid completed experiment.
