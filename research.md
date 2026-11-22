# Research Results and Research-Question Analysis

## Scope and evidence

This document analyzes the three research questions in
`FAI_Project_Idea_Review.md` using:

- the current synthetic implementation;
- 57 passing automated tests;
- seeded simulations using 8 agents and 100 rounds;
- five seeds (42-46) for each RQ sweep point;
- shared model and metric definitions from `src/simulation_runner.py` and `src/metrics.py`.

Raw and summarized outputs are in:

- [research_evidence.csv](./results/research_evidence.csv)
- [rq1_summary.csv](./results/rq1_summary.csv)
- [rq2_summary.csv](./results/rq2_summary.csv)
- [rq3_summary.csv](./results/rq3_summary.csv)

This is a synthetic-model analysis. It is not an answer for real repositories or real human teams. The real issue-to-module experiment remains future work, and the honest or full-information metric was removed from the comparison where it was not one of the implemented baselines.

## Core metrics

For each task, gross welfare is

$$W(M)=R\,V(A)-\sum_{i\in A}c_i.$$

Regret is measured against the ground-truth oracle:

$$\mathrm{Reg}(M)=W(M^\circ)-W(M)
=\Delta V+\Delta C,$$

where $\Delta V$ is miss loss and $\Delta C$ is excess assignment cost. The polling baseline B2 applies the mechanism selection rule to reports without scoring or audits.

---

## RQ1 - Cost of misalignment

### Research question

How do polling-router regret, miss loss $\Delta V$, and excess cost $\Delta C$ change with the fraction of misaligned agents $\phi$ and with the sign of $\ell$?

### Theory

For deflectors, $\ell_i>0$: ownership is privately costly, so a rational report should shift toward 0. This may reduce assignment of true owners and increase miss loss.

For acquirers, $\ell_i<0$: ownership is privately attractive, so a rational report should shift toward 1. This may assign agents that are not required and increase excess cost.

Therefore, the sign is expected to separate the two loss components:

- deflection: miss loss should increase;
- acquisition: excess cost should increase;
- removing misalignment: aligned agents should make both components approach the no-strategic-loss control.

### Observations

![RQ1 polling regret](./results/plots/rq1_b2_regret.png)

![RQ1 regret decomposition](./results/plots/rq1_regret_decomposition.png)

| Sign | $\phi$ | B2 regret | B2 miss loss $\Delta V$ | B2 excess cost $\Delta C$ | Mechanism regret | Regret recovered |
|---|---:|---:|---:|---:|---:|---:|
| Deflection | 0.25 | 11.606 | 10.530 | 1.076 | 5.352 | 6.254 |
| Deflection | 0.50 | 13.358 | 13.530 | -0.172 | 6.568 | 6.790 |
| Deflection | 0.75 | 12.878 | 14.430 | -1.552 | 6.896 | 5.982 |
| Deflection | 1.00 | 11.848 | 14.810 | -2.962 | 7.644 | 4.204 |
| Acquisition | 0.25 | 4.154 | 1.710 | 2.444 | 4.374 | -0.220 |
| Acquisition | 0.50 | 4.136 | 1.480 | 2.656 | 4.376 | -0.240 |
| Acquisition | 0.75 | 4.240 | 1.260 | 2.980 | 4.386 | -0.146 |
| Acquisition | 1.00 | 4.396 | 1.069 | 3.326 | 4.396 | 0.000 |

Deflection strongly matches the expected sign effect. At $\phi=0.50$, miss loss is 13.530 and excess cost is slightly negative, while B2 resolution falls to 17% in this sweep. Increasing deflection from 25% to 100% keeps miss loss dominant.

Acquisition instead is dominated by excess cost: 2.444 at 25% and 3.326 at 100%. In this implementation, the tested Mechanism M does not materially reduce acquisition regret and, with the combined mixed setting used here, may be slightly worse than B2. This is not evidence that the mechanism's tools are ineffective in general; it shows that the fixed scoring/audit/credit setting was not sufficient for the pure acquisition sweep.

### RQ1 conclusion

Supported, with a sign-specific qualification:

- higher deflector prevalence creates a large and persistent miss-loss cost;
- the mechanism recovers 35.6%-50.8% of deflection regret in the nonzero-deflection sweep;
- higher acquirer prevalence increases excess assignment cost;
- the pure acquisition sweep does not show statistically meaningful recovery in the tested setting.

The result supports the paper's signed-misalignment model. Deflection and acquisition are not symmetric in this setup; they affect different parts of the oracle-regret decomposition.

---

## RQ2 - Recovery by scoring and audits

### Research question

How much B2 regret is recovered by report scoring and graph audits, and how does recovery interact with incentive cost, shading, and tampering?

### Theory

The centered Brier rule is strictly proper, so reporting the true belief uniquely maximizes expected centered score. A larger $\lambda$ should reduce the private-cost advantage of extreme shading.

Graph audits should reduce payload tampering. Their immediate effect on the current assignment is limited because static audits occur after selection. Gross assignment welfare may therefore improve mainly through report discipline rather than retroactive correction.

Net recovery should count centered-score transfers and audit cost. A setting can reduce gross regret while not improving net welfare.

### Observations

![RQ2 regret grid](./results/plots/rq2_regret_heatmap.png)

![RQ2 report shading](./results/plots/rq2_shading.png)

| $\lambda$ | $\alpha$ | Mechanism regret | Gross recovery vs B2 | Net welfare | Absolute shading M | Tampering rate | Incentive overhead |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 12.700 | -0.564 | -0.852 | 0.527 | 63.38% | 0.000 |
| 0.0 | 0.8 | 12.700 | -0.564 | -1.498 | 0.527 | 0.00% | 0.646 |
| 1.0 | 0.0 | 6.884 | 5.252 | 6.066 | 0.386 | 72.73% | -1.102 |
| 2.5 | 0.0 | 5.364 | 6.772 | 6.864 | 0.260 | 72.90% | -0.380 |
| 2.5 | 0.2 | 5.364 | 6.772 | 6.698 | 0.260 | 26.28% | -0.214 |
| 6.0 | 0.0 | 4.400 | 7.736 | 4.386 | 0.138 | 75.05% | 3.062 |
| 6.0 | 0.8 | 4.400 | 7.736 | 3.746 | 0.138 | 0.00% | 3.702 |

B2 mean regret was 12.136 in this grid. Scoring at $\lambda=1.0$ recovered 43.0% of that regret. Scoring at $\lambda=2.5$ recovered 55.8%, and $\lambda=6.0$ recovered 63.6%.

Absolute Mechanism shading fell monotonically:

$$0.527\;(\lambda=0)\rightarrow0.386\;(1)\rightarrow0.260\;(2.5)\rightarrow0.138\;(6).$$

This is the clearest evidence that stronger proper scoring contracts reports toward private beliefs.

Audits reduce tampering substantially at positive scores:

- $\lambda=2.5$: 72.90% tampering at $\alpha=0$, 26.28% at $\alpha=0.2$, and 0% at $\alpha=0.5$;
- $\lambda=6.0$: 75.05% at $\alpha=0$, 26.20% at $\alpha=0.2$, and 0% at $\alpha=0.5$.

The audit-only result is weaker: at $\lambda=0$, changing $\alpha$ from 0 to 0.8 reduces tampering from 63.38% to 0 but does not change the gross regret of 12.700. This matches the theory: static audits deter future/other tampering but cannot repair a report used for the already-completed assignment.

Net welfare favors moderate scoring in this grid. $\lambda=2.5$, $\alpha=0$ had the highest mean net welfare (6.864). $\lambda=6.0$ had lower gross regret but the largest incentive overhead and lower net welfare.

### RQ2 conclusion

Supported:

- strong scoring directly reduces mean absolute report shading;
- audits sharply reduce or eliminate measured tampering;
- audits alone do not provide gross assignment recovery;
- scoring plus audits recovers between 43% and 64% of B2 regret in the tested grid;
- recovery is not identical to net benefit: very strong scoring has higher gross recovery but a larger transfer cost.

The likely practical result is a moderate scoring scale around $\lambda=2.5$ rather than the maximum $\lambda=6.0$ in this experiment.

---

## RQ3 - Coalition ownership

### Research question

How do required-owner coverage and assignment padding vary with coalition size $k$, resolution rule, and credit rule?

### Theory

Under conjunctive resolution, every required owner must be assigned. Coverage should rise as $k$ rises and the mechanism becomes more complete, but padding is especially sensitive to acquisition and Equal Split.

Under disjunctive resolution, one required owner suffices, so additional assigned agents are usually unnecessary. Mean coverage of a required set can decrease toward $1/k$ as $k$ grows because only one owner needs to be selected.

Under Shapley credit, an assigned non-owner should receive zero credit in a conjunctive resolved coalition. Equal Split divides resolved value equally among all assigned agents, so it may reward padding.

### Observations

![RQ3 conjunctive coverage](./results/plots/rq3_coverage.png)

![RQ3 conjunctive padding](./results/plots/rq3_padding.png)

| Rule | $k$ | Equal-Split coverage | Shapley coverage | Equal-Split padding | Shapley padding | Equal-Split regret | Shapley regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conjunctive | 1 | 0.718 | 0.488 | 1.600 | 1.094 | 2.728 | 3.142 |
| Conjunctive | 2 | 0.935 | 0.950 | 4.422 | 4.392 | 5.592 | 5.292 |
| Conjunctive | 3 | 0.961 | 0.970 | 3.904 | 4.028 | 5.556 | 5.288 |
| Conjunctive | 5 | 0.993 | 0.993 | 2.850 | 2.786 | 3.666 | 3.650 |
| Conjunctive | 8 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Disjunctive | 1 | 1.000 | 1.000 | 1.402 | 0.880 | 1.402 | 0.880 |
| Disjunctive | 2 | 0.500 | 0.500 | 0.524 | 0.406 | 0.524 | 0.406 |
| Disjunctive | 3 | 0.333 | 0.333 | 0.242 | 0.216 | 0.216 | 0.216 |
| Disjunctive | 5 | 0.200 | 0.200 | 0.108 | 0.110 | 0.108 | 0.110 |
| Disjunctive | 8 | 0.125 | 0.125 | 0.000 | 0.000 | 0.000 | 0.000 |

The conjunctive results show strong coverage as $k$ increases:

- $k=1$: 71.8% Equal-Split coverage and 48.8% Shapley coverage;
- $k=5$: 99.3% under both credit rules;
- $k=8$: 100% under both rules.

Padding is large in medium conjunctive coalitions and not monotonic in $k$ in this run: the highest mean padding was 4.422 at $k=2$ under Equal Split and 4.392 under Shapley.

In disjunctive settings, Shapley generally produced less padding than Equal Split:

- $k=1$: 1.402 mean padding under Equal Split versus 0.880 under Shapley;
- $k=2$: 0.524 versus 0.406;
- $k=3$: 0.242 versus 0.216.

The difference is smaller at larger $k$, and both rules converge to no padding at $k=8$. At $k=5$, Equal Split had slightly lower regret in this run, so the credit-rule advantage is not uniform.

### RQ3 conclusion

Supported for the main structural predictions:

- conjunctive coverage improves substantially with $k$ and approaches completeness;
- padding is most visible in conjunctive medium-size coalitions;
- disjunctive tasks require at most one of the required owners, and padding collapses as the fixed coalition behavior becomes more complete;
- Shapley reduces padding in small disjunctive coalitions compared with Equal Split;
- no universal credit-rule winner is observed; Shapley was better in most small/medium cases, while Equal Split was marginally better in isolated cases.

The closed-form and implementation credit tests provide separate unit-level support:

- Shapley gives zero credit to an unneeded conjunctive agent;
- Equal Split gives that agent an equal share;
- closed-form Shapley matches brute-force Shapley for all tested resolution and assigned-set combinations.

---

## Cross-RQ conclusion

The evidence supports the central project claim that:

1. misalignment creates measurable assignment regret;
2. deflection mainly increases miss loss, while acquisition mainly increases excess cost;
3. proper scoring is the most reliable recovery tool in this implementation;
4. audits deter tampering but are insufficient as a gross-welfare repair mechanism by themselves;
5. coalition resolution depends on both task structure and credit allocation.

The strongest combined synthetic result is not the maximum score scale: it is the moderate scoring point $\lambda=2.5$, with low or moderate auditing, which balances recovery and incentive cost. The strongest gross score point was $\lambda=6.0$, but it had three times the measured incentive overhead of $\lambda=2.5$ in the audit-free comparison.

## Limitations

- The experiments use 8 agents and 100 rounds for 5 seeds, not the 10,000-task, 20-seed target proposed in the roadmap.
- No confidence intervals or bootstrap intervals are reported in this first evidence pass.
- The repository/issue experiment is not implemented.
- The oracle uses the hidden owner set and is an optimistic upper benchmark.
- Audits are executed after assignment, so they cannot repair the current round.
- RQ3 coverage and padding are empirical averages; they should be paired with uncertainty intervals before publication.
- Dynamic workload is separate from the static theoretical model and was not used in the RQ sweeps.

## Verification

- Full test suite: **57 passed**.
- Test inventory: [test_description.md](./test_description.md).
