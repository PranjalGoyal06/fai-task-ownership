---
title: Deciding Task Ownership from Private Evidence under Misaligned Incentives
date: 2026-10-18
source-file: FAI_Project_Idea_Review.pdf
---

# DECIDING TASK OWNERSHIP FROM PRIVATE EVIDENCE UNDER MISALIGNED INCENTIVES

Pranjal Goyal, Ujas Shah, Harshal Kundanlal Rana, Satyakam Mishra
Indian Institute of Technology Gandhinagar

## ABSTRACT

Many multi-agent systems must decide which agents own a task: which team owns an incident, which software modules a user story touches, which specialist answers a query. Central routers that rely on written descriptions of agent capabilities go stale, while the agents themselves hold fresh private evidence but may have incentives that distort what they report. We study assignment from evidence reports. Agents submit reports grounded in private evidence, a lightweight mechanism selects the owners, and three incentive tools discipline the reports: ex-post proper scoring, audits along a dependency graph, and credit rules for coalitions. A single signed parameter models both deflection (owning is costly) and acquisition (owning is rewarded), and a task may be owned by a set of agents rather than by one. We formulate the resulting game, outline the analytical benchmarks we plan to study (the oracle, unmediated claiming, report shading under proper scoring, and coalition coverage), and sketch experiments on a synthetic simulator and on issue-to-module data from open-source repositories. The central question is how much assignment efficiency misaligned incentives cost and how much of that loss scoring, audits, and credit rules recover.

**Keywords:** Multi-agent systems · Mechanism design · Proper scoring rules · Cooperative games · Calibration

## 1 Introduction

### 1.1 Motivation

Organizations repeatedly face the question of who should handle a piece of work. At 2 a.m. an alert fires on a checkout service, yet the fault is a configuration change several hops upstream, owned by a different team. A new user story arrives and must be assigned to the software modules it touches. A support request must reach the right specialist. In each case the party best placed to judge whether it is responsible is the candidate owner itself, because it holds detailed local evidence (its own logs, code history, or records) that no central party can cheaply collect.

The same question arises inside software built from cooperating language-model agents. Frameworks such as AutoGen and MetaGPT coordinate specialized agents through conversation or fixed roles [1, 2], and a common design places a central router in front of them that selects an agent from short textual descriptions of what each agent can do. Work on choosing among language models under a cost budget likewise treats selection as a decision taken by a central component [3].

Central routing has two weaknesses. First, descriptions go stale, because ownership and capabilities change faster than the text that records them, and misrouting is expensive: in a study of cloud incident routing, the time to diagnosis can increase by up to ten times when incidents are misrouted [4]. Second, the candidate owners' own assessments are informative but cannot be taken at face value. They can be noisy, since language models that verbalize their confidence tend to be overconfident [5]. They can also be strategic: an agent judged on its own paging load gains from reporting that the fault lies elsewhere, while an agent rewarded for owning work gains from claiming more than its share.

### 1.2 Approach

We study a middle path between a central router and unmediated self-selection. Each candidate owner submits a report grounded in its private evidence, a lightweight mechanism selects the owners, and three incentive tools discipline the reports: ex-post proper scoring once the true owners are revealed, cross-checks along a dependency or similarity graph, and credit rules for tasks owned by a coalition rather than by a single agent. The mechanism never models what an agent can do. It only aggregates and checks what agents claim, so it does not inherit the staleness of a description-based router.

### 1.3 Related work and gap

Scouts equips each team with a gatekeeper model that decides whether an incident belongs to the team [4]; its focus is the accuracy of per-team predictors rather than the incentives of the teams that own them. The strategic ingredients we need are classical. Uncoordinated claiming among symmetric agents is a volunteer's dilemma [6]. Costless reports from agents whose preferences do not depend on the state cannot be informative [7]. Proper scoring rules elicit honest probabilities [8, 9]. Audits by the mechanism resemble an inspection game [10], although detection here is exogenous rather than strategic. Credit among cooperating agents is divided by the Shapley value [11] or, on a graph, by the Myerson value [12]. Textbook treatments are given in [13, 14]. To our knowledge, these pieces have not been combined in a single model of assignment in which the sign of the incentive, the calibration of evidence produced by machine learning, and coalition ownership are studied together against a common efficiency benchmark.

### 1.4 What distinguishes the project

- **Signed misalignment.** One parameter $\ell_i$ covers deflection (owning is costly) and acquisition (owning is rewarded), so blame scenarios and credit scenarios form one model with the sign flipped.
- **Calibration meets equilibrium.** The calibration error of machine-learned evidence is a parameter of the game, and efficiency is measured as a function of it.
- **Coalitions as a first-class case.** Tasks may be owned by a set of agents, with explicit resolution rules and credit division.
- **A reusable model.** The model is domain-agnostic. Incident ownership and issue-to-module assignment are instantiations, not targets.

### 1.5 Question and outline

We investigate how much assignment efficiency is lost, relative to an oracle, when agents hold private and partially verifiable evidence about a task and some of them are rewarded or penalized for owning it, and how much of that loss ex-post scoring, audits, and credit rules recover. Section 2 fixes notation and definitions, Section 3 states the model and the research questions formally, and Section 4 outlines the work planned for the final submission. Team contributions are given in Section 5.

## 2 Preliminaries

Table 1 collects the notation used throughout. Vectors are written in bold, for example $\mathbf{r} = (r_1, \ldots, r_N)$.

**Definition 1 (Resolution rules).** Let $A \subseteq [N]$ be the set of agents assigned to a task with required set $S^\star$. The conjunctive rule is $V_\wedge(A; S^\star) = \mathbf{1}[S^\star \subseteq A]$, so every required owner must be assigned. The disjunctive rule is $V_\vee(A; S^\star) = \mathbf{1}[A \cap S^\star \neq \emptyset]$, so one required owner suffices. For $|S^\star| = 1$ the two rules coincide, and we write $V$.

**Definition 2 (Strictly proper scoring rule).** A function $s\colon [0,1] \times \{0,1\} \to \mathbb{R}$ is a proper scoring rule if, for every $p \in [0,1]$, the map $r \mapsto p\, s(r,1) + (1-p)\, s(r,0)$ is maximized at $r = p$, and strictly proper if $r = p$ is the unique maximizer. The Brier score $s(r,z) = 1 - (z - r)^2$ is strictly proper [8, 9]. For a public reference forecast $\pi_i^0$ (the base rate at which agent $i$ is required), the centered score is $\tilde{s}_i(r,z) = s(r,z) - s(\pi_i^0, z)$. It has the same maximizers as $s$, since the subtracted term does not depend on $r$, and an agent that reports $\pi_i^0$ earns zero in expectation.

**Definition 3 (Calibration error).** For a confidence $\hat\pi \in [0,1]$ and outcome $z \in \{0,1\}$, the expected calibration error is $\mathrm{ECE} = \mathbb{E}\big[\,|\mathbb{P}(z=1 \mid \hat\pi) - \hat\pi|\,\big]$, estimated in practice by binning [15].

**Definition 4 (Shapley and Myerson values).** Let $v\colon 2^{[N]} \to \mathbb{R}$ with $v(\emptyset) = 0$ be a cooperative game. The Shapley value of agent $i$ is

$$\psi_i(v) = \sum_{T \subseteq [N]\setminus\{i\}} \frac{|T|!\,(N - |T| - 1)!}{N!}\big[v(T \cup \{i\}) - v(T)\big]. \tag{1}$$

Given a graph $G$, the graph-restricted game is $v^G(T) = \sum_C v(C)$, where $C$ ranges over the connected components of the subgraph of $G$ induced by $T$, and the Myerson value is $\psi(v^G)$ [11, 12].

**Table 1: Notation.**

| Symbol | Meaning |
|---|---|
| $N$, $[N]$ | number of agents; the agent set $\{1, \ldots, N\}$ |
| $S^\star \subseteq [N]$, $k = \lvert S^\star\rvert$ | latent set of required owners of a task; its size |
| $z_i = \mathbf{1}[i \in S^\star]$ | membership indicator of agent $i$ |
| $G = ([N], E)$, $\mathcal{N}(i)$, $m_i$ | relation graph (possibly empty); neighbours of $i$; $m_i = \lvert\mathcal{N}(i)\rvert$ |
| $e_i$ | private evidence observed by agent $i$ |
| $\pi_i^\star$ | full-information posterior $\mathbb{P}(z_i = 1 \mid e_1, \ldots, e_N)$ |
| $\pi_i$ | local belief $\mathbb{P}(z_i = 1 \mid e_i)$ |
| $\hat\pi_i$ | agent's own estimate of $\pi_i$, possibly miscalibrated |
| $r_i \in [0,1]$ | report submitted by agent $i$ |
| $A \subseteq [N]$, $x_i(\mathbf{r})$ | assigned set; probability $\mathbb{P}(i \in A \mid \mathbf{r})$ that $i$ is assigned |
| $V(A; S^\star) \in \{0,1\}$ | resolution indicator (Definition 1) |
| $R$, $c_i$ | value of resolving a task; social cost of assigning agent $i$ |
| $\ell_i \in \mathbb{R}$ | private net cost of being assigned (signed) |
| $\phi$ | fraction of agents with $\ell_i \neq 0$ |
| $\lambda$, $s$, $\tilde{s}_i$ | score scale; Brier score; score centered at the prior $\pi_i^0$ |
| $\pi_i^0$, $X_i(r_i)$ | prior probability that $i \in S^\star$; interim assignment probability |
| $\alpha$, $\delta$, $q_i$, $F$ | audit rate; per-neighbour detection probability; contradiction probability; penalty |
| $\chi_i$, $\psi_i$ | credit paid to agent $i$; Shapley value of agent $i$ |
| $W$, $\mathrm{Reg}$, $\eta$ | social welfare; regret against the oracle; efficiency ratio |

**Table 2: Information structure.**

| Quantity | Agent $i$ | Mechanism | Ground truth |
|---|---|---|---|
| $S^\star$ | unknown | unknown | revealed after resolution |
| $e_i$ | known | payload only, partly checkable | not revealed |
| $e_j$, $j \neq i$ | unknown | payloads only | not revealed |
| $G$, $c_i$, $R$ | known | known | — |
| $\ell_i$ | known to $i$ | unknown | not revealed |
| calibration of $\hat\pi_i$ | unknown | estimated from history | estimable from revealed outcomes |

## 3 Problem Formulation

### 3.1 Tasks, agents, and information

Fix $N \geq 2$ agents and a relation graph $G$, which may be empty. A task draws a required set $S^\star \subseteq [N]$ from a prior and an evidence vector $\mathbf{e} = (e_1, \ldots, e_N)$ from a likelihood $\mathbb{P}(\cdot \mid S^\star, G)$. Agent $i$ observes only $e_i$.

**Assumption 1 (Conditional independence).** Given $\mathbf{e}$, the indicators $z_1, \ldots, z_N$ are conditionally independent with $\mathbb{P}(z_i = 1 \mid \mathbf{e}) = \pi_i^\star$.

This holds exactly when memberships are independent a priori and $e_i$ depends only on $z_i$. Otherwise it is an approximation used to obtain analytical benchmarks; it is violated, for example, when evidence is coupled through $G$, and exact posteriors are then computed numerically for small $N$. It is not needed when $|S^\star| = 1$.

**Assumption 2 (Calibrated beliefs in the analysis).** In the closed-form analysis, $\hat\pi_i = \pi_i$. Miscalibration enters only in the experiments, through a distortion map $\hat\pi_i = g_i(\pi_i)$.

**Assumption 3 (Ex-post verification).** The full set $S^\star$ is revealed after resolution, for example by a post-mortem or by the resolving patch, and payloads are verifiable artefacts (such as logs) that other agents cannot forge.

Each agent $i$ has a private type $\ell_i \in \mathbb{R}$, the net private cost of being assigned. If $\ell_i > 0$ the agent is a deflector, if $\ell_i < 0$ an acquirer, and if $\ell_i = 0$ it is aligned with the system. A fraction $\phi$ of the agents has $\ell_i \neq 0$. The social cost $c_i \geq 0$ of assigning agent $i$ (tokens, hours) is public and differs from $\ell_i$. Table 2 lists what each party knows.

Agent $i$ controls its report $r_i \in [0,1]$ and the evidence payload attached to it, and does not control whether it is assigned. The mechanism controls the selection rule, the score scale $\lambda$, the audit rate $\alpha$, and the credit rule $\chi$. Figure 1 summarizes one assignment round.

```
[Task with hidden owner set S*]
  -> [Private evidence e_i, estimate π̂_i]
  -> [Report r_i and evidence payload]
  -> [Cross-check, then select set A]
  -> [Resolve, reveal S*, score, credit]

(incentives through λ, χ, F)
```

**Figure 1:** One assignment round. Agents control the report; the mechanism controls selection, audits, scoring, and credit.

### 3.2 Mechanisms and agent utilities

**Definition 5 (Assignment mechanism).** A mechanism $M = (A(\cdot), \lambda, \alpha, \chi)$ consists of a selection rule mapping reports to an assigned set $A(\mathbf{r})$, a score scale $\lambda \geq 0$, an audit rate $\alpha \in [0,1]$, and a credit rule $\chi$.

For single-owner and disjunctive tasks the selection rule is escalation: agents are ordered by $r_i / c_i$ in nonincreasing order and assigned one after another until the task is resolved, so that $A$ is the set of agents reached. For conjunctive tasks the rule is set selection, which treats reports as posteriors:

$$A(\mathbf{r}) \in \arg\max_{A \subseteq [N]} \Big\{ R \prod_{i \notin A} (1 - r_i) - \sum_{i \in A} c_i \Big\}. \tag{2}$$

With probability $\alpha$ the mechanism cross-checks the payload of agent $i$ against the payloads of its neighbours, and each neighbour independently exposes an altered payload with probability $\delta$. The mechanism, not the neighbours, plays the inspector and commits to $\alpha$. An altered payload is therefore contradicted with probability

$$q_i = \alpha\big[1 - (1 - \delta)^{m_i}\big]. \tag{3}$$

After resolution $S^\star$ is revealed (assumption 3), and agent $i$ receives the centered score $\lambda\, \tilde{s}_i(r_i, z_i)$ with $s$ the Brier score. Credit is paid only to assigned agents, so $\chi_i = 0$ for $i \notin A$. For $i \in A$ two credit rules are compared: equal split, $\chi_i = R\, V(A; S^\star)/|A|$, and Shapley credit, $\chi_i = \psi_i(v_A)$ with $v_A(T) = R\, V(T; S^\star)$ for $T \subseteq A$. The utility of agent $i$ is

$$U_i = -\ell_i\, \mathbf{1}[i \in A] + \lambda\, \tilde{s}_i(r_i, z_i) - F\, \mathbf{1}[\text{payload of } i \text{ contradicted}] + \chi_i, \tag{4}$$

and agent $i$ chooses $r_i$ as a function of $(e_i, \ell_i)$ to maximize $\mathbb{E}[U_i \mid e_i, \ell_i]$. A strategy profile is a Bayesian Nash equilibrium if every agent's choice is a best response to the others.

For fixed $\mathbf{r}_{-i}$ the realized assignment is a step function of $r_i$, but agent $i$ responds to its interim assignment probability $X_i(r_i) = \mathbb{E}\big[x_i(r_i, \mathbf{r}_{-i}) \mid e_i, \ell_i\big]$, where the expectation is over the task and over the other agents' types and reports. $X_i$ is continuous when the others' reports are atomless, and it is the object that the analysis differentiates. Let $\bar\chi_i = \mathbb{E}[\chi_i \mid i \in A, e_i]$, treated as locally independent of $r_i$. Since credit is paid only on assignment, the report-dependent part of $\mathbb{E}[U_i \mid e_i, \ell_i]$ is $-\ell_i^{\mathrm{eff}} X_i(r_i) + \lambda\, \mathbb{E}[\tilde{s}_i(r_i, z_i) \mid e_i]$ with $\ell_i^{\mathrm{eff}} = \ell_i - \bar\chi_i$, apart from the audit term. We keep $\ell_i$ as the primitive type, and use $\ell_i^{\mathrm{eff}}$ when credit rules are active.

### 3.3 Objective, oracle, and regret

The social welfare of a mechanism, evaluated at equilibrium reports, is

$$W(M) = \mathbb{E}\big[R\, V(A; S^\star) - c(A)\big], \qquad c(A) = \sum_{i \in A} c_i. \tag{5}$$

The oracle $M^\circ$ observes $\mathbf{e}$ and chooses $A^\circ$ to maximize $\mathbb{E}[R\, V(A; S^\star) - c(A) \mid \mathbf{e}]$. The regret and the efficiency ratio of $M$ are

$$\mathrm{Reg}(M) = W(M^\circ) - W(M) = \underbrace{R\, \mathbb{E}\big[V(A^\circ) - V(A)\big]}_{\Delta V\ (\text{miss loss})} + \underbrace{\mathbb{E}\big[c(A) - c(A^\circ)\big]}_{\Delta C\ (\text{excess cost})}, \qquad \eta(M) = \frac{W(M)}{W(M^\circ)}, \tag{6}$$

where $\eta$ is defined when $W(M^\circ) > 0$ and plays the role of a price-of-anarchy ratio [16] with the oracle in place of the social optimum. Incentive tools are not free. With $c_a$ the cost of one audit, their expected cost is $\mathrm{Cost}_{\mathrm{inc}}(\lambda, \alpha) = \lambda \sum_i \mathbb{E}[\tilde{s}_i(r_i, z_i)] + c_a\, \alpha N$. Centering ensures that the budget pays for information rather than for a baseline score that every agent earns.

**Table 3: Instantiations of the model.** The sign is that of $\ell_i$.

| Application | Agents | Required set $S^\star$ | Graph $G$ | Sign | Resolution |
|---|---|---|---|---|---|
| Incident ownership | service teams | teams containing the fault | call graph | $+$ | either |
| Issue to module | module owners | modules changed by the fix | import graph | $-$ or $0$ | conjunctive |
| Specialist routing | specialist agents | capable agents | skill overlap | $+$ | disjunctive |
| Reviewer assignment | reviewers | qualified reviewers | ownership graph | $+$ | disjunctive |

### 3.4 Baselines

- **B1, description router.** Applies the selection rule to stale description-based scores $\tilde\pi_i$ in place of reports, where a fraction $\rho$ of the description entries is outdated.
- **B2, polling router.** Applies the selection rule to the reports with no incentive layer, that is, $M_{0,0}$. It matches the oracle if all agents are truthful and calibrated and $\pi_i = \pi_i^\star$.
- **B3, central decider with voting.** Each agent votes for the agent it believes owns the task, and the plurality winner is assigned.
- **B4, oracle.** $M^\circ$.

### 3.5 Research questions

Let $\Theta = (N, k, G, \phi, \ell, \rho, \gamma)$ collect the instance parameters, where $\gamma$ is the probability that an agent's evidence is missing. The project addresses three questions.

- **RQ1 Cost of misalignment.** How do $\mathrm{Reg}$, $\Delta V$, and $\Delta C$ of the polling router B2 depend on $\phi$ and on the sign of $\ell_i$?
- **RQ2 Recovery.** For a given incentive budget $B$, how much of that regret is recovered by scoring and audits, that is, what is the value of

$$\min_{\lambda \geq 0,\ \alpha \in [0,1]} \mathrm{Reg}(M_{\lambda,\alpha,\chi}) \quad \text{subject to} \quad \mathrm{Cost}_{\mathrm{inc}}(\lambda, \alpha) \leq B, \tag{7}$$

  with $\mathrm{Reg}$ evaluated at an equilibrium of the induced reporting game, and how does report shading depend on $\lambda$ and on $\ell_i$?
- **RQ3 Coalitions.** How do coverage and padding depend on the coalition size $k$, on the resolution rule, and on the credit rule?

### 3.6 Instantiations

The model is fixed and only the environment, that is, how $S^\star$, $\mathbf{e}$, and $G$ are generated, changes between applications. Table 3 lists four instantiations.

## 4 Proposed Solutions

The plan below is intentionally open-ended. The model of Section 3 is the core of the project, and the scope, methods, and experiments listed here will be refined as the work progresses.

### 4.1 Planned theoretical work

- **Benchmarks without incentives.** Characterize the oracle and how a stale or polling router deviates from it, and study unmediated claiming in both the deflection and the acquisition regimes.
- **Reporting incentives.** Study how proper scoring shapes the reports of misaligned agents, and when reports are uninformative without it.
- **Audits.** Study when graph-based cross-checks deter manipulated evidence.
- **Coalitions.** Study coverage under the two resolution rules and compare credit rules.

Where simplifying assumptions allow, we aim for closed forms and check them numerically; otherwise results will be numerical only.

### 4.2 Experiments

We plan a synthetic simulator built on the model, in which instance parameters are swept and equilibrium reports are computed numerically, and at least one grounded instantiation that uses issue-to-module data from open-source repositories [17]. Other applications, such as incident ownership, are optional. We compare against the baselines B1–B4 and against ablations that switch scoring, audits, and credit rules on and off, and we measure regret, efficiency, incentive cost, and coalition coverage. Our working hypothesis is that regret under misalignment grows with the share of misaligned agents and that the incentive layer pays for itself beyond some share; we also want to learn where it does not.

### 4.3 Limitations and risks

- Incentives are injected, since real agents do not necessarily deflect or acquire, so the study measures robustness to misalignment as an explicit assumption.
- Ex-post verification and unforgeable payloads (assumption 3) are strong, and labels in real data are proxies for true ownership.
- Independence assumptions and perfect inspection are simplifications, so analytical results serve as benchmarks.
- Collusion, strategic cross-checks by neighbours, and voluntary disclosure are outside the scope of this project.
- The main schedule risk is scope, which we manage by staging the work and dropping optional parts first.

## 5 Contributions of Team Members

All members took part in the ideation of the problem and the model. The division of work below is tentative and may change as the project progresses.

| Member | Assigned tasks |
|---|---|
| Pranjal | Real-data experiment pipeline; coalition and credit analysis |
| Ujas | Analytical benchmarks for the model; evaluation, plots, and report writing |
| Harshal | Literature review and related work; synthetic simulator |
| Satyakam | Incentive mechanisms (scoring and audits); formalizing the model and notation |

## References

[1] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Awadallah, Ryen W. White, Doug Burger, and Chi Wang. AutoGen: Enabling next-gen LLM applications via multi-agent conversation. arXiv preprint arXiv:2308.08155, 2023.

[2] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, et al. MetaGPT: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations (ICLR), 2024.

[3] Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv preprint arXiv:2305.05176, 2023.

[4] Jiaqi Gao, Nofel Yaseen, Robert MacDavid, Felipe Vieira Frujeri, Vincent Liu, Ricardo Bianchini, Ramaswamy Aditya, Xiaohang Wang, Henry Lee, David Maltz, Minlan Yu, and Behnaz Arzani. Scouts: Improving the diagnosis process through domain-customized incident routing. In Proceedings of the ACM SIGCOMM Conference, pages 253–269. ACM, 2020.

[5] Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In International Conference on Learning Representations (ICLR), 2024.

[6] Andreas Diekmann. Volunteer's dilemma. Journal of Conflict Resolution, 29(4):605–610, 1985.

[7] Vincent P. Crawford and Joel Sobel. Strategic information transmission. Econometrica, 50(6):1431–1451, 1982.

[8] Glenn W. Brier. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1–3, 1950.

[9] Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477):359–378, 2007.

[10] Rudolf Avenhaus, Bernhard von Stengel, and Shmuel Zamir. Inspection games. In Robert J. Aumann and Sergiu Hart, editors, Handbook of Game Theory with Economic Applications, volume 3, pages 1947–1987. Elsevier, 2002.

[11] Lloyd S. Shapley. A value for n-person games. In Harold W. Kuhn and Albert W. Tucker, editors, Contributions to the Theory of Games, Volume II, number 28 in Annals of Mathematics Studies, pages 307–317. Princeton University Press, 1953.

[12] Roger B. Myerson. Graphs and cooperation in games. Mathematics of Operations Research, 2(3):225–229, 1977.

[13] Yoav Shoham and Kevin Leyton-Brown. Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations. Cambridge University Press, 2008.

[14] Noam Nisan, Tim Roughgarden, Éva Tardos, and Vijay V. Vazirani, editors. Algorithmic Game Theory. Cambridge University Press, 2007.

[15] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine Learning Research, pages 1321–1330, 2017.

[16] Elias Koutsoupias and Christos Papadimitriou. Worst-case equilibria. In Proceedings of the 16th Annual Symposium on Theoretical Aspects of Computer Science (STACS), volume 1563 of Lecture Notes in Computer Science, pages 404–413. Springer, 1999.

[17] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations (ICLR), 2024.
