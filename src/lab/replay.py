"""Fit a reporting policy on earlier cases, replay it, then reveal the patch for settlement."""
import argparse
import io
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from src.e1.dataset import MODULES
from src.e1.game import ACTIONS, BITS, Condition, fit_strategy, information, population, select, submit
from src.e1.replay import SEED, settle
from src.lab.analysis import compare
from src.lab.preset import DjangoPreset, load_preset
from src.mechanisms import AssignmentMechanism
from src.types import CreditRule, MechanismConfig, ResolutionRule, Task


@dataclass(frozen=True)
class Settings:
    private_cost: float = -1.0  # E1 ell: negative values reward assignment
    strategic_agents: int = 6
    scoring: float = 8.0
    audit: float = 0.0
    credit: str = "off"
    seed: int = SEED
    reward: int = 8
    detection: float = .4
    fine: float = 2.

    def condition(self):
        if type(self.strategic_agents) is not int or not 0 <= self.strategic_agents <= len(MODULES):
            raise ValueError("Choose a whole number of strategic agents from zero to nine")
        if self.credit not in ("off", "equal", "shapley"):
            raise ValueError("Unknown credit rule")
        numbers = (self.audit, self.private_cost, self.scoring, self.detection, self.fine)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers):
            raise ValueError("Reporting parameters must be finite numbers")
        if not 0 <= self.audit <= 1 or self.scoring < 0:
            raise ValueError("Invalid reporting-game parameters")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("The seed must be a 32-bit unsigned integer")
        if self.reward not in (5, 8, 12) or not 0 <= self.detection <= 1 or self.fine < 0:
            raise ValueError("Invalid resolution reward, detection probability or fine")
        return Condition(
            reward=self.reward, delta=self.detection, fine=self.fine, ell=self.private_cost,
            phi=self.strategic_agents / len(MODULES), scale=self.scoring, audit=self.audit,
            credit=self.credit,
        )

    def polling(self):
        return replace(self, scoring=0., audit=0., credit="off")


@dataclass
class Run:
    settings: Settings
    diagnostic: dict
    issues: pd.DataFrame
    agents: pd.DataFrame


def replay(preset: DjangoPreset, settings: Settings) -> Run:
    c = settings.condition()
    policy, thresholds, diagnostic = fit_strategy(preset.validation_beliefs, preset.validation_outcomes,
                                                 preset.validation_degree, preset.prior, c, settings.seed)
    beliefs = preset.beliefs
    types, checks, exposure = population(settings.seed + 1, len(beliefs), c.phi, len(MODULES))
    info = information(beliefs, thresholds, types)
    reports, effective, caught = submit(policy, beliefs, info, (checks, exposure), preset.degrees,
                                        preset.prior, c)
    assigned = select(effective, c.reward)
    selected = (assigned[:, None] & BITS) != 0

    # Held-out labels enter only after all assignment decisions have been made.
    y = preset.outcomes
    issues = settle(preset.issues.index, y, assigned, preset.pooled_assignments[c.reward], c.reward).set_index("issue")
    audited = checks < c.audit
    rule = CreditRule.SHAPLEY if c.credit == "shapley" else CreditRule.EQUAL_SPLIT
    mechanism = AssignmentMechanism(MechanismConfig(credit_rule=rule))
    payouts = np.zeros_like(reports)
    if c.credit != "off":
        for i, outcome in enumerate(y):
            task = Task(i, np.array([]), -1, set(np.flatnonzero(outcome)), outcome, c.reward,
                        np.ones(len(MODULES)), ResolutionRule.CONJUNCTIVE)
            for agent, value in mechanism.divide_credit(task, set(np.flatnonzero(selected[i]))).items():
                payouts[i, agent] = value
    scores = c.scale * np.vectorize(mechanism.centered_brier_score)(reports, y, preset.prior)
    issues["score"] = scores.sum(axis=1)
    issues["incentive_cost"] = issues.score + c.audit_cost * c.audit * len(MODULES)
    issues["credit_paid"] = payouts.sum(axis=1)
    issues["fines"] = c.fine * caught.sum(axis=1)
    issues["assignments"] = selected.sum(axis=1)
    issues["audits"] = audited.sum(axis=1)
    issues["messages"] += (2 * audited * preset.degrees).sum(axis=1)
    agents = pd.DataFrame({
        "private_cost": (c.ell * types).ravel(), "belief": beliefs.ravel(), "report": reports.ravel(),
        "altered_payload": (policy[np.arange(len(MODULES)), info] >= ACTIONS).ravel(),
        "audited": audited.ravel(), "contradicted": caught.ravel(), "selection_input": effective.ravel(),
        "assigned": selected.ravel(), "patch_owner": y.astype(bool).ravel(),
        "score": scores.ravel(), "credit": payouts.ravel(),
    }, index=pd.MultiIndex.from_product([issues.index, MODULES], names=["issue", "module"]))
    return Run(settings, diagnostic, issues, agents)


def recipe(preset, settings):
    return {"version": 1, "preset": "django-e1", "inputs_sha256": preset.digest,
            "corpus_revision": preset.manifest["revision"], "settings": asdict(settings)}


def read_recipe(content, preset):
    saved = json.loads(content)
    if not isinstance(saved, dict) or not isinstance(saved.get("settings"), dict):
        raise ValueError("Expected a lab recipe containing a settings object")
    settings = Settings(**saved["settings"])
    settings.condition()
    if saved != recipe(preset, settings):
        raise ValueError("Recipe version, preset or input digest differs from this checkout")
    return settings


def export(preset, run, polling):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        def add(name, content):
            archive.writestr(ZipInfo(name), content, compress_type=ZIP_DEFLATED)
        add("recipe.json", json.dumps(recipe(preset, run.settings), indent=2))
        add("paired-summary.csv", compare(run.issues, polling.issues).totals.to_csv())
        for name, result in [("selected", run), ("polling", polling)]:
            add(f"{name}/issues.csv", result.issues.to_csv())
            add(f"{name}/agents.csv", result.agents.to_csv())
            add(f"{name}/solver.json", json.dumps(result.diagnostic, indent=2))
    return buffer.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, help="Extract recipe.json from a lab export")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preset = load_preset()
    settings = read_recipe(args.recipe.read_bytes(), preset) if args.recipe else Settings()
    selected = replay(preset, settings)
    polling = selected if settings == settings.polling() else replay(preset, settings.polling())
    args.output.write_bytes(export(preset, selected, polling))
