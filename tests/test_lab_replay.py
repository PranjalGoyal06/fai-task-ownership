"""Keep the portable lab aligned with every published Django policy."""
import io
import json
from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.e1.game import Condition
from src.e1.replay import conditions, tag
from src.lab.preset import ROOT, load_preset
from src.lab.replay import Settings, export, read_recipe, replay


@pytest.fixture(scope="module")
def preset():
    return load_preset()


@pytest.fixture(scope="module")
def archived():
    directory = ROOT / "reports/e1"
    rows = pd.read_csv(directory / "replay.csv.gz")
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    return rows[rows.split == "test"], {d["config"]: d for d in diagnostics}


@pytest.mark.parametrize("condition", list(conditions()), ids=tag)
def test_policy_matches_frozen_e1(preset, archived, condition: Condition):
    c = condition
    settings = Settings(private_cost=c.ell, strategic_agents=round(9 * c.phi), scoring=c.scale,
                        audit=c.audit, credit=c.credit, reward=c.reward, detection=c.delta, fine=c.fine)
    result = replay(preset, settings)
    rows, diagnostics = archived
    expected = rows[rows.config == tag(c)].set_index("issue").loc[result.issues.index]
    for name in ("assigned", "coverage", "padding", "welfare", "gap", "miss_loss", "excess_cost",
                 "incentive_cost", "messages", "score", "audits", "credit_paid", "fines"):
        np.testing.assert_allclose(result.issues[name], expected[name], rtol=0, atol=1e-12)
    for field, observed in result.diagnostic.items():
        wanted = diagnostics[tag(c)][field]
        if field == "residual":
            assert observed == pytest.approx(wanted, abs=1e-12)
        else:
            assert observed == wanted


def test_labels_enter_after_assignment(preset):
    settings = Settings(scoring=3.5, audit=.15, strategic_agents=4, seed=72)
    first = replay(preset, settings)
    rotated = replace(preset, outcomes=np.roll(preset.outcomes, 1, axis=1))
    second = replay(rotated, settings)
    cols = ["private_cost", "report", "selection_input", "assigned"]
    pd.testing.assert_frame_equal(first.agents[cols], second.agents[cols])
    assert not first.issues.coverage.equals(second.issues.coverage)
    control = replay(preset, settings.polling())
    pd.testing.assert_series_equal(first.agents.private_cost, control.agents.private_cost)


def test_recipe_restores_identical_export(preset):
    settings = Settings(private_cost=1., strategic_agents=2, scoring=2., seed=497)
    result = replay(preset, settings)
    control = replay(preset, settings.polling())
    data = export(preset, result, control)
    with ZipFile(io.BytesIO(data)) as archive:
        content = archive.read("recipe.json")
        assert len(pd.read_csv(archive.open("selected/agents.csv"))) == 234 * 9
    restored = read_recipe(content, preset)
    assert restored == settings
    assert export(preset, replay(preset, restored), replay(preset, restored.polling())) == data
    changed = json.loads(content)
    changed["inputs_sha256"] = "another preset"
    with pytest.raises(ValueError, match="digest"):
        read_recipe(json.dumps(changed), preset)


def test_release_graph_and_ownership_order(preset):
    assert list(preset.issues.index) == list(preset.issues.index.unique())
    assert preset.manifest["split_counts"] == {"train": 429, "validation": 181, "test": 234}
    for (key, issue), row in zip(preset.issues.iterrows(), preset.outcomes, strict=True):
        snapshot = preset.snapshots[key]
        assert snapshot["base"] == issue.base
        assert len(snapshot["features"]) == len(row)
        assert all(path.startswith("django/") and line > 0 for _, _, path, line in snapshot["edges"])
        names = [name for name, owner in zip(preset.manifest["modules"], row, strict=True) if owner]
        assert names == issue.owners.split(",")
