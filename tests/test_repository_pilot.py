"""Contract tests for resumable intake, historical labels and configurable agent counts."""
import json
from dataclasses import replace
from zipfile import ZipFile
import io

import numpy as np
import pandas as pd
import pytest

from src.e1.game import select
from src.e1.replay import settle
from src.lab import training
from src.lab.intake import prepare_scan, scan_months
from src.lab.ownership import Ownership, label_cases, partition
from src.lab.preset import load_prepared
from src.lab.replay import Settings, export, read_recipe, replay


def test_monthly_resume_after_quota(tmp_path):
    fetched = []

    def limited(month):
        if month == 3:
            raise RuntimeError("quota")
        fetched.append(month)
        return [{"number": month}]

    with pytest.raises(RuntimeError, match="quota"):
        scan_months(tmp_path, lambda _: None, limited)
    assert fetched == [1, 2]
    assert not (tmp_path / "scan.json").exists()
    retried = []

    def remaining(month):
        retried.append(month)
        return [{"number": month}]

    rows = scan_months(tmp_path, lambda _: None, remaining)
    assert retried == list(range(3, 13))
    assert len(rows) == 12
    with pytest.raises(ValueError, match="pilot supports"):
        prepare_scan("https://github.com/other/repo", tmp_path)


def groups():
    return [{"name": "Storage", "paths": ["sklearn/store"]},
            {"name": "Cache", "paths": ["sklearn/store/cache", "sklearn/store/io.py"]},
            {"name": "Unused", "paths": ["sklearn/unused"]}]


def case(i, day, path="sklearn/store/code.py"):
    return dict(id=f"sklearn-{i}", issue=i, pr=i, merge=str(i), base=f"parent{i}",
                opened=f"2022-01-{day:02}T08:00:00Z", merged=f"2022-01-{day:02}T12:00:00Z",
                title=f"Store read error {i}", text=f"Different report for storage error {i}",
                paths=[path], compiled=[])


def test_rules_exclusions_and_boundaries():
    owners = Ownership(groups())
    assert owners.owner("sklearn/store/code.py") == 0
    assert owners.owner("sklearn/store/cache/code.py") == owners.owner("sklearn/store/io.py") == 1
    assert owners.owner("sklearn/storehouse/code.py") is None
    with pytest.raises(ValueError, match="Conflicting"):
        Ownership(groups() + [dict(name="Overlap", paths=["sklearn/store"])])
    rows = [case(i, i) for i in range(1, 9)]
    rows[0]["paths"].append("sklearn/other.py")
    rows[2]["text"] = rows[1]["text"].upper()
    rows[4]["merge"] = rows[3]["merge"]
    rows[5]["merged"] = "2022-01-07T00:00:00Z"
    retained, excluded = label_cases({"cases": rows}, owners)
    assert len(excluded) == 5 and "unmapped" in excluded["sklearn-1"]
    kept, late = partition(retained, ["2022-01-07", "2022-01-08"])
    assert [c["split"] for c in kept] == ["validation", "test"]
    assert list(late) == ["sklearn-6"]


@pytest.mark.parametrize("count", [2, 6, 9])
def test_settlement_counts_last_agent(count):
    truth = np.zeros((1, count), dtype=int)
    truth[0, -1] = 1
    chosen = select(truth.astype(float), 8)
    row = settle(["fix"], truth, chosen, chosen, 8).iloc[0]
    assert row.owners == 1 << (count - 1)
    assert row.coverage == 1 and row.padding == 0 and row.messages == count


def test_training_cache_sparse_and_replay(tmp_path, monkeypatch):
    class Source:
        def __init__(self, repository):
            pass

        def close(self):
            pass

    calls = []

    def evidence(source, case, ownership):
        calls.append(case["id"])
        return dict(features=[[.1 * i, 0., 0., 0.] for i in range(len(ownership.names))],
                    base=case["base"], citations={}, edges=[])

    monkeypatch.setattr(training, "Snapshot", Source)
    monkeypatch.setattr(training, "extract_one", evidence)
    cases = [case(i, i, "sklearn/store/cache/sub.py" if i % 2 else "sklearn/store/sub.py")
             for i in range(1, 13)]
    scan = {"revision": "abc", "counts": {}, "excluded": {}, "cases": cases}
    cuts = ["2022-01-07", "2022-01-10"]
    path = training.train(scan, groups(), cuts, tmp_path, lambda _: None)
    preset = load_prepared(path)
    assert len(calls) == len(cases)
    assert training.train(scan, groups(), cuts, tmp_path, lambda _: None) == path
    assert len(calls) == len(cases)
    sparse = preset.manifest["model_metrics"][-1]
    assert sparse["predictor"] == "smoothed prior" and sparse["calibration"] == "unavailable"
    np.testing.assert_allclose(preset.beliefs[:, -1], 1 / 8)
    settings = Settings(strategic_agents=2)
    selected, polling = replay(preset, settings), replay(preset, settings.polling())
    altered = replay(replace(preset, outcomes=np.roll(preset.outcomes, 1, axis=1)), settings)
    pd.testing.assert_series_equal(selected.agents.assigned, altered.agents.assigned)
    data = export(preset, selected, polling)
    with ZipFile(io.BytesIO(data)) as archive:
        restored_path = tmp_path / "exported.json.gz"
        restored_path.write_bytes(archive.read("inputs.json.gz"))
        restored = load_prepared(restored_path)
        config = read_recipe(archive.read("recipe.json"), restored)
        assert json.loads(archive.read("recipe.json"))["preset"] == "sklearn-2022"
    assert export(restored, replay(restored, config), replay(restored, config.polling())) == data
    assert len(calls) == len(cases)


def test_rule_must_own_a_real_historical_path():
    from src.lab.ownership import rule_counts, verify_rules
    first = ["sklearn/base.py", "sklearn/utils/core.py"]
    history = first + ["sklearn/later.py"]
    mapping = Ownership([dict(name="Rest", paths=["sklearn"]), dict(name="New", paths=["sklearn/later.py"]),
                         dict(name="Utilities", paths=["sklearn/utlis"])])
    rows = rule_counts(mapping, first, history)
    assert rows[1]["First snapshot"] == 0 and rows[1]["Across snapshots"] == 1
    with pytest.raises(ValueError, match="sklearn/utlis"):
        verify_rules(rows)
    verify_rules(rows[:2])
