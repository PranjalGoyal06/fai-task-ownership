import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.lab.preset import load_prepared
from src.e2.research_grid import (
    run_condition_metrics,
    sweep_rq1,
    sweep_rq2,
    sweep_rq3,
    generate_methods_benchmark,
    export_full_research_suite,
)
from src.lab.replay import Settings


@pytest.fixture
def django_preset():
    path = Path("results/repositories/django/django/direct/1959fde65a655179b49d2c667e6a370e3d823213e2da11ac3a82bdcc2ff3dff7/inputs.json.gz")
    if not path.exists():
        pytest.skip(f"Test requires E2 preset at {path}")
    return load_prepared(path)


def test_run_condition_metrics_reproducibility(django_preset):
    s = Settings(private_cost=-1.0, strategic_agents=6, scoring=8.0, audit=0.25, seed=331)
    m1 = run_condition_metrics(django_preset, s)
    m2 = run_condition_metrics(django_preset, s)
    assert m1["welfare"] == m2["welfare"]
    assert m1["miss_loss"] == m2["miss_loss"]
    assert m1["excess_cost"] == m2["excess_cost"]
    assert m1["padding"] == m2["padding"]
    assert m1["converged"] is True


def test_sweep_rq1_schema_and_monotonicity(django_preset):
    df = sweep_rq1(django_preset, seed=331)
    assert isinstance(df, pd.DataFrame)
    assert "private_cost_ell" in df.columns
    assert "strategic_fraction_phi" in df.columns
    assert "excess_cost" in df.columns
    assert "miss_loss" in df.columns
    assert len(df) == 7 * 4  # 7 cost levels x 4 phi levels


def test_sweep_rq2_scoring_suppresses_padding(django_preset):
    df = sweep_rq2(django_preset, seed=331)
    assert isinstance(df, pd.DataFrame)
    assert "scoring_lambda" in df.columns
    assert "audit_rate_alpha" in df.columns
    
    # In acquisition (ell = -1), verify scoring lambda=8 reduces padding vs lambda=0
    acq = df[(df["regime"].str.startswith("Acquisition")) & (df["audit_rate_alpha"] == 0.0)]
    pad_l0 = acq[acq["scoring_lambda"] == 0.0]["padding"].values[0]
    pad_l8 = acq[acq["scoring_lambda"] == 8.0]["padding"].values[0]
    assert pad_l8 < pad_l0, f"Expected scoring to reduce padding: {pad_l8} vs {pad_l0}"


def test_export_full_research_suite(tmp_path, django_preset):
    outputs = export_full_research_suite(django_preset, output_dir=tmp_path)
    for key, path in outputs.items():
        assert path.exists(), f"Missing output file for {key}: {path}"
        assert path.stat().st_size > 0
