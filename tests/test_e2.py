import json
import io
import urllib.error
from email.message import Message
from unittest.mock import patch

import pytest
import pandas as pd
import numpy as np

from src.e1.dataset import MODULES
from src.e1.models import Cases
from src.e2.evidence import validate_findings
from src.e2.models import paired_predictors, select_cohort
from src.e2.providers import GeminiProvider, OllamaProvider, provider_from_environment
from src.e2.replay import cached_evidence_provenance
import src.e2.direct as direct_module
from src.e2.direct import (
    DIRECT_MANIFEST_VERSION,
    base_rates,
    complete_belief_issues,
    compare_modes,
    direct_prompt,
    replay as replay_direct,
    validate_direct_response,
    run as run_direct,
)


def test_provider_requires_declared_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("E2_LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY|GEMINI_API_KEY"):
        provider_from_environment()


def test_unknown_explicit_provider_does_not_fall_back_to_hosted_key(monkeypatch):
    monkeypatch.setenv("E2_LLM_PROVIDER", "ollma")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    with pytest.raises(ValueError, match="E2_LLM_PROVIDER"):
        provider_from_environment()


class DummyToolbox:
    module_id = 0

    def read(self, path, start, end):
        return {
            "snippet": "real source text",
            "sha256": "fixture-digest",
        }


def test_cross_module_citation_is_rejected():
    findings = [
        {
            "claim": "This function is relevant",
            "evidence_type": "symbol_match",
            "citations": [
                {
                    "path": "django/forms/fields.py",
                    "start_line": 10,
                    "end_line": 12,
                    "snippet": "real source text",
                }
            ],
        }
    ]
    accepted, rejected = validate_findings(findings, DummyToolbox())
    assert accepted == []
    assert rejected[0]["reason"] == "invalid_or_cross_module_citation"


def test_finding_cannot_cite_source_not_retrieved():
    finding = {
        "claim": "This function is relevant",
        "evidence_type": "symbol_match",
        "citations": [
            {
                "path": "django/db/models/fields.py",
                "start_line": 10,
                "end_line": 12,
                "snippet": "real source text",
            }
        ],
    }
    accepted, rejected = validate_findings(
        [finding],
        DummyToolbox(),
        allowed_citations=[],
    )
    assert accepted == []
    assert rejected[0]["reason"] == "invalid_or_cross_module_citation"


def test_module_local_citation_is_accepted():
    finding = {
        "claim": "This function encrypts all database values",
        "evidence_type": "trace_match",
        "citations": [
            {
                "path": "django/db/models/fields.py",
                "start_line": 10,
                "end_line": 12,
                "snippet": "real source text",
            }
        ],
    }
    accepted, rejected = validate_findings([finding], DummyToolbox())
    assert len(accepted) == 1
    assert rejected == []
    assert accepted[0]["citations"][0]["sha256"] == "fixture-digest"
    assert accepted[0]["claim_support"] == "not_verified"


def test_cached_evidence_provenance_uses_records_not_a_fixed_model(tmp_path):
    issue_cache = tmp_path / "e2_evidence" / "issue"
    issue_cache.mkdir(parents=True)
    for module_id in range(len(MODULES)):
        record = (
            {
                "provider": "OllamaProvider",
                "model": "qwen3:4b",
                "endpoint": "http://localhost:11434/v1",
                "model_cutoff": "unknown",
            }
            if module_id == 0
            else {"provider": "gemini-3.5-flash-lite"}
        )
        (issue_cache / f"{module_id}.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )

    provenance = cached_evidence_provenance(tmp_path, ["issue"])

    assert provenance["records"] == len(MODULES)
    assert provenance["providers"] == ["OllamaProvider", "unspecified"]
    assert provenance["models"] == ["gemini-3.5-flash-lite", "qwen3:4b"]


def test_gemini_retries_transient_server_errors():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"ok": true}'}}]}
            ).encode()

    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "Unavailable",
                Message(),
                io.BytesIO(b""),
            )
        return Response()

    with patch("src.e2.providers.urllib.request.urlopen", fake_urlopen):
        with patch("src.e2.providers.time.sleep") as sleep:
            result = GeminiProvider(api_key="not-a-real-key", model="fixture").complete(
                "system",
                "prompt",
            )

    assert json.loads(result) == {"ok": True}
    assert len(calls) == 2
    sleep.assert_called_once_with(1)


def test_ollama_provider_uses_local_structured_chat_endpoint(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {"content": '{"probability": 0.5}'},
                    "prompt_eval_count": 12,
                    "eval_count": 8,
                }
            ).encode()

    monkeypatch.setenv("E2_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("E2_LLM_MODEL", "qwen3:4b")
    provider = provider_from_environment()
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return Response()

    with patch("src.e2.providers.urllib.request.urlopen", fake_urlopen):
        response = provider.complete("system", "prompt")

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen3:4b"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["authorization"] is None
    assert captured["timeout"] == 300
    assert json.loads(response)["probability"] == 0.5
    assert provider.last_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
    }


def test_semantic_features_are_frozen_and_append_only():
    cache_record = {
        "issue_id": "fixture",
        "module_id": 0,
        "findings": [{"claim": "fixture", "citations": []}],
        "rejected": [],
        "semantic_features": [1.0, 1.0, 0.0, 0.0],
        "provider": "fixture",
    }
    encoded = json.dumps(cache_record)
    decoded = json.loads(encoded)
    assert decoded["semantic_features"] == [1.0, 1.0, 0.0, 0.0]


def test_cohort_selection_is_first_n_per_frozen_split():
    issues = [
        {"id": "tr1", "split": "train"},
        {"id": "tr2", "split": "train"},
        {"id": "va1", "split": "validation"},
        {"id": "va2", "split": "validation"},
        {"id": "te1", "split": "test"},
    ]
    selected = select_cohort(issues, per_split=1)
    assert [issue["id"] for issue in selected] == ["tr1", "va1", "te1"]


def test_paired_predictors_use_same_subset_for_both_conditions(tmp_path):
    issues = []
    labels = {}
    for split_name, prefix in (
        ("train", "tr"),
        ("validation", "va"),
        ("test", "te"),
    ):
        for index in range(2):
            issue_id = f"{prefix}{index}"
            issues.append(
                {
                    "id": issue_id,
                    "split": split_name,
                    "text": f"module feature issue {prefix}token{index}",
                }
            )
            labels[issue_id] = {"owners": [index]}

    (tmp_path / "issues.json").write_text(json.dumps(issues), encoding="utf-8")
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (tmp_path / "descriptions.json").write_text(
        json.dumps({"docs": [f"module feature {name}" for name in MODULES]}),
        encoding="utf-8",
    )
    evidence_directory = tmp_path / "evidence"
    semantic_directory = tmp_path / "e2_evidence"
    evidence_directory.mkdir()
    for issue_index, issue in enumerate(issues):
        lexical_row = {
            "features": [
                [float(issue_index + module_id), float(module_id), 0.0, 1.0]
                for module_id in range(len(MODULES))
            ],
            "edges": [],
        }
        (evidence_directory / f"{issue['id']}.json").write_text(
            json.dumps(lexical_row),
            encoding="utf-8",
        )
        issue_directory = semantic_directory / issue["id"]
        issue_directory.mkdir(parents=True)
        for module_id in range(len(MODULES)):
            (issue_directory / f"{module_id}.json").write_text(
                json.dumps(
                    {
                        "semantic_features": [
                            float(issue_index == module_id % 2),
                            0.0,
                            0.0,
                            0.0,
                        ]
                    }
                ),
                encoding="utf-8",
            )

    lexical, semantic, validation_y = paired_predictors(
        tmp_path,
        issues_per_split=2,
    )
    assert lexical.ids == semantic.ids
    assert len(lexical.ids) == 6
    assert lexical.local.shape == semantic.local.shape == (6, len(MODULES))
    assert lexical.joint_masks.shape == semantic.joint_masks.shape == (6, 2)
    assert validation_y.shape == (2, len(MODULES))


def test_direct_belief_parser_accepts_probability_and_rejects_invalid_value():
    valid = validate_direct_response(
        '{"probability": 0.37, "evidence": ["No source supplied"], "citations": []}',
        repository=None,
        base_commit="fixture",
        module_id=0,
        allowed_snippets=[],
    )
    assert valid["probability"] == 0.37
    assert valid["citations"] == []
    assert valid["citation_validation"] == "source_match_only"

    with pytest.raises(ValueError, match="probability"):
        validate_direct_response(
            '{"probability": 1.4, "evidence": [], "citations": []}',
            repository=None,
            base_commit="fixture",
            module_id=0,
            allowed_snippets=[],
        )

    with pytest.raises(ValueError, match="JSON object"):
        validate_direct_response(
            '["not", "an object"]',
            repository=None,
            base_commit="fixture",
            module_id=0,
            allowed_snippets=[],
        )


def test_direct_no_tools_prompt_excludes_patch_and_cits():
    issue = {
        "id": "fixture",
        "text": "A historical issue body",
        "base": "fixture-base",
    }
    system, prompt = direct_prompt(issue, 4, 0.12, "no_tools")
    assert "gold patch" in system
    assert "Module: core" in prompt
    assert "A historical issue body" in prompt
    assert "fixture" not in prompt
    assert "Every citation must be an empty list" in prompt


def test_split_count_helpers_require_complete_module_matrix():
    beliefs = pd.DataFrame(
        [
            {"split": "test", "issue": "i1", "module_id": module_id}
            for module_id in range(9)
        ]
        + [
            {"split": "test", "issue": "i2", "module_id": module_id}
            for module_id in range(8)
        ]
        + [
            {"split": "test", "issue": "i3", "module_id": module_id}
            for module_id in range(9)
        ]
        + [{"split": "test", "issue": "i3", "module_id": 0}]
    )
    assert complete_belief_issues(beliefs, "test") == {"i1"}


def test_direct_base_rates_use_only_training_labels(tmp_path):
    issues = [
        {"id": "train-1", "split": "train"},
        {"id": "train-2", "split": "train"},
        {"id": "validation", "split": "validation"},
    ]
    (tmp_path / "issues.json").write_text(json.dumps(issues), encoding="utf-8")
    (tmp_path / "labels.json").write_text(
        json.dumps(
            {
                "train-1": {"owners": [0]},
                "train-2": {"owners": [0, 1]},
                "validation": {"owners": [1]},
            }
        ),
        encoding="utf-8",
    )

    priors = base_rates(tmp_path)

    assert priors[0] == 1.0
    assert priors[1] == 0.5
    assert np.all(priors[2:] == 0.0)


def test_direct_run_makes_one_cached_record_per_issue_module(tmp_path, monkeypatch):
    class FixtureProvider:
        model = "fixture-direct"

        def __init__(self):
            self.calls = 0

        def complete(self, system, prompt):
            self.calls += 1
            return '{"probability": 0.25, "evidence": ["fixture"], "citations": []}'

    (tmp_path / "issues.json").write_text(
        json.dumps(
            [
                {
                    "id": "training-issue",
                    "split": "train",
                    "base": "fixture-base",
                    "text": "Training issue text",
                },
                {
                    "id": "fixture-issue",
                    "split": "test",
                    "base": "fixture-base",
                    "text": "Fixture issue text",
                },
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "labels.json").write_text(
        json.dumps(
            {
                "training-issue": {"owners": [0]},
                "fixture-issue": {"owners": [4]},
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    provider = FixtureProvider()
    run_direct(
        tmp_path,
        output,
        splits=("test",),
        mode="no_tools",
        provider=provider,
    )

    beliefs = pd.read_csv(output / "e2d-beliefs.csv")
    assert len(beliefs) == len(MODULES)
    assert set(beliefs["probability"]) == {0.25}
    cache_files = list((output / "e2d" / "beliefs").glob("*.json"))
    assert len(cache_files) == len(MODULES)
    assert json.loads(cache_files[0].read_text(encoding="utf-8"))["raw_response"]

    run_direct(
        tmp_path,
        output,
        splits=("test",),
        mode="no_tools",
        provider=provider,
    )
    assert provider.calls == len(MODULES)
    manifest = json.loads(
        (output / "e2d-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["cache_hits"] == len(MODULES)
    assert manifest["inference_seconds"] == 0.0
    assert manifest["issue_ids"] == ["fixture-issue"]

    monkeypatch.setattr(direct_module, "local_snippets", lambda *_args: [])
    run_direct(
        tmp_path,
        output,
        repository="fixture-repository",
        splits=("test",),
        mode="snippets",
        provider=provider,
    )
    cache_files = list((output / "e2d" / "beliefs").glob("*.json"))
    assert len(cache_files) == 2 * len(MODULES)
    assert {path.stem.rsplit("__", 1)[-1] for path in cache_files} == {
        "no_tools",
        "snippets",
    }

def test_direct_run_limits_issues_per_selected_split(tmp_path):
    class FixtureProvider:
        model = "fixture-direct"

        def __init__(self):
            self.calls = 0

        def complete(self, system, prompt):
            self.calls += 1
            return '{"probability": 0.25, "evidence": [], "citations": []}'

    issues = [
        {
            "id": issue_id,
            "split": split,
            "base": "fixture-base",
            "text": "Fixture issue text",
        }
        for split, issue_id in (
            ("train", "training-1"),
            ("validation", "validation-1"),
            ("validation", "validation-2"),
            ("test", "test-1"),
            ("test", "test-2"),
        )
    ]
    (tmp_path / "issues.json").write_text(json.dumps(issues), encoding="utf-8")
    (tmp_path / "labels.json").write_text(
        json.dumps({issue["id"]: {"owners": [4]} for issue in issues}),
        encoding="utf-8",
    )
    provider = FixtureProvider()

    output = tmp_path / "output"
    run_direct(
        tmp_path,
        output,
        splits=("validation", "test"),
        max_issues=1,
        provider=provider,
    )

    beliefs = pd.read_csv(output / "e2d-beliefs.csv")
    assert beliefs.groupby("split")["issue"].nunique().to_dict() == {
        "test": 1,
        "validation": 1,
    }
    assert provider.calls == 2 * len(MODULES)
    manifest = json.loads(
        (output / "e2d-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["issue_ids"] == ["validation-1", "test-1"]


def test_direct_replay_uses_only_complete_test_beliefs_without_validation(tmp_path, monkeypatch):
    issue_id = "test-issue"
    cases = Cases(
        ids=[issue_id],
        split=np.array(["test"]),
        local=np.full((1, len(MODULES)), 0.1),
        central=np.full((1, len(MODULES)), 0.1),
        descriptions=np.full((1, len(MODULES)), 0.1),
        prior=np.full(len(MODULES), 0.1),
        joint_masks=np.zeros((1, 25), dtype=int),
        joint_weights=np.full((1, 25), 1 / 25),
        degree=np.zeros((1, len(MODULES)), dtype=int),
    )
    monkeypatch.setattr(direct_module, "predictors", lambda _directory: (cases, None))
    monkeypatch.setattr(
        direct_module,
        "comparators",
        lambda _cases, _mask, _reward: {
            "B1": np.array([1]),
            "B3": np.array([1]),
            "B4": np.array([1]),
            "central": np.array([1]),
        },
    )

    (tmp_path / "labels.json").write_text(
        json.dumps({issue_id: {"owners": [0]}}),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    pd.DataFrame(
        [
            {
                "issue": issue_id,
                "split": "test",
                "module": module_name,
                "module_id": module_id,
                "probability": 0.2,
            }
            for module_id, module_name in enumerate(MODULES)
        ]
    ).to_csv(output / "e2d-beliefs.csv", index=False)
    (output / "e2d-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": DIRECT_MANIFEST_VERSION,
                "successful_rows": len(MODULES),
                "issue_ids": [issue_id],
            }
        ),
        encoding="utf-8",
    )

    replay_direct(tmp_path, output)

    diagnostics = json.loads(
        (output / "e2d-replay-diagnostics.json").read_text(encoding="utf-8")
    )
    replay_rows = pd.read_csv(output / "e2d-replay.csv")
    calibration = pd.read_csv(output / "e2d-calibration.csv")
    summary = pd.read_csv(output / "e2d-summary.csv")
    assert any(item["status"] == "mechanism_replay_not_run" for item in diagnostics)
    assert "B4-E1-lexical" in set(replay_rows["config"])
    assert set(replay_rows["regret_reference"]) == {"E1 lexical B4 estimate"}
    assert set(calibration[calibration["split"] == "validation"]["count"]) == {0}
    assert set(calibration[
        (calibration["split"] == "test") & (calibration["module"] == "overall")
    ]["count"]) == {len(MODULES)}
    assert set(summary["efficiency_vs_e1_lexical_b4"].dropna()).issubset({0.0, 1.0})


def test_direct_replay_pairs_lexical_controls_once_and_includes_audit_only(
    tmp_path, monkeypatch
):
    issue_ids = ["validation-issue", "test-issue"]
    cases = Cases(
        ids=issue_ids,
        split=np.array(["validation", "test"]),
        local=np.full((2, len(MODULES)), 0.1),
        central=np.full((2, len(MODULES)), 0.1),
        descriptions=np.full((2, len(MODULES)), 0.1),
        prior=np.full(len(MODULES), 0.1),
        joint_masks=np.zeros((2, 25), dtype=int),
        joint_weights=np.full((2, 25), 1 / 25),
        degree=np.zeros((2, len(MODULES)), dtype=int),
    )
    monkeypatch.setattr(direct_module, "predictors", lambda _directory: (cases, None))
    monkeypatch.setattr(
        direct_module,
        "comparators",
        lambda _cases, mask, _reward: {
            "B1": np.ones(int(mask.sum()), dtype=int),
            "B3": np.ones(int(mask.sum()), dtype=int),
            "B4": np.ones(int(mask.sum()), dtype=int),
            "central": np.ones(int(mask.sum()), dtype=int),
        },
    )
    monkeypatch.setattr(
        direct_module,
        "fit_strategy",
        lambda probabilities, _outcomes, _degree, _prior, _condition, _seed: (
            np.zeros((len(MODULES), 4), dtype=int),
            np.full(len(MODULES), 0.1),
            {"status": "converged"},
        ),
    )
    (tmp_path / "labels.json").write_text(
        json.dumps(
            {
                "validation-issue": {"owners": [0]},
                "test-issue": {"owners": [0]},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    beliefs = [
        {
            "issue": issue_id,
            "split": split,
            "module": module_name,
            "module_id": module_id,
            "probability": 0.2,
        }
        for issue_id, split in zip(issue_ids, ("validation", "test"))
        for module_id, module_name in enumerate(MODULES)
    ]
    pd.DataFrame(beliefs).to_csv(output / "e2d-beliefs.csv", index=False)
    (output / "e2d-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": DIRECT_MANIFEST_VERSION,
                "successful_rows": len(beliefs),
                "issue_ids": issue_ids,
            }
        ),
        encoding="utf-8",
    )

    replay_direct(tmp_path, output)

    replay_rows = pd.read_csv(output / "e2d-replay.csv")
    diagnostics = json.loads(
        (output / "e2d-replay-diagnostics.json").read_text(encoding="utf-8")
    )
    summary = pd.read_csv(output / "e2d-paired-summary.csv")
    test_baselines = replay_rows[
        (replay_rows["split"] == "test")
        & replay_rows["config"].isin(["B1", "B3", "B4-E1-lexical", "central"])
    ]
    counts = test_baselines.groupby(["config", "evidence"]).size()
    assert set(counts) == {1}
    assert all(
        replay_rows[
            (replay_rows["split"] == "test")
            & (replay_rows["config"] == "B2-strategic")
        ]["method"]
        == "Strategic B2"
    )
    assert set(
        replay_rows[
            (replay_rows["split"] == "test")
            & (replay_rows["config"] == "B2-honest")
        ]["routing_label"]
    ) == {"B2-direct", "B2-E1-lexical"}
    audit_only = replay_rows[
        (replay_rows["split"] == "test")
        & (replay_rows["ell"].fillna(0) == -1)
        & np.isclose(replay_rows["phi"].fillna(0), 2 / 3)
        & (replay_rows["scale"].fillna(-1) == 0)
        & (replay_rows["audit"].fillna(0) == 0.25)
    ]
    assert set(audit_only["evidence"]) == {"direct_llm", "e1_lexical"}
    assert not any(item["status"] == "mechanism_replay_not_run" for item in diagnostics)
    assert {
        "brier_loss",
        "welfare",
        "coverage",
        "padding",
        "gross_recovery_vs_strategic_b2",
        "net_recovery_vs_strategic_b2",
    }.issubset(set(summary["metric"]))


def test_mode_comparison_requires_same_cohort_and_reports_paired_brier(tmp_path):
    mode_dirs = [tmp_path / "no_tools", tmp_path / "snippets"]
    for mode, directory, probability in (
        ("no_tools", mode_dirs[0], 0.2),
        ("snippets", mode_dirs[1], 0.8),
    ):
        directory.mkdir()
        predictions = [
            {
                "issue": "test-issue",
                "split": "test",
                "module_id": module_id,
                "probability": probability,
                "outcome": int(module_id == 0),
            }
            for module_id in range(len(MODULES))
        ]
        pd.DataFrame(predictions).to_csv(
            directory / "e2d-predictions.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "issue": "test-issue",
                    "split": "test",
                    "config": "B2-honest",
                    "method": "Honest routing",
                    "evidence": "direct_llm",
                    "welfare": probability,
                    "coverage": probability,
                    "padding": 0,
                    "any_padding": False,
                }
            ]
        ).to_csv(directory / "e2d-replay.csv", index=False)
        (directory / "e2d-manifest.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "provider": "OllamaProvider",
                    "model": "qwen3:4b",
                    "issue_ids": ["test-issue"],
                }
            ),
            encoding="utf-8",
        )

    result_path = compare_modes(mode_dirs[1], mode_dirs[0])

    results = pd.read_csv(result_path)
    brier = results[results["metric"] == "brier_loss"].iloc[0]
    assert brier["count"] == 1
    assert brier["mode_a"] == "snippets"
    assert brier["mode_b"] == "no_tools"
    assert brier["mode_a_minus_mode_b"] > 0


def test_mode_comparison_rejects_mismatched_issue_cohorts(tmp_path):
    paths = [tmp_path / "a", tmp_path / "b"]
    for mode, path, issue_id in (
        ("no_tools", paths[0], "issue-1"),
        ("snippets", paths[1], "issue-2"),
    ):
        path.mkdir()
        (path / "e2d-manifest.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "provider": "OllamaProvider",
                    "model": "qwen3:4b",
                    "issue_ids": [issue_id],
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="same selected issue IDs"):
        compare_modes(paths[0], paths[1])
