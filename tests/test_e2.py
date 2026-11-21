import json
import io
import urllib.error
from email.message import Message
from unittest.mock import patch

import pytest

from src.e1.dataset import MODULES
from src.e2.evidence import validate_findings
from src.e2.models import paired_predictors, select_cohort
from src.e2.providers import GeminiProvider, provider_from_environment


def test_provider_requires_declared_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("E2_LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY|GEMINI_API_KEY"):
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
        "claim": "The local function handles the issue behavior",
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
