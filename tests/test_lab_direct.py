import json

import numpy as np

from src.lab import direct
from src.lab.preset import load_prepared


class Source:
    def __init__(self, *_args):
        pass

    def files(self, _base, _root):
        return [("one", "pkg/one.py"), ("two", "pkg/two.py")]

    def blob(self, oid):
        return "def repair_widget():\n    return 'widget'\n" if oid == "one" else "def repair_store():\n    return 'store'\n"

    def close(self):
        pass


class Provider:
    model = "test-model"
    base_url = "http://test"

    def __init__(self):
        self.calls = 0

    def complete(self, _system, prompt):
        self.calls += 1
        owner = json.loads(prompt)["owner"]
        return json.dumps({"probability": .75 if owner == "One" else .25, "evidence": ["matching source"], "citations": []})


class RetryProvider(Provider):
    def complete(self, _system, prompt):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("timed out")
        owner = json.loads(prompt)["owner"]
        return json.dumps({"probability": .75 if owner == "One" else .25, "evidence": ["matching source"], "citations": []})


def case(identifier, opened, path):
    number = {"train": 1, "valid": 2, "test": 3}[identifier]
    return {"id": identifier, "issue": number, "pr": number, "base": identifier + "base", "merge": identifier + "merge",
            "opened": opened + "T00:00:00Z", "merged": opened + "T01:00:00Z", "title": "Repair widget",
            "text": f"Repair widget {identifier}", "paths": [path], "compiled": []}


def test_direct_probabilities_are_resumable_and_open_as_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(direct, "FileSnapshot", Source)
    monkeypatch.setattr(direct, "extract_one", lambda *_args: {
        "features": [[0, 0, 0, 0], [0, 0, 0, 0]], "edges": [[0, 1, "pkg/one.py", 1]], "citations": {}})
    scan = {"repository": "owner/project", "revision": "0" * 40, "roots": ["pkg"], "window": ["2022-01-01", "2022-12-31"],
            "cases": [case("train", "2022-01-01", "pkg/one.py"), case("valid", "2022-02-01", "pkg/two.py"),
                      case("test", "2022-03-01", "pkg/one.py")], "excluded": {}, "counts": {}}
    groups = [{"name": "One", "paths": ["pkg/one.py"]}, {"name": "Two", "paths": ["pkg/two.py"]}]
    provider = Provider()
    path = direct.prepare(scan, groups, ["2022-02-01", "2022-03-01"], selected_per_period=1,
                          cache=tmp_path, provider=provider)
    assert provider.calls == 4
    replay = load_prepared(path)
    assert replay.manifest["probability_source"] == "direct_model"
    assert replay.beliefs.shape == (1, 2)
    assert replay.snapshots["test"]["direct_evidence"]["0"]["probability"] == .75
    assert direct.prepare(scan, groups, ["2022-02-01", "2022-03-01"], selected_per_period=1,
                          cache=tmp_path, provider=provider) == path
    assert provider.calls == 4


def test_direct_probabilities_retry_a_transient_call(tmp_path, monkeypatch):
    monkeypatch.setattr(direct, "FileSnapshot", Source)
    monkeypatch.setattr(direct, "extract_one", lambda *_args: {
        "features": [[0, 0, 0, 0], [0, 0, 0, 0]], "edges": [[0, 1, "pkg/one.py", 1]], "citations": {}})
    scan = {"repository": "owner/project", "revision": "0" * 40, "roots": ["pkg"], "window": ["2022-01-01", "2022-12-31"],
            "cases": [case("train", "2022-01-01", "pkg/one.py"), case("valid", "2022-02-01", "pkg/two.py"),
                      case("test", "2022-03-01", "pkg/one.py")], "excluded": {}, "counts": {}}
    groups = [{"name": "One", "paths": ["pkg/one.py"]}, {"name": "Two", "paths": ["pkg/two.py"]}]
    provider = RetryProvider()
    path = direct.prepare(scan, groups, ["2022-02-01", "2022-03-01"], selected_per_period=1,
                          cache=tmp_path, provider=provider)
    assert provider.calls == 5
    replay = load_prepared(path)
    assert replay.beliefs.shape == (1, 2)


def test_direct_response_rejects_unprovided_citations():
    allowed = [{"path": "pkg/a.py", "start_line": 1, "end_line": 2, "snippet": "one\ntwo"}]
    value = direct.validate(json.dumps({"probability": .5, "citations": [{"path": "pkg/b.py", "start_line": 1,
                                                                              "end_line": 2, "snippet": "one\ntwo"}]}), allowed)
    assert value["probability"] == .5
    assert not value["citations"]


def test_direct_probabilities_filter_by_length_and_report_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(direct, "FileSnapshot", Source)
    monkeypatch.setattr(direct, "extract_one", lambda *_args: {
        "features": [[0, 0, 0, 0], [0, 0, 0, 0]], "edges": [[0, 1, "pkg/one.py", 1]], "citations": {}})

    long_case = case("valid", "2022-02-01", "pkg/two.py")
    long_case["id"] = "valid_long"
    long_case["issue"] = 10
    long_case["pr"] = 10
    long_case["merge"] = "valid_long_merge"
    long_case["text"] = "X" * 6000  # Longer than 5000 chars

    scan = {
        "repository": "owner/project",
        "revision": "0" * 40,
        "roots": ["pkg"],
        "window": ["2022-01-01", "2022-12-31"],
        "cases": [
            case("train", "2022-01-01", "pkg/one.py"),
            long_case,
            case("valid", "2022-02-02", "pkg/two.py"),
            case("test", "2022-03-01", "pkg/one.py"),
        ],
        "excluded": {},
        "counts": {},
    }
    groups = [{"name": "One", "paths": ["pkg/one.py"]}, {"name": "Two", "paths": ["pkg/two.py"]}]
    provider = Provider()

    events = []
    path = direct.prepare(
        scan,
        groups,
        ["2022-02-01", "2022-03-01"],
        selected_per_period=1,
        max_issue_length=5000,
        cache=tmp_path,
        on_progress=events.append,
        provider=provider,
    )

    replay = load_prepared(path)
    # The long issue should be excluded
    assert replay.manifest["skipped_length_count"] == 1
    assert "valid_long" in replay.manifest["exclusions"]
    assert replay.manifest["max_issue_length"] == 5000

    # The provider should only be called for valid (1) and test (1) = 2 issues * 2 agents = 4 calls
    assert provider.calls == 4

    # Events were fired for start and agent updates (both calling and done)
    start_events = [e for e in events if e.get("type") == "issue_start"]
    agent_events = [e for e in events if e.get("type") == "agent_update"]
    assert len(start_events) == 2
    assert len(agent_events) == 8  # 4 calling + 4 done
    assert len([e for e in agent_events if e["status"] == "done"]) == 4
    assert len([e for e in agent_events if e["status"] == "calling"]) == 4

