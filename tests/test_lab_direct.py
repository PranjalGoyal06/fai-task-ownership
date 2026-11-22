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


def test_direct_response_rejects_unprovided_citations():
    allowed = [{"path": "pkg/a.py", "start_line": 1, "end_line": 2, "snippet": "one\ntwo"}]
    value = direct.validate(json.dumps({"probability": .5, "citations": [{"path": "pkg/b.py", "start_line": 1,
                                                                              "end_line": 2, "snippet": "one\ntwo"}]}), allowed)
    assert value["probability"] == .5
    assert not value["citations"]
