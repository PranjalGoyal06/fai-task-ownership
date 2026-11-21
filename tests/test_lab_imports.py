from src.lab.imports import import_graph
from src.lab.preset import load_preset


def test_complete_graph_with_reciprocal_citations():
    preset = load_preset()
    snapshot = next(iter(preset.snapshots.values()))
    modules = preset.manifest["modules"]
    graph = import_graph(modules, snapshot, "django/django", modules[:2], modules[1:3])
    assert [n["name"] for n in graph["nodes"]] == modules
    assert [n["name"] for n in graph["nodes"] if n["assigned"]] == modules[:2]
    assert [n["name"] for n in graph["nodes"] if n["owner"]] == modules[1:3]
    pairs = graph["connections"]
    assert len(pairs) == 32
    assert sum(len(pair) == 2 for pair in pairs) == 14
    expected = {(a, b): f"https://github.com/django/django/blob/{snapshot['base']}/{path}#L{line}"
                for a, b, path, line in snapshot["edges"]}
    assert len(expected) == 46
    assert {(c["source"], c["target"]): c["url"] for pair in pairs for c in pair} == expected
    for pair in pairs:
        if len(pair) == 2:
            assert (pair[0]["source"], pair[0]["target"]) == (pair[1]["target"], pair[1]["source"])


def test_changed_snapshot_keeps_isolated_modules():
    graph = import_graph(["a", "b", "c"], {"base": "abc", "edges": [(2, 1, "pkg/c.py", 9)]}, "org/repo")
    assert [n["name"] for n in graph["nodes"]] == ["a", "b", "c"]
    assert len(graph["connections"]) == 1
    assert graph["connections"][0][0]["url"] == "https://github.com/org/repo/blob/abc/pkg/c.py#L9"
