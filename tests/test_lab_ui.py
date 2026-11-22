import json

from streamlit.testing.v1 import AppTest

from src.lab.preset import ROOT, load_preset
from src.lab.replay import Settings, recipe


def click(page, label):
    next(w for w in page.button if w.label == label).click().run()
    assert not page.exception


def test_guided_replay_and_fix_inspection():
    page = AppTest.from_file(str(ROOT / "lab.py"), default_timeout=30).run()
    assert not page.exception
    assert not page.number_input
    assert page.metric[0].value == "792 → 610"
    story = " ".join(item.value for item in page.markdown)
    for fragment in ("114 fixes", "110", "27 multi-module fixes", "9 to 6"):
        assert fragment in story
    click(page, "Inspect a fix")
    assert page.radio[0].value == "Newly covered"
    assert "Polling **missed**" in " ".join(item.value for item in page.markdown)
    page.radio[0].set_value("Newly missed").run()
    assert "Polling **covered**" in " ".join(item.value for item in page.markdown)
    assert not page.exception


def test_change_a_django_run():
    page = AppTest.from_file(str(ROOT / "lab.py"), default_timeout=30).run()
    click(page, "Change the experiment")
    page.radio[0].set_value("Multi-module").run()
    assert "27 issues" in " ".join(p.value for p in page.markdown)
    score = next(w for w in page.number_input if w.label == "Scoring strength")
    score.set_value(0.)
    click(page, "Run replay")
    assert page.session_state["historical_run"].scoring == 0.
    click(page, "Inspect a fix")
    assert page.radio[0].options == ["All fixes"]
    assert len(page.selectbox[0].options) == 27


def test_restore_recipe_and_show_nonconvergence_warning():
    page = AppTest.from_file(str(ROOT / "lab.py"), default_timeout=30).run()
    click(page, "Change the experiment")
    settings = Settings(scoring=2., audit=.25, credit="shapley")
    content = json.dumps(recipe(load_preset(), settings)).encode()
    page.file_uploader[0].set_value(("recipe.json", content, "application/json")).run()
    click(page, "Apply recipe")
    assert page.session_state["historical_run"] == settings
    assert next(w for w in page.number_input if w.label == "Scoring strength").value == 2.
    assert page.warning
    click(page, "Inspect a fix")
    assert page.warning


def test_new_repository_delete_and_reset_owner_row():
    scan_paths = list((ROOT / "results" / "repositories").glob("**/scan.json"))
    if not scan_paths:
        return
    scan = json.loads(scan_paths[0].read_bytes())
    page = AppTest.from_file(str(ROOT / "app" / "lab" / "setup.py"), default_timeout=30)
    page.session_state["setup_scan"] = scan
    page.session_state["setup_stage"] = "ownership"
    page.run()
    assert not page.exception
    initial_count = len(page.data_editor[0].value)
    assert len(page.selectbox[0].options) == initial_count

    # Delete the selected owner row
    click(page, "Delete row")
    assert len(page.data_editor[0].value) == initial_count - 1
    assert len(page.selectbox[0].options) == initial_count - 1

    # Reset back to suggested
    click(page, "Reset suggested")
    assert len(page.data_editor[0].value) == initial_count
    assert len(page.selectbox[0].options) == initial_count


def test_new_repository_direct_source_length_filter_and_size_selection():
    scan_paths = list((ROOT / "results" / "repositories").glob("**/scan.json"))
    if not scan_paths:
        return
    scan = json.loads(scan_paths[0].read_bytes())
    page = AppTest.from_file(str(ROOT / "app" / "lab" / "setup.py"), default_timeout=30)
    page.session_state["setup_scan"] = scan
    page.session_state["setup_groups"] = [
        {"name": "utils", "paths": [scan["roots"][0] + "/utils"]},
        {"name": "metrics", "paths": [scan["roots"][0] + "/metrics"]},
    ]
    page.session_state["setup_cutoffs"] = ("2025-08-08", "2025-09-19")
    page.session_state["setup_stage"] = "source"
    page.run()
    assert not page.exception

    # Select Direct model probabilities
    page.radio[0].set_value("Direct model probabilities").run()
    assert not page.exception

    # Verify length and selection number inputs are present
    length_input = next(w for w in page.number_input if "Max issue length" in w.label)
    size_input = next(w for w in page.number_input if "Number of issues to evaluate" in w.label)
    assert length_input.value == 5000
    assert size_input.value >= 1

    # Modify size and continue
    target_size = int(min(3, size_input.max))
    size_input.set_value(target_size).run()
    click(page, "Continue")
    assert page.session_state["setup_stage"] == "direct"
    assert page.session_state["setup_direct_size"] == target_size
    assert page.session_state["setup_max_issue_length"] == 5000


