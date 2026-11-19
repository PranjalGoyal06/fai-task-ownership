import json

from streamlit.testing.v1 import AppTest

from src.lab.preset import ROOT, load_preset
from src.lab.replay import Settings, recipe


def test_review_and_change_a_django_run():
    page = AppTest.from_file(str(ROOT / "lab.py"), default_timeout=30).run()
    assert not page.exception
    assert "234 issues" in " ".join(p.value for p in page.markdown)
    page.radio[0].set_value("Multi-module").run()
    assert "27 issues" in " ".join(p.value for p in page.markdown)
    for stage in ("Evidence", "Assignment", "Settlement", "Import graph"):
        page.radio[1].set_value(stage).run()
        assert not page.exception
    assert len(page.get("graphviz_chart")) == 1
    score = next(w for w in page.number_input if w.label == "Scoring strength")
    score.set_value(3.5)
    next(w for w in page.button if w.label == "Run replay").click().run()
    assert not page.exception
    assert page.session_state["historical_run"].scoring == 3.5
    assert "scoring 3.5" in " ".join(p.value for p in page.caption)


def test_restore_recipe_in_sidebar():
    page = AppTest.from_file(str(ROOT / "lab.py"), default_timeout=30).run()
    settings = Settings(private_cost=1., strategic_agents=4, scoring=3.5, reward=12, seed=107)
    content = json.dumps(recipe(load_preset(), settings)).encode()
    page.file_uploader[0].set_value(("recipe.json", content, "application/json")).run()
    next(w for w in page.button if w.label == "Apply recipe").click().run()
    assert not page.exception
    assert page.session_state["historical_run"] == settings
    assert next(w for w in page.number_input if w.label == "Scoring strength").value == 3.5
    assert next(w for w in page.select_slider if w.label == "Reward for a covered fix").value == 12
