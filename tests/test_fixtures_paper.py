import jsonschema

from gate_a.config import load_config
from gate_a.paper import PAPER_SCHEMA, REVIEW_SCHEMA, render_latex
from gate_a.pipeline import IDEA_SCHEMA, fixture_models
from gate_a.provider import FixtureProvider


def test_offline_fixtures_satisfy_structured_contracts(project_root):
    config = load_config(project_root / "configs" / "gate_a.yaml")
    provider = FixtureProvider(fixture_models(config))
    idea, _ = provider.call_json("ideation", "x", "", "", "submit", IDEA_SCHEMA)
    paper, _ = provider.call_json("paper_writer", "y", "", "", "submit", PAPER_SCHEMA)
    review, _ = provider.call_json("reviewer", "z", "", "", "submit", REVIEW_SCHEMA)
    jsonschema.validate(idea, IDEA_SCHEMA)
    jsonschema.validate(paper, PAPER_SCHEMA)
    jsonschema.validate(review, REVIEW_SCHEMA)


def test_latex_always_contains_required_disclosure(project_root):
    config = load_config(project_root / "configs" / "gate_a.yaml")
    provider = FixtureProvider(fixture_models(config))
    paper, _ = provider.call_json("paper_writer", "y", "", "", "submit", PAPER_SCHEMA)
    latex = render_latex(paper, "validation_accuracy.png")
    assert "Human review is required" in latex
    assert "machine-generated" in latex
    assert "\\includegraphics" in latex
