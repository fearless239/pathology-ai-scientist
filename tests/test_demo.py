import json
from pathlib import Path

from pathmnist.autonomous_stages import V2_STAGES
from pathmnist.demo import build_demo


def test_offline_demo_is_complete_zero_cost_and_deterministic(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_demo(first)
    build_demo(second)

    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    timeline = json.loads((first / "timeline.json").read_text(encoding="utf-8"))
    acceptance = json.loads((first / "acceptance_report.json").read_text(encoding="utf-8"))

    assert first_manifest["artifacts"] == second_manifest["artifacts"]
    assert [item["stage"] for item in timeline["stages"]] == list(V2_STAGES)
    assert timeline["network_used"] is False
    assert timeline["paid_calls"] == 0
    assert acceptance["passed"] is True


def test_demo_metrics_cannot_be_mistaken_for_scientific_results(tmp_path: Path):
    build_demo(tmp_path)
    metrics = json.loads((tmp_path / "trusted_metrics.json").read_text(encoding="utf-8"))
    assert metrics["fixture_only"] is True
    assert "not a scientific result" in metrics["claim"]


def test_compose_demo_enforces_runtime_security_boundary():
    compose = (Path(__file__).parents[1] / "compose.yaml").read_text(encoding="utf-8")
    assert "read_only: true" in compose
    assert "no-new-privileges:true" in compose
    assert "cap_drop:" in compose
