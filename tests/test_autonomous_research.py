import json

from pathmnist.autonomous_research import prepare_research
from pathmnist.autonomous_stages import V2_STAGES


def test_research_writes_only_relevance_accepted_references(tmp_path):
    task_root = tmp_path / "task"
    (task_root / "dataset").mkdir(parents=True)
    stages = {stage: "waiting" for stage in V2_STAGES}
    for stage in V2_STAGES[: V2_STAGES.index("dataset_validated") + 1]:
        stages[stage] = "completed"
    (task_root / "task.json").write_text(json.dumps({
        "schema_version": 2, "task_id": "task", "research_direction": "PathMNIST robust classification",
        "stages": stages, "completed_stage": "dataset_validated",
    }), encoding="utf-8")
    (task_root / "dataset/dataset_profile.json").write_text(json.dumps({
        "content_sha256": "a" * 64, "recommended_metrics": ["macro_f1", "accuracy"]
    }), encoding="utf-8")

    def search(_query):
        return [
            {"title": "PathMNIST histopathology image classification", "authors": "A", "year": 2024, "venue": "SPIE", "doi": "10.1/path"},
            {"title": "Decision letter for a pathology paper", "authors": "Editor", "year": 2025, "doi": "10.1/x/decision1"},
            {"title": "Dynamic routing for remote sensing images", "authors": "B", "year": 2024, "venue": "Remote Sensing", "doi": "10.1/remote"},
        ]

    result = prepare_research(tmp_path, "task", search=search)
    literature = json.loads((task_root / "research/literature.json").read_text(encoding="utf-8"))
    assert result["verified_references"] == 1
    assert literature["schema_version"] == 2
    assert [item["title"] for item in literature["references"]] == ["PathMNIST histopathology image classification"]
    assert literature["references"][0]["relevance_status"] == "directly_relevant"
    assert result["completed_stage"] == "research_contract_generated"
    contract = json.loads((task_root / "research/research_contract.json").read_text(encoding="utf-8"))
    assert contract["resource_plan"]["api_hard_limit_usd"] == 8.0
    assert {item["rejection_reason"] for item in literature["relevance_report"]["rejected"]} >= {
        "non_article_record", "off_topic_domain"
    }
