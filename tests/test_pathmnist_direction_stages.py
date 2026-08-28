import json
import shutil

import pytest

from pathmnist.literature import LiteratureError
from pathmnist.workflow import (
    WorkflowConfig,
    WorkflowContext,
    WorkflowError,
    WorkflowExecutor,
)


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.text_calls = []

    def call_json(self, role, request_id, system, prompt, function_name, schema):
        self.calls.append(
            {
                "role": role,
                "request_id": request_id,
                "system": system,
                "prompt": prompt,
                "function_name": function_name,
            }
        )
        if request_id not in self.responses:
            raise AssertionError(f"unexpected LLM request: {request_id}")
        return self.responses[request_id], {}

    def call_text(self, role, request_id, system, prompt):
        self.text_calls.append(
            {"role": role, "request_id": request_id, "system": system, "prompt": prompt}
        )
        if request_id not in self.responses:
            raise AssertionError(f"unexpected LLM request: {request_id}")
        return self.responses[request_id], {}


def _make_executor(tmp_path, llm_client=None, literature_search=None):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "SOURCE_PROVENANCE.md").write_text("# provenance\n", encoding="utf-8")
    context = WorkflowContext(
        project_root=tmp_path,
        config_path=tmp_path / "configs/pathmnist_m4.yaml",
        run_root=tmp_path / "runs/pathmnist-m4",
        report_path=tmp_path / "docs/M4_FINAL_REPORT.md",
        candidate_path=tmp_path / "configs/pathmnist_final_candidate.json",
    )
    return WorkflowExecutor(
        context,
        llm_client=llm_client,
        literature_search=literature_search,
    )


def _task_state(task_id="dir-task", direction="", goal=""):
    from pathmnist.workflow import TaskState

    return TaskState(
        task_id=task_id,
        schema_version=1,
        config={
            "research_direction": direction,
            "research_goal": goal,
        },
        stages={},
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-19T00:00:00+00:00",
        completed_stage="",
        control="running",
        spent_usd=0.0,
        reserved_usd=0.0,
        execution_seconds=0.0,
    )


def test_config_parses_and_bounds_research_inputs():
    parsed = WorkflowConfig.from_mapping(
        {
            "mode": "staged_approval",
            "budget_limit_usd": 2.0,
            "execution_limit_seconds": 3600,
            "research_direction": " 多尺度特征与泛化差距 ",
            "research_goal": " 缩小 gap ",
        }
    )
    assert parsed.research_direction == "多尺度特征与泛化差距"
    assert parsed.research_goal == "缩小 gap"
    with pytest.raises(WorkflowError, match="2000"):
        WorkflowConfig.from_mapping(
            {
                "mode": "staged_approval",
                "budget_limit_usd": 2.0,
                "execution_limit_seconds": 3600,
                "research_direction": "x" * 2001,
            }
        )


def test_research_stage_is_direction_and_llm_driven(tmp_path):
    llm = FakeLLMClient(
        {
            "dir-task-research": {
                "objective": "研究多尺度特征对泛化差距的影响",
                "background": "PathMNIST 上的验证-测试差距",
                "key_questions": ["多尺度是否缩小差距？"],
                "constraints": ["仅使用 train/val 调参"],
            }
        }
    )
    executor = _make_executor(tmp_path, llm_client=llm)
    state = _task_state(direction="多尺度特征与泛化差距", goal="缩小 Macro-F1 差距")
    outputs = executor.execute(
        "research_understood", state, tmp_path / "artifacts" / "research_understood"
    )
    assert outputs["research.json"]
    artifact = json.loads(
        (tmp_path / "artifacts/research_understood/research.json").read_text(encoding="utf-8")
    )
    assert artifact["research_direction"] == "多尺度特征与泛化差距"
    assert artifact["understanding_source"] == "llm"
    assert artifact["llm_analysis"]["objective"] == "研究多尺度特征对泛化差距的影响"
    assert llm.calls[0]["role"] == "ideation"
    assert "多尺度特征与泛化差距" in llm.calls[0]["prompt"]
    assert "exactly once after candidate freezing" in llm.calls[0]["prompt"]


def test_research_deterministic_fallback_without_llm(tmp_path):
    executor = _make_executor(tmp_path)
    state = _task_state(direction="数据增强策略")
    executor.execute(
        "research_understood", state, tmp_path / "artifacts" / "research_understood"
    )
    artifact = json.loads(
        (tmp_path / "artifacts/research_understood/research.json").read_text(encoding="utf-8")
    )
    assert artifact["understanding_source"] == "deterministic"
    assert "数据增强策略" in artifact["objective"]


def test_non_ascii_task_ids_get_provider_safe_request_ids(tmp_path):
    import hashlib

    from gate_a.provider import _safe_request_id

    expected_request_id = (
        "task-"
        + hashlib.sha256("测试0819".encode("utf-8")).hexdigest()[:12]
        + "-research"
    )
    llm = FakeLLMClient(
        {
            expected_request_id: {
                "objective": "目标",
                "background": "背景",
                "key_questions": [],
                "constraints": [],
            }
        }
    )
    executor = _make_executor(tmp_path, llm_client=llm)
    state = _task_state(task_id="测试0819", direction="多尺度特征")
    executor.execute(
        "research_understood", state, tmp_path / "artifacts" / "research_understood"
    )
    request_id = llm.calls[0]["request_id"]
    assert request_id.startswith("task-")
    assert request_id.endswith("-research")
    _safe_request_id(request_id)


def test_literature_records_verified_and_failed_queries(tmp_path):
    llm = FakeLLMClient(
        {
            "dir-task-literature-queries": {"queries": ["multiscale pathology", "augmentation"]}
        }
    )
    calls = []

    def fake_search(query):
        calls.append(query)
        if query == "augmentation":
            raise LiteratureError("rate limited")
        return [
            {
                "title": "Multiscale Histopathology Classification",
                "authors": "A, B",
                "venue": "MICCAI",
                "year": 2024,
                "citation_count": 5,
                "url": "https://example.org/1",
                "abstract": "abstract",
            }
        ]

    executor = _make_executor(tmp_path, llm_client=llm, literature_search=fake_search)
    state = _task_state(direction="多尺度特征")
    stage_root = tmp_path / "artifacts" / "literature_collected"
    executor.execute("literature_collected", state, stage_root)
    artifact = json.loads((stage_root / "literature.json").read_text(encoding="utf-8"))
    assert artifact["source"] == "semantic_scholar"
    assert artifact["verification_status"] == "partial"
    assert artifact["references"][0]["status"] == "api_verified"
    assert artifact["failures"] == [{"query": "augmentation", "error": "rate limited"}]
    first_raw = (stage_root / "literature_raw.json").read_text(encoding="utf-8")
    executor.execute("literature_collected", state, stage_root)
    assert calls == ["multiscale pathology", "augmentation"]
    assert (stage_root / "literature_raw.json").read_text(encoding="utf-8") == first_raw


def test_literature_degrades_to_pending_when_all_queries_fail(tmp_path):
    def fake_search(query):
        raise LiteratureError("unavailable")

    executor = _make_executor(tmp_path, literature_search=fake_search)
    state = _task_state(direction="多尺度特征")
    stage_root = tmp_path / "artifacts" / "literature_collected"
    executor.execute("literature_collected", state, stage_root)
    artifact = json.loads((stage_root / "literature.json").read_text(encoding="utf-8"))
    assert artifact["verification_status"] == "pending_manual_verification"
    assert artifact["references"] == []
    assert len(artifact["failures"]) == 3


def test_topic_stage_uses_llm_with_prior_artifacts(tmp_path):
    llm = FakeLLMClient(
        {
            "dir-task-topic": {
                "name": "multiscale_gap_study",
                "title": "Multiscale Features and the Validation-Test Gap on PathMNIST",
                "short_hypothesis": "多尺度特征能缩小验证-测试差距",
                "experiments": "对比 baseline/multiscale/combined，三种子",
                "related_work": "基于两篇已核验文献",
                "risk_factors_and_limitations": "初步工程研究，无临床声明",
            }
        }
    )
    executor = _make_executor(tmp_path, llm_client=llm)
    research_root = tmp_path / "artifacts" / "research_understood"
    literature_root = tmp_path / "artifacts" / "literature_collected"
    research_root.mkdir(parents=True)
    literature_root.mkdir(parents=True)
    (research_root / "research.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "llm_analysis": {"objective": "obj", "key_questions": ["q1"], "constraints": []},
            }
        ),
        encoding="utf-8",
    )
    (literature_root / "literature.json").write_text(
        json.dumps(
            {
                "references": [
                    {"title": "Paper One", "authors": "A", "venue": "V", "year": 2024}
                ]
            }
        ),
        encoding="utf-8",
    )
    state = _task_state(direction="多尺度特征", goal="缩小差距")
    stage_root = tmp_path / "artifacts" / "topic_proposed"
    executor.execute("topic_proposed", state, stage_root)
    artifact = json.loads((stage_root / "topic.json").read_text(encoding="utf-8"))
    assert artifact["proposer"] == "llm"
    assert artifact["title"].startswith("Multiscale Features")
    assert artifact["research_direction"] == "多尺度特征"
    prompt = llm.calls[0]["prompt"]
    assert "多尺度特征" in prompt
    assert "Paper One" in prompt
    assert "FinalizeIdea" in llm.calls[0]["function_name"]


def test_topic_deterministic_fallback_without_llm(tmp_path):
    executor = _make_executor(tmp_path)
    state = _task_state(direction="标签平滑与优化器")
    stage_root = tmp_path / "artifacts" / "topic_proposed"
    executor.execute("topic_proposed", state, stage_root)
    artifact = json.loads((stage_root / "topic.json").read_text(encoding="utf-8"))
    assert artifact["proposer"] == "deterministic"
    assert "标签平滑与优化器" in artifact["title"]


def test_experiment_plan_maps_direction_to_supported_variants(project_root, tmp_path):
    config_root = tmp_path / "configs"
    config_root.mkdir()
    shutil.copyfile(
        project_root / "configs" / "pathmnist_m4.yaml",
        config_root / "pathmnist_m4.yaml",
    )
    executor = _make_executor(tmp_path)
    state = _task_state(direction="研究染色增强和多尺度形态特征")
    stage_root = tmp_path / "artifacts" / "experiment_planned"

    executor.execute("experiment_planned", state, stage_root)

    plan = json.loads((stage_root / "experiment_plan.json").read_text(encoding="utf-8"))
    assert plan["direction_selected_variants"] == [
        "augmentation",
        "multiscale",
        "combined",
    ]
    assert plan["primary_comparisons"] == [
        "baseline_vs_augmentation",
        "baseline_vs_multiscale",
        "baseline_vs_combined",
    ]
    assert plan["selection_source"] == "research_direction_capability_mapping"


def test_resolution_direction_maps_to_multiscale(project_root, tmp_path):
    config_root = tmp_path / "configs"
    config_root.mkdir()
    shutil.copyfile(
        project_root / "configs" / "pathmnist_m4.yaml",
        config_root / "pathmnist_m4.yaml",
    )
    executor = _make_executor(tmp_path)
    state = _task_state(direction="研究 PathMNIST 不同图像分辨率的影响")
    stage_root = tmp_path / "artifacts" / "experiment_planned"

    executor.execute("experiment_planned", state, stage_root)

    plan = json.loads((stage_root / "experiment_plan.json").read_text(encoding="utf-8"))
    assert plan["direction_selected_variants"] == ["multiscale"]
    assert plan["primary_comparisons"] == ["baseline_vs_multiscale"]


def test_paper_prompt_consumes_topic_and_literature(tmp_path):
    llm = FakeLLMClient({"dir-task-paper": "paper body"})
    artifacts = tmp_path / "artifacts"
    topic_root = artifacts / "topic_proposed"
    literature_root = artifacts / "literature_collected"
    topic_root.mkdir(parents=True)
    literature_root.mkdir(parents=True)
    (topic_root / "topic.json").write_text(
        json.dumps(
            {
                "title": "Directed Topic Title",
                "short_hypothesis": "hypothesis",
                "experiments": "experiments",
                "risk_factors_and_limitations": "risks",
            }
        ),
        encoding="utf-8",
    )
    (literature_root / "literature.json").write_text(
        json.dumps(
            {
                "references": [
                    {"title": "Paper One", "authors": "A", "venue": "V", "year": 2024}
                ]
            }
        ),
        encoding="utf-8",
    )
    executor = _make_executor(tmp_path, llm_client=llm)
    state = _task_state(direction="方向")
    stage_root = artifacts / "english_paper_completed"
    executor.execute("english_paper_completed", state, stage_root)
    prompt = llm.text_calls[0]["prompt"]
    assert llm.text_calls[0]["role"] == "paper_writer"
    assert "Directed Topic Title" in prompt
    assert "Paper One" in prompt
    assert "APPROVED_TOPIC_PROPOSAL" in prompt
    artifact = json.loads((stage_root / "paper.json").read_text(encoding="utf-8"))
    assert artifact["llm_output"] == "paper body"
    assert (stage_root / "paper.md").read_text(encoding="utf-8").startswith("#")
    assert "\\begin{document}" in (stage_root / "paper.tex").read_text(encoding="utf-8")
