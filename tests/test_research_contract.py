import json

import pytest

from pathmnist.autonomous_stages import V2_STAGES
from pathmnist.research_contract import (
    ResearchContractError,
    approve_contract,
    code_semantic_evidence,
    contract_from_extraction,
    evaluate_fulfillment,
    generate_contract,
    load_contract,
    write_contract,
)


DIRECTION = (
    "面向PathMNIST易混淆组织类别的难例挖掘与分类性能提升。仅使用PathMNIST数据集，"
    "先训练ResNet-18作为基线并通过混淆矩阵确定最易混淆的一对组织类别，再引入基于预测置信度的"
    "难例挖掘和监督对比学习；以该易混淆类别对的平均F1值为核心指标，目标是在保持九分类宏平均F1"
    "不下降的前提下，将其相较基线提高至少3个百分点，并通过固定数据划分下的五次重复实验和统计检验验证稳定性。"
)


def test_single_comparison_has_no_unapproved_work_or_inference():
    contract = generate_contract('Single-run comparison of label smoothing accuracy', {'classes': ['0', '1']})
    assert contract['repeat_plan']['count'] == 1
    assert contract['experiment_policy'] == {'tuning': False, 'ablation': False}
    assert contract['required_ablations'] == []
    assert contract['statistical_plan']['test'] is None


@pytest.mark.parametrize('direction', ['单次对比，不需要调参，不做消融', 'single-run comparison without tuning, no ablation'])
def test_explicitly_excluded_work_is_not_scheduled(direction):
    contract = generate_contract(direction, {'classes': ['0', '1']})
    assert contract['experiment_policy'] == {'tuning': False, 'ablation': False}


def test_explicit_input_size_is_part_of_new_contract():
    contract = generate_contract('仅使用PathMNIST，将图像缩放为28×28训练CNN', {'classes': ['0', '1']})
    assert contract['execution_requirements']['input_sizes'] == [[28, 28]]


def test_legacy_input_size_derivation_does_not_modify_approved_contract():
    from gate_a.model_contract import input_sizes
    contract = {'research_question': '原始28×28图像训练', 'contract_sha256': 'unchanged'}
    assert input_sizes(contract) == [[28, 28]]
    assert contract == {'research_question': '原始28×28图像训练', 'contract_sha256': 'unchanged'}


def test_ambiguous_sizes_require_explicit_contract():
    from gate_a.model_contract import input_sizes
    with pytest.raises(ValueError, match='Ambiguous'):
        input_sizes({'research_question': '将64×64调整到28×28'})
    assert input_sizes({'execution_requirements': {'input_sizes': [[28, 28], [56, 56]]}}) == [[28, 28], [56, 56]]


def test_specific_direction_becomes_an_enforceable_contract():
    contract = generate_contract(DIRECTION, {"classes": [str(index) for index in range(9)]}, split_seed=7)
    assert contract["baseline"]["name"] == "ResNet-18"
    assert "hard-example mining" in contract["interventions"][0]["name"]
    assert "supervised contrastive" in contract["interventions"][0]["name"]
    assert contract["metrics"]["primary"]["name"] == "confusion_pair_mean_f1"
    assert contract["metrics"]["guardrails"][0]["name"] == "macro_f1"
    assert contract["success_criteria"][0]["minimum_delta"] == pytest.approx(0.03)
    assert contract["repeat_plan"] == {"count": 5, "seeds": [0, 1, 2, 3, 4], "fixed_split": True, "split_seed": 7}


def test_accuracy_augmentation_direction_is_preserved_exactly():
    direction = (
        "仅使用PathMNIST数据集，以ResNet-18为基线模型，引入颜色扰动、旋转和翻转等"
        "数据增强方法，并与不使用数据增强的模型进行对比。以测试集九分类准确率为核心指标，"
        "目标是在相同数据划分和训练条件下，使模型准确率较基线提高至少2个百分点。"
    )
    contract = generate_contract(direction, {"classes": [str(index) for index in range(9)]})
    assert contract["baseline"]["name"] == "ResNet-18"
    assert contract["metrics"]["primary"]["name"] == "accuracy"
    assert contract["success_criteria"][0]["metric"] == "accuracy"
    assert contract["success_criteria"][0]["minimum_delta"] == pytest.approx(0.02)
    assert set(contract["interventions"][0]["implementation_signals"]) == {
        "transform",
        "color_jitter",
        "rotation",
        "flip",
    }


def test_unsupported_task_is_diagnosed_before_execution():
    contract = generate_contract("使用WSI进行生存分析", {"classes": ["a", "b"]})
    assert not contract["capability"]["supported"]
    assert contract["capability"]["reasons"]


def test_approval_is_bound_to_the_contract_hash(tmp_path):
    task = {"schema_version": 2, "completed_stage": "research_contract_generated", "stages": {stage: "waiting" for stage in V2_STAGES}}
    task["stages"]["research_contract_generated"] = "completed"
    (tmp_path / "task.json").write_text(json.dumps(task), encoding="utf-8")
    write_contract(tmp_path, generate_contract("比较一种增强方法的宏平均F1", {"classes": ["a", "b"]}))
    approval = approve_contract(tmp_path)
    assert load_contract(tmp_path, require_approved=True)["contract_sha256"] == approval["contract_sha256"]


def test_tampered_approved_contract_is_rejected(tmp_path):
    task = {"schema_version": 2, "completed_stage": "research_contract_generated", "stages": {stage: "waiting" for stage in V2_STAGES}}
    task["stages"]["research_contract_generated"] = "completed"
    (tmp_path / "task.json").write_text(json.dumps(task), encoding="utf-8")
    path = write_contract(tmp_path, generate_contract("比较增强方法", {"classes": ["a", "b"]}))
    approve_contract(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["baseline"]["name"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ResearchContractError, match="hash"):
        load_contract(tmp_path, require_approved=True)


def test_semantic_code_evidence_requires_signals_and_training_operations():
    code = "confidence=0.8\nhard_examples=[1]\ncontrastive_loss=temperature=0.1\nloss.backward()\noptimizer.step()\n"
    assert not code_semantic_evidence(code, ["confidence", "hard", "contrastive", "temperature"])["passed"]
    code = 'loss = contrastive_loss(hard_example_mining(outputs))\nloss.backward()\noptimizer.step()'
    assert code_semantic_evidence(code, ["confidence", "hard", "contrastive", "temperature"])["passed"]


def test_structured_extraction_supports_a_different_classification_topic():
    extraction = {
        "supported": True, "unsupported_reasons": [], "baseline_name": "ConvNeXt-Tiny",
        "intervention_name": "stain-invariant augmentation", "intervention_description": "Apply stain perturbations during training.",
        "implementation_signals": ["stain_jitter"], "primary_metric": "weighted_f1", "primary_scope": "all_classes",
        "guardrail_metrics": ["accuracy"], "repeat_count": 4, "has_improvement_threshold": False,
        "minimum_improvement_delta": 0.0, "required_ablation": "Disable stain perturbations.",
    }
    contract = contract_from_extraction("研究染色增强", {"classes": ["a", "b"]}, extraction, split_seed=11)
    assert contract["baseline"]["name"] == "ConvNeXt-Tiny"
    assert contract["metrics"]["primary"]["name"] == "weighted_f1"
    assert contract["repeat_plan"]["seeds"] == [0, 1, 2, 3]


def _fulfillment_fixture(root, *, omit_proposed_seed=None, proposed_gain=0.04):
    stages = {stage: "waiting" for stage in V2_STAGES}
    stages["research_contract_generated"] = "completed"
    (root / "task.json").write_text(json.dumps({"schema_version": 2, "completed_stage": "research_contract_generated", "stages": stages}), encoding="utf-8")
    write_contract(root, generate_contract(DIRECTION, {"classes": [str(index) for index in range(9)]}, split_seed=7))
    approve_contract(root)
    experiments = []
    for role in ("baseline", "proposed_method"):
        for seed in range(5):
            if role == "proposed_method" and seed == omit_proposed_seed:
                continue
            experiment_id = f"{role}-{seed}"
            result_dir = root / "experiment_logs/results" / experiment_id
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "experiment_result.json").write_text("{}", encoding="utf-8")
            (result_dir / "contract_execution.json").write_text(json.dumps({
                "role_id": role, "seed": seed, "semantic_evidence": {"passed": True}
            }), encoding="utf-8")
            pair_f1 = 0.60 + (proposed_gain if role == "proposed_method" else 0.0)
            (result_dir / "trusted_metrics.json").write_text(json.dumps({"metrics": {
                "macro_f1": 0.70 + (0.01 if role == "proposed_method" else 0.0),
                "per_class": [{"f1": pair_f1}, {"f1": pair_f1}],
                "confusion_matrix": [[40, 10], [8, 42]],
            }}), encoding="utf-8")
            experiments.append({"experiment_id": experiment_id, "result": str((result_dir / "experiment_result.json").relative_to(root)), "code_sha256": ("a" if role == "baseline" else "b") * 64})
    ablation_dir = root / "experiment_logs/results/ablation"
    ablation_dir.mkdir(parents=True)
    (ablation_dir / "experiment_result.json").write_text("{}", encoding="utf-8")
    (ablation_dir / "contract_execution.json").write_text(json.dumps({"role_id": "component_ablation", "seed": 0, "semantic_evidence": {"passed": True}}), encoding="utf-8")
    (ablation_dir / "trusted_metrics.json").write_text(json.dumps({"metrics": {"macro_f1": 0.69}}), encoding="utf-8")
    experiments.append({"experiment_id": "ablation", "result": str((ablation_dir / "experiment_result.json").relative_to(root)), "code_sha256": "c" * 64})
    manifest = root / "experiment_logs/results/manifest.json"
    manifest.write_text(json.dumps({"experiments": experiments}), encoding="utf-8")


def test_fulfillment_distinguishes_completed_positive_result(tmp_path):
    _fulfillment_fixture(tmp_path)
    report = evaluate_fulfillment(tmp_path)
    assert report["passed"]
    assert report["hypothesis_supported"] is True
    assert report["locked_confusion_pair"] == [0, 1]
    assert report["statistics"]["mean_difference"] == pytest.approx(0.04)


def test_fulfillment_blocks_missing_repeat_instead_of_accepting_stage_name(tmp_path):
    _fulfillment_fixture(tmp_path, omit_proposed_seed=4)
    report = evaluate_fulfillment(tmp_path)
    assert not report["passed"]
    assert any("required 5 unique repeat seeds" in error for error in report["errors"])


def test_fulfillment_prefers_stage_two_seed_baseline_and_records_pair_sources(tmp_path):
    _fulfillment_fixture(tmp_path)
    manifest_path = tmp_path / "experiment_logs/results/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = next(
        item for item in manifest["experiments"]
        if item["experiment_id"] == "baseline-0"
    )
    binding_path = tmp_path / baseline["result"]
    binding_path = binding_path.parent / "contract_execution.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding.update({"upstream_stage": "2_baseline_tuning_1_first_attempt", "is_seed_node": True})
    binding_path.write_text(json.dumps(binding), encoding="utf-8")

    old_dir = tmp_path / "experiment_logs/results/baseline-stage1-0"
    old_dir.mkdir(parents=True)
    (old_dir / "experiment_result.json").write_text("{}", encoding="utf-8")
    (old_dir / "trusted_metrics.json").write_text(
        json.dumps({"metrics": {"macro_f1": 0.01, "per_class": [{"f1": 0.01}, {"f1": 0.01}], "confusion_matrix": [[40, 10], [8, 42]]}}),
        encoding="utf-8",
    )
    (old_dir / "contract_execution.json").write_text(
        json.dumps({"role_id": "baseline", "seed": 0, "upstream_stage": "1_initial_baseline", "semantic_evidence": {"passed": True}}),
        encoding="utf-8",
    )
    manifest["experiments"].append({
        "experiment_id": "baseline-stage1-0",
        "result": str((old_dir / "experiment_result.json").relative_to(tmp_path)),
        "code_sha256": "d" * 64,
    })
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_fulfillment(tmp_path)
    assert report["passed"]
    pair = next(item for item in report["paired_sources"] if item["seed"] == 0)
    assert pair["baseline_experiment_id"] == "baseline-0"
    assert pair["baseline_stage"] == "2_baseline_tuning_1_first_attempt"


def test_completed_but_unmet_threshold_is_a_publishable_negative_result(tmp_path):
    _fulfillment_fixture(tmp_path, proposed_gain=0.01)
    report = evaluate_fulfillment(tmp_path)
    assert report["passed"]
    assert report["hypothesis_supported"] is False


@pytest.mark.parametrize("change,passed", [(None, True), ("limit", False), ("stop", False),
                                         ("legacy", False)])
def test_fulfillment_compares_training_policy_not_completed_epochs(tmp_path, change, passed):
    _fulfillment_fixture(tmp_path)
    for role, epochs in (("baseline", 14), ("proposed_method", 15)):
        path = tmp_path / f"experiment_logs/results/{role}-0/contract_execution.json"
        binding = json.loads(path.read_text())
        controls = {"epochs": epochs, "max_epochs": 15,
                    "early_stopping": {"enabled": True, "monitor": "validation_loss",
                                       "mode": "min", "patience": 5, "min_delta": 0.0}}
        if role == "proposed_method":
            if change == "limit":
                controls["max_epochs"] = 16
            elif change == "stop":
                controls["early_stopping"]["patience"] = 3
            elif change == "legacy":
                controls = {"epochs": 15}
        binding["execution_controls"] = controls
        path.write_text(json.dumps(binding))
    report = evaluate_fulfillment(tmp_path)
    assert report["passed"] is passed
    if not passed:
        assert any("fixed controls differ" in error for error in report["errors"])
