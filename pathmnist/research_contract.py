from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .trusted_statistics import StatisticsError, comparison_summary
from .metrics import metric_value
from gate_a.model_contract import input_sizes


CONTRACT_SCHEMA_VERSION = 2
SUPPORTED_TASK_FAMILY = "supervised_image_classification"


class ResearchContractError(RuntimeError):
    """Raised when a research request cannot be converted into an enforceable contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def contract_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _repeat_count(direction: str) -> int:
    if re.search(r'各一次|单次对比|一次训练|single[- ](?:run|repeat)|one repeat', direction, re.I):
        return 1
    chinese = {"三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    match = re.search(r"(?:固定[^。；,，]{0,24})?([2-9]|10)\s*(?:次|个)\s*(?:重复|独立|随机种子|seed)", direction, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"([三四五六七八九十])次(?:重复|独立|实验)", direction)
    if match:
        return chinese[match.group(1)]
    return 3


def _requested_optional_work(text: str, pattern: str) -> bool:
    # Explicitly excluded work must not become an approved stage merely because
    # its name occurs in the research direction.
    negative = rf'(?:不需要|无需|不要|不做|不进行|不开展|不|跳过|without|skip|no|do not|don.t)\s*(?:再|进行|做|perform|run)?\s*(?:{pattern})'
    remaining = re.sub(negative, '', text, flags=re.I)
    return bool(re.search(pattern, remaining, re.I))


def _threshold(direction: str) -> float | None:
    match = re.search(r"(?:提高|提升|改善|增加)\s*(?:至少|不低于)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:个)?百分点", direction)
    if match:
        return float(match.group(1)) / 100.0
    match = re.search(r"(?:提高|提升|改善|增加)\s*(?:至少|不低于)?\s*([0-9]+(?:\.[0-9]+)?)\s*%", direction)
    return float(match.group(1)) / 100.0 if match else None


def _unsupported_reasons(direction: str) -> list[str]:
    checks = {
        "segmentation": ("分割", "segmentation", "mask prediction"),
        "whole-slide imaging": ("全切片", "wsi", "whole slide"),
        "survival analysis": ("生存分析", "survival", "cox"),
        "multimodal learning": ("多模态", "multimodal", "影像组学联合"),
    }
    lowered = direction.casefold()
    return [name for name, tokens in checks.items() if any(token in lowered for token in tokens)]


def _intervention(direction: str) -> tuple[str, list[str]]:
    lowered = direction.casefold()
    names: list[str] = []
    signals: list[str] = []
    if any(token in lowered for token in ("难例", "hard example", "hard mining", "置信度")):
        names.append("confidence-based hard-example mining")
        signals.extend(("confidence", "hard"))
    if any(token in lowered for token in ("监督对比", "supervised contrastive", "supcon")):
        names.append("supervised contrastive learning")
        signals.extend(("contrastive", "temperature"))
    if any(token in lowered for token in ("增强", "augmentation")) and not names:
        names.append("task-specified data augmentation")
        signals.append("transform")
        if any(token in lowered for token in ("颜色", "染色", "color", "stain")):
            signals.append("color_jitter")
        if any(token in lowered for token in ("旋转", "rotation")):
            signals.append("rotation")
        if any(token in lowered for token in ("翻转", "flip")):
            signals.append("flip")
    if not names:
        names.append("the method specified in the submitted research direction")
    return " + ".join(names), list(dict.fromkeys(signals))


def generate_contract(
    direction: str, profile: dict[str, Any], *, split_seed: int = 7, revision_feedback: str = ""
) -> dict[str, Any]:
    direction = direction.strip()
    if not direction:
        raise ResearchContractError("Research direction is empty")
    interpretation = direction + ("\nUser revision: " + revision_feedback.strip() if revision_feedback.strip() else "")
    unsupported = _unsupported_reasons(interpretation)
    intervention_name, signals = _intervention(interpretation)
    baseline_name = "ResNet-18" if re.search(r"resnet\s*[-_]?\s*18", interpretation, re.I) else "standard image-classification baseline"
    hard_pair = any(token in interpretation.casefold() for token in ("易混淆", "类别对", "confusion pair", "难例"))
    accuracy_requested = bool(
        re.search(
            r"(?:九分类|分类|test|测试集)?[^。；,，]{0,20}(?:准确率|accuracy)",
            interpretation,
            re.I,
        )
    )
    primary_name = (
        "confusion_pair_mean_f1"
        if hard_pair
        else "accuracy"
        if accuracy_requested
        else "macro_f1"
    )
    repeats = _repeat_count(revision_feedback) if revision_feedback.strip() and re.search(r"(?:次|seed)", revision_feedback, re.I) else _repeat_count(direction)
    # The data split seed is a separate reproducibility control.  Upstream's
    # multi-seed evaluator deterministically uses 0..n-1 for training repeats.
    seeds = list(range(repeats))
    threshold = _threshold(revision_feedback) if revision_feedback.strip() and _threshold(revision_feedback) is not None else _threshold(direction)
    guardrails = []
    if re.search(r"(?:宏平均|macro)[-_ ]?f1[^。；]{0,24}(?:不下降|不降低|non[- ]?inferior)", interpretation, re.I):
        guardrails.append({
            "id": "macro_f1_non_decrease",
            "name": "macro_f1",
            "scope": "all_classes",
            "direction": "maximize",
            "minimum_delta": 0.0,
        })
    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_revision": 1,
        "task_family": SUPPORTED_TASK_FAMILY,
        "capability": {
            "supported": not unsupported,
            "reasons": [f"P0 does not support {name}" for name in unsupported],
        },
        "research_question": direction,
        "execution_requirements": {"input_sizes": input_sizes({'research_question': direction})} if re.search(r'\d\s*[×xX]\s*\d', direction) else {},
        "baseline": {"id": "baseline", "name": baseline_name, "description": f"Train {baseline_name} on the fixed research split."},
        "interventions": [{
            "id": "proposed_method",
            "name": intervention_name,
            "description": direction,
            "required": True,
            "implementation_signals": signals,
        }],
        "comparisons": [{
            "id": "baseline_vs_proposed",
            "baseline_id": "baseline",
            "intervention_id": "proposed_method",
            "required": True,
        }],
        "metrics": {
            "primary": {
                "id": "primary_metric",
                "name": primary_name,
                "scope": "baseline_locked_confusion_pair" if hard_pair else "all_classes",
                "direction": "maximize",
            },
            "guardrails": guardrails,
            "secondary": [
                *(
                    []
                    if primary_name == "accuracy"
                    else [{"name": "accuracy", "scope": "all_classes"}]
                ),
                {"name": "macro_f1", "scope": "all_classes"},
                {"name": "weighted_f1", "scope": "all_classes"},
                {"name": "confusion_matrix", "scope": "all_classes"},
            ],
        },
        "subgroup_policy": {
            "kind": "lock_most_confused_pair_from_baseline" if hard_pair else "none",
            "selection_split": "validation",
            "locked_before_intervention_comparison": hard_pair,
        },
        "repeat_plan": {"count": repeats, "seeds": seeds, "fixed_split": True, "split_seed": int(split_seed)},
        "experiment_policy": {"tuning": _requested_optional_work(interpretation, r'调参|确定参数|参数选择|tuning|hyperparameter'),
                              "ablation": _requested_optional_work(interpretation, r'消融|ablation')},
        "statistical_plan": {
            "paired": True,
            "test": "exact_paired_sign_flip",
            "confidence_interval": "paired_bootstrap_95",
            "effect_size": "standardized_mean_paired_difference",
            "alpha": 0.05,
        },
        "success_criteria": [{
            "id": "primary_improvement",
            "metric": primary_name,
            "comparison_id": "baseline_vs_proposed",
            "minimum_delta": threshold,
            "required_for_positive_claim": threshold is not None,
        }],
        "required_ablations": [{"id": "component_ablation", "description": "Remove or disable an essential proposed-method component while keeping the split and seed fixed."}],
        "resource_plan": {
            "mode": "full",
            "api_hard_limit_usd": 8.0,
            "stage_max_iterations": [20, 12, 12, 18],
            "experiment_timeout_seconds": 3600,
        },
        "claim_boundary": "supervised patch-level image classification; no clinical or patient-level claims",
        "revision_feedback": revision_feedback.strip(),
        "generated_at": _now(),
    }
    _normalize_optional_work(contract)
    validate_contract(contract)
    contract["contract_sha256"] = contract_sha256(contract)
    return contract


def extraction_schema() -> dict[str, Any]:
    """Compact LLM interface; deterministic validation builds the actual contract."""
    return {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "unsupported_reasons": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 5},
            "baseline_name": {"type": "string", "minLength": 1, "maxLength": 120},
            "intervention_name": {"type": "string", "minLength": 1, "maxLength": 240},
            "intervention_description": {"type": "string", "minLength": 1, "maxLength": 1200},
            "implementation_signals": {"type": "array", "items": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]{2,39}$"}, "maxItems": 8},
            "primary_metric": {"type": "string", "enum": ["accuracy", "macro_f1", "weighted_f1", "class_f1", "confusion_pair_mean_f1"]},
            "primary_scope": {"type": "string", "enum": ["all_classes", "specified_class", "baseline_locked_confusion_pair"]},
            "guardrail_metrics": {"type": "array", "items": {"type": "string", "enum": ["accuracy", "macro_f1", "weighted_f1"]}, "maxItems": 3},
            "repeat_count": {"type": "integer", "minimum": 1, "maximum": 10},
            "primary_class_id": {"type": "integer", "minimum": 0},
            "has_improvement_threshold": {"type": "boolean"},
            "minimum_improvement_delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            "required_ablation": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": [
            "supported", "unsupported_reasons", "baseline_name", "intervention_name",
            "intervention_description", "implementation_signals", "primary_metric", "primary_scope",
            "guardrail_metrics", "repeat_count", "has_improvement_threshold",
            "minimum_improvement_delta", "required_ablation",
        ],
        "additionalProperties": False,
    }


def contract_from_extraction(
    direction: str,
    profile: dict[str, Any],
    extraction: dict[str, Any],
    *,
    split_seed: int = 7,
    revision_feedback: str = "",
) -> dict[str, Any]:
    contract = generate_contract(direction, profile, split_seed=split_seed, revision_feedback=revision_feedback)
    contract["capability"] = {
        "supported": bool(extraction["supported"]),
        "reasons": [str(item) for item in extraction["unsupported_reasons"]],
    }
    contract["baseline"]["name"] = str(extraction["baseline_name"])
    contract["baseline"]["description"] = f"Train {extraction['baseline_name']} on the fixed research split."
    intervention = contract["interventions"][0]
    intervention["name"] = str(extraction["intervention_name"])
    intervention["description"] = str(extraction["intervention_description"])
    intervention["implementation_signals"] = list(dict.fromkeys(map(str, extraction["implementation_signals"])))
    from .method_spec import classify_requirements
    intervention['requirement_groups'] = classify_requirements(intervention['implementation_signals'])
    primary = contract["metrics"]["primary"]
    primary["name"] = str(extraction["primary_metric"])
    if primary['name'] == 'class_f1':
        primary['class_id'] = extraction.get('primary_class_id')
    primary["scope"] = str(extraction["primary_scope"])
    contract["subgroup_policy"] = {
        "kind": "lock_most_confused_pair_from_baseline" if primary["scope"] == "baseline_locked_confusion_pair" else "none",
        "selection_split": "validation",
        "locked_before_intervention_comparison": primary["scope"] == "baseline_locked_confusion_pair",
    }
    contract["metrics"]["guardrails"] = [
        {"id": f"{name}_non_decrease", "name": name, "scope": "all_classes", "direction": "maximize", "minimum_delta": 0.0}
        for name in dict.fromkeys(map(str, extraction["guardrail_metrics"]))
    ]
    repeat_count = int(extraction["repeat_count"])
    if _repeat_count(direction) == 1:
        repeat_count = 1
    contract["repeat_plan"] = {"count": repeat_count, "seeds": list(range(repeat_count)), "fixed_split": True, "split_seed": int(split_seed)}
    criterion = contract["success_criteria"][0]
    criterion["metric"] = primary["name"]
    criterion["minimum_delta"] = float(extraction["minimum_improvement_delta"]) if extraction["has_improvement_threshold"] else None
    criterion["required_for_positive_claim"] = bool(extraction["has_improvement_threshold"])
    contract["required_ablations"] = [{"id": "component_ablation", "description": str(extraction["required_ablation"])}]
    _normalize_optional_work(contract)
    contract.pop("contract_sha256", None)
    validate_contract(contract)
    contract["contract_sha256"] = contract_sha256(contract)
    return contract


def _normalize_optional_work(contract: dict[str, Any]) -> None:
    """Only used while creating a new contract, never while loading history."""
    if not contract['experiment_policy']['ablation']:
        contract['required_ablations'] = []
    if contract['repeat_plan']['count'] == 1:
        contract['statistical_plan'] = {
            'paired': True, 'test': None, 'confidence_interval': None,
            'effect_size': None, 'mode': 'descriptive_only',
        }


def validate_contract(contract: dict[str, Any]) -> None:
    if "execution_plan" in contract:
        from .execution_plan import ExecutionPlanError, validate_plan

        try:
            validate_plan(contract["execution_plan"])
        except (ExecutionPlanError, TypeError, ValueError, KeyError) as error:
            raise ResearchContractError(f"Invalid execution plan: {error}") from error
    if contract.get("schema_version") not in (1, CONTRACT_SCHEMA_VERSION):
        raise ResearchContractError("Unsupported research contract schema")
    if contract.get('schema_version') == 2:
        policy = contract.get('experiment_policy')
        if not isinstance(policy, dict) or any(type(policy.get(key)) is not bool for key in ('tuning', 'ablation')):
            raise ResearchContractError('experiment_policy must specify tuning and ablation')
    primary_definition = contract.get('metrics', {}).get('primary', {})
    if primary_definition.get('name') == 'class_f1' and (
        type(primary_definition.get('class_id')) is not int or primary_definition['class_id'] < 0
    ):
        raise ResearchContractError('class_f1 requires an explicit class_id')
    if contract.get("task_family") != SUPPORTED_TASK_FAMILY:
        raise ResearchContractError("Only supervised image classification is supported in P0")
    capability = contract.get("capability")
    if not isinstance(capability, dict) or not isinstance(capability.get("supported"), bool):
        raise ResearchContractError("Contract has no capability decision")
    baseline = contract.get("baseline")
    interventions = contract.get("interventions")
    comparisons = contract.get("comparisons")
    if not isinstance(baseline, dict) or not baseline.get("id"):
        raise ResearchContractError("Contract has no baseline")
    if not isinstance(interventions, list) or not interventions:
        raise ResearchContractError("Contract has no intervention")
    if not isinstance(comparisons, list) or not comparisons:
        raise ResearchContractError("Contract has no comparison")
    metrics = contract.get("metrics")
    primary = metrics.get("primary") if isinstance(metrics, dict) else None
    if not isinstance(primary, dict) or primary.get("name") not in {
        "accuracy", "macro_f1", "weighted_f1", "class_f1", "confusion_pair_mean_f1"
    }:
        raise ResearchContractError("Contract primary metric is unsupported")
    repeat = contract.get("repeat_plan")
    if not isinstance(repeat, dict) or not 1 <= int(repeat.get("count", 0)) <= 10:
        raise ResearchContractError("Repeat count must be between 1 and 10")
    seeds = repeat.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != repeat["count"] or len(set(seeds)) != len(seeds):
        raise ResearchContractError("Repeat seeds must be complete and unique")


def write_contract(task_root: Path, contract: dict[str, Any]) -> Path:
    validate_contract(contract)
    path = task_root / "research/research_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_contract(task_root: Path, *, require_approved: bool = False) -> dict[str, Any]:
    path = task_root / "research/research_contract.json"
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ResearchContractError("Research contract is missing or invalid") from exc
    validate_contract(contract)
    expected = contract.get("contract_sha256")
    unhashed = dict(contract)
    unhashed.pop("contract_sha256", None)
    if expected != contract_sha256(unhashed):
        raise ResearchContractError("Research contract hash does not match its contents")
    if require_approved:
        approval_path = task_root / "research/research_contract_approval.json"
        try:
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ResearchContractError("Research contract has not been approved") from exc
        if approval.get("contract_sha256") != expected or approval.get("approved") is not True:
            raise ResearchContractError("Research contract approval does not match the active contract")
    return contract


def approve_contract(task_root: Path) -> dict[str, Any]:
    task_path = task_root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("completed_stage") != "research_contract_generated":
        raise ResearchContractError("Only a generated contract can be approved")
    contract = load_contract(task_root)
    if not contract["capability"]["supported"]:
        raise ResearchContractError("Unsupported research contracts cannot be approved")
    approval = {
        "schema_version": 1,
        "approved": True,
        "contract_sha256": contract["contract_sha256"],
        "approved_at": _now(),
    }
    approval_path = task_root / "research/research_contract_approval.json"
    approval_path.write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for stage in ("research_contract_approved", "experiment_spec_validated"):
        task["stages"][stage] = "completed"
    task["completed_stage"] = "experiment_spec_validated"
    task["control"] = "paused"
    task["updated_at"] = _now()
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return approval


def code_semantic_evidence(code: str, signals: list[str]) -> dict[str, Any]:
    from .method_spec import canonical_concept, normalize_symbol, semantic_report

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"passed": False, "signals": {}, "reason": "invalid Python syntax"}
    identifiers: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.append((node.id.casefold(), int(getattr(node, "lineno", 0))))
        elif isinstance(node, ast.Attribute):
            identifiers.append((node.attr.casefold(), int(getattr(node, "lineno", 0))))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.append((node.name.casefold(), int(getattr(node, "lineno", 0))))
        elif isinstance(node, ast.alias):
            # Imported transform classes are executable semantic evidence too;
            # ast.alias nodes do not carry their own line number, so preserve a
            # positive sentinel for the existing evidence shape.
            identifiers.append((node.name.casefold(), 1))
    report = semantic_report(code, signals)
    found = {}
    for signal in signals:
        concept = canonical_concept(signal)
        if concept in report.get('detected', []):
            found[signal] = report["signals"][concept]
            continue
        normalized_signal = normalize_symbol(signal)
        found[signal] = sorted(
            {
                line
                for identifier, line in identifiers
                if normalized_signal in normalize_symbol(identifier) and line > 0
            }
        )
    has_training = any(isinstance(node, ast.Call) and getattr(node.func, "attr", "") in {"backward", "step"} for node in ast.walk(tree))
    # Export must not accept a method that the shared preflight rejected merely
    # because a comment, import or variable happens to contain its name.
    passed = report['passed']
    return {
        **report,
        "passed": passed,
        "signals": found,
        "has_training_operations": has_training,
    }


def _metric_value(metrics: dict[str, Any], name: str, pair: tuple[int, int] | None) -> float | None:
    return metric_value(metrics, name, pair)


def _confusion_pair(metrics: dict[str, Any]) -> tuple[int, int] | None:
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) < 2:
        return None
    best: tuple[int, int, int] | None = None
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            try:
                score = int(matrix[left][right]) + int(matrix[right][left])
            except (IndexError, TypeError, ValueError):
                return None
            candidate = (score, -left, -right)
            if best is None or candidate > best:
                best = candidate
    return (-best[1], -best[2]) if best else None


def review_implementation_semantics(project_root: Path, task_root: Path) -> dict[str, Any]:
    """Use the independent reviewer role to map method claims to executed source."""
    from gate_a.budget import BudgetLedger
    from gate_a.config import load_config
    from gate_a.pipeline import select_live_models
    from gate_a.provider import ZhipuProvider

    contract = load_contract(task_root, require_approved=True)
    manifest = json.loads((task_root / "experiment_logs/results/manifest.json").read_text(encoding="utf-8"))
    proposed = next((item for item in manifest.get("experiments", []) if item.get("contract_role") == "proposed_method"), None)
    baseline = next((item for item in manifest.get("experiments", []) if item.get("contract_role") == "baseline"), None)
    if proposed is None or baseline is None:
        raise ResearchContractError("Semantic review requires exported baseline and intervention code")
    proposed_code = (task_root / str(proposed["result"])).parent.joinpath("run.py").read_text(encoding="utf-8")
    baseline_code = (task_root / str(baseline["result"])).parent.joinpath("run.py").read_text(encoding="utf-8")
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    config = load_config(project_root.resolve() / "configs/gate_a_llm.yaml")
    ledger = BudgetLedger(task_root / "budget.json", float(task.get("budget_limit_usd", 8.0)))
    provider = ZhipuProvider(config, select_live_models(config), ledger, task_root / "research/responses")
    schema = {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "mappings": {"type": "array", "maxItems": 12, "items": {
                "type": "object", "properties": {
                    "contract_item": {"type": "string", "maxLength": 160},
                    "code_evidence": {"type": "string", "maxLength": 500},
                    "implemented": {"type": "boolean"},
                }, "required": ["contract_item", "code_evidence", "implemented"], "additionalProperties": False,
            }},
            "issues": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 500}},
        },
        "required": ["passed", "mappings", "issues"], "additionalProperties": False,
    }
    scientific_contract = {
        "baseline": contract["baseline"], "interventions": contract["interventions"],
        "comparisons": contract["comparisons"], "required_ablations": contract["required_ablations"],
    }
    prompt = (
        "Independently review whether the intervention source actually implements every method component in the approved "
        "scientific contract and is materially distinct from the baseline. Map claims to concrete functions, classes, losses, "
        "sampling operations, or training branches. Identifiers placed only in comments, output JSON, or unused variables are not "
        "implementation. Set passed=false if a required component is absent, cosmetic, or unverifiable.\n\nCONTRACT:\n"
        + json.dumps(scientific_contract, ensure_ascii=False, indent=2)
        + "\n\nBASELINE SOURCE (possibly truncated):\n" + baseline_code[:20_000]
        + "\n\nINTERVENTION SOURCE (possibly truncated):\n" + proposed_code[:35_000]
    )
    review, _ = provider.call_json(
        "reviewer", f"{task['task_id']}-contract-semantic-review-v1-{contract['contract_sha256'][:16]}",
        "You are an independent ML implementation reviewer. Return only JSON.", prompt,
        "semantic_contract_review", schema,
    )
    record = {
        "schema_version": 1, "contract_sha256": contract["contract_sha256"],
        "reviewer_role": "independent_reviewer", **review, "reviewed_at": _now(),
    }
    path = task_root / "research/semantic_review.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def evaluate_fulfillment(task_root: Path, *, require_semantic_review: bool = False) -> dict[str, Any]:
    contract = load_contract(task_root, require_approved=True)
    manifest_path = task_root / "experiment_logs/results/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_count = int(contract["repeat_plan"]["count"])
    expected_seeds = set(map(int, contract["repeat_plan"]["seeds"]))
    records: dict[str, list[dict[str, Any]]] = {"baseline": [], "proposed_method": [], "component_ablation": []}
    errors: list[str] = []
    if require_semantic_review:
        review_path = task_root / "research/semantic_review.json"
        if not review_path.is_file():
            errors.append("independent implementation semantic review is missing")
        else:
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if review.get("contract_sha256") != contract["contract_sha256"] or review.get("passed") is not True:
                errors.append("independent implementation semantic review did not pass")
    for item in manifest.get("experiments", []):
        result_path = task_root / str(item.get("result", ""))
        binding_path = result_path.parent / "contract_execution.json"
        trusted_path = result_path.parent / "trusted_metrics.json"
        if not binding_path.is_file() or not trusted_path.is_file():
            continue
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        trusted = json.loads(trusted_path.read_text(encoding="utf-8")).get("metrics", {})
        role = str(binding.get("role_id", ""))
        if role not in records:
            continue
        records[role].append({**item, "seed": int(binding.get("seed", -1)), "metrics": trusted, "binding": binding})
    def select_seed_references(role: str) -> dict[int, dict[str, Any]]:
        """Resolve one auditable result per seed without iteration-order wins."""
        selected: dict[int, dict[str, Any]] = {}
        for seed in sorted(expected_seeds):
            candidates = [
                row for row in records[role]
                if row["seed"] == seed
                and not row["binding"].get("is_seed_aggregation", False)
            ]
            if role == "baseline":
                tuned = [
                    row for row in candidates
                    if str(row["binding"].get("upstream_stage", "")).startswith("2_")
                ]
                if tuned:
                    candidates = tuned
            seeded = [row for row in candidates if row["binding"].get("is_seed_node") is True]
            if seeded:
                candidates = seeded
            if len(candidates) == 1:
                selected[seed] = candidates[0]
            elif len(candidates) > 1:
                errors.append(
                    f"{role} seed {seed} has {len(candidates)} equally preferred trusted results; "
                    "host pairing refuses an ambiguous reference"
                )
        return selected

    selected_records = {
        role: select_seed_references(role)
        for role in ("baseline", "proposed_method")
    }
    for role in ("baseline", "proposed_method"):
        seeds = set(selected_records[role])
        if not expected_seeds.issubset(seeds):
            errors.append(f"{role} lacks the required {required_count} unique repeat seeds")
    if contract.get('experiment_policy', {}).get('ablation', True) and not records["component_ablation"]:
        errors.append("required component ablation is missing")
    baseline_hashes = {row.get("code_sha256") for row in selected_records["baseline"].values()}
    proposed_hashes = {row.get("code_sha256") for row in selected_records["proposed_method"].values()}
    if baseline_hashes & proposed_hashes:
        errors.append("baseline and intervention reuse an identical code hash")
    if any(not row["binding"].get("semantic_evidence", {}).get("passed") for row in records["proposed_method"]):
        errors.append("intervention code does not satisfy its semantic evidence signals")
    ablation_hashes = {row.get("code_sha256") for row in records["component_ablation"]}
    if ablation_hashes and ablation_hashes.issubset(proposed_hashes):
        errors.append("ablation reuses the complete proposed-method code without a distinct implementation")
    pair = None
    if contract["subgroup_policy"]["kind"] == "lock_most_confused_pair_from_baseline" and records["baseline"]:
        baseline_by_seed = {seed: row["metrics"] for seed, row in selected_records["baseline"].items()}
        matrices = [baseline_by_seed[seed].get("confusion_matrix") for seed in sorted(expected_seeds) if seed in baseline_by_seed]
        aggregate = None
        if matrices and all(isinstance(matrix, list) for matrix in matrices):
            try:
                size = len(matrices[0])
                aggregate = [[sum(int(matrix[row][column]) for matrix in matrices) for column in range(size)] for row in range(size)]
            except (IndexError, TypeError, ValueError):
                aggregate = None
        pair = _confusion_pair({"confusion_matrix": aggregate}) if aggregate else None
        if pair is None:
            errors.append("baseline confusion matrix cannot lock a class pair")
        locked = task_root / 'research/locked_metric.json'
        if locked.is_file():
            lock_record = json.loads(locked.read_text(encoding='utf-8'))
            if lock_record.get('contract_sha256') != contract['contract_sha256']:
                errors.append('Locked metric contract hash mismatch')
            pair = lock_record.get('pair')
    primary = contract["metrics"]["primary"]["name"]
    control_keys = (
        "dataset", "model", "optimizer", "learning_rate",
        "batch_size", "input_resolutions", "selection_metric",
    )
    pair_sources = []
    for seed in sorted(expected_seeds & selected_records["baseline"].keys() & selected_records["proposed_method"].keys()):
        baseline_row = selected_records["baseline"][seed]
        method_row = selected_records["proposed_method"][seed]
        baseline_controls = baseline_row["binding"].get("execution_controls")
        method_controls = method_row["binding"].get("execution_controls")
        mismatches = []
        if isinstance(baseline_controls, dict) and isinstance(method_controls, dict):
            from .experiment_manifest import ManifestError, training_policy

            mismatches = [
                key for key in control_keys
                if baseline_controls.get(key) != method_controls.get(key)
            ]
            try:
                baseline_policy = training_policy(baseline_controls)
                method_policy = training_policy(method_controls)
                if baseline_policy != method_policy:
                    mismatches.append("max_epochs/early_stopping")
                elif baseline_policy is None and baseline_controls.get("epochs") != method_controls.get("epochs"):
                    mismatches.append("epochs")
            except ManifestError as error:
                mismatches.append(f"invalid training policy: {error}")
            if mismatches:
                errors.append(
                    f"seed {seed} baseline/proposed fixed controls differ: {mismatches}"
                )
        pair_sources.append({
            "seed": seed,
            "baseline_experiment_id": baseline_row.get("experiment_id"),
            "proposed_experiment_id": method_row.get("experiment_id"),
            "baseline_stage": baseline_row["binding"].get("upstream_stage"),
            "proposed_stage": method_row["binding"].get("upstream_stage"),
            "fixed_controls_match": not mismatches,
        })
    class_id = contract['metrics']['primary'].get('class_id')
    base_by_seed = {seed: metric_value(row["metrics"], primary, pair, class_id) for seed, row in selected_records["baseline"].items()}
    method_by_seed = {seed: metric_value(row["metrics"], primary, pair, class_id) for seed, row in selected_records["proposed_method"].items()}
    paired_seeds = sorted(expected_seeds & base_by_seed.keys() & method_by_seed.keys())
    if any(base_by_seed.get(seed) is None or method_by_seed.get(seed) is None for seed in paired_seeds):
        errors.append(f"trusted primary metric {primary!r} is missing for a required repeat")
    statistics = None
    if not errors:
        try:
            statistics = comparison_summary(
                [base_by_seed[seed] for seed in paired_seeds],
                [method_by_seed[seed] for seed in paired_seeds],
                bootstrap_seed=int(contract["repeat_plan"]["split_seed"]),
            )
        except StatisticsError as exc:
            errors.append(str(exc))
    guardrail_results = []
    for guardrail in contract["metrics"].get("guardrails", []):
        name = guardrail["name"]
        base_map = {
            seed: _metric_value(row["metrics"], name, pair)
            for seed, row in selected_records["baseline"].items()
        }
        method_map = {
            seed: _metric_value(row["metrics"], name, pair)
            for seed, row in selected_records["proposed_method"].items()
        }
        base_values = [base_map.get(seed) for seed in sorted(expected_seeds)]
        method_values = [method_map.get(seed) for seed in sorted(expected_seeds)]
        complete = len(base_values) == required_count and len(method_values) == required_count and None not in base_values + method_values
        delta = (sum(method_values) / len(method_values) - sum(base_values) / len(base_values)) if complete else None
        passed = complete and delta is not None and delta >= float(guardrail.get("minimum_delta", 0.0))
        guardrail_results.append({"name": name, "delta": delta, "passed": passed})
        if not complete:
            errors.append(f"guardrail metric {name!r} is incomplete")
    criterion = contract["success_criteria"][0]
    target = criterion.get("minimum_delta")
    hypothesis_supported = (
        None
        if target is None
        else bool(
            statistics is not None
            and statistics["mean_difference"] >= float(target)
            and all(row["passed"] for row in guardrail_results)
        )
    )
    report = {
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "passed": not errors,
        "hypothesis_supported": hypothesis_supported if not errors else None,
        "positive_claim_threshold": target,
        "locked_confusion_pair": list(pair) if pair else None,
        "primary_metric": primary,
        "repeat_count": required_count,
        "statistics": statistics,
        "paired_sources": pair_sources,
        "guardrails": guardrail_results,
        "checks": {
            "research_contract_integrity": "passed",
            "comparison_integrity": "passed" if not any(any(token in error for token in ("baseline", "intervention", "semantic review", "ablation")) for error in errors) else "failed",
            "repeat_integrity": "passed" if not any("repeat" in error or "seeds" in error for error in errors) else "failed",
            "statistical_integrity": "passed" if statistics is not None else "failed",
        },
        "errors": errors,
        "evaluated_at": _now(),
    }
    output = task_root / "research/contract_fulfillment.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
