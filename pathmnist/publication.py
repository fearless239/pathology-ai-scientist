from __future__ import annotations

import re
from typing import Any


def publication_dataset_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Select scientific dataset facts while excluding audit-only identifiers."""
    allowed = {"image_shape", "channels", "classes", "label_mapping", "split_counts", "class_counts", "has_group_ids", "warnings", "recommended_metrics"}
    result = {key: profile[key] for key in allowed if key in profile}
    name = profile.get("display_name") or profile.get("name")
    if isinstance(name, str) and not profile.get("display_name"):
        name = re.sub(r"[_-](?:32|64|128|224)$", "", name)
    result["dataset_name"] = name
    result['preprocessing_reporting_guidance'] = (
        'image_shape describes the source dataset. Report model input_resolutions separately from the '
        'experiment manifest. Resizing the supplied images is allowed; when sizes differ, describe '
        'the resizing explicitly and never claim the source images were originally at the model input resolution.'
    )
    return result


def publication_integrity_summary(validation_receipt: dict[str, Any], test_receipt: dict[str, Any], metric_provenance: dict[str, Any]) -> dict[str, Any]:
    """Expose scientific conclusions without leaking audit implementation details."""
    return {
        "validation_data_consumption_verified": bool(validation_receipt),
        "held_out_test_consumption_verified": bool(test_receipt),
        "metrics_independently_recomputed": bool(metric_provenance),
        "reporting_guidance": (
            "State once in Methods that the final model and analysis protocol were fixed before a single evaluation on the held-out test set. "
            "Do not report hashes, artifact paths, runner identities, timestamps, attempts, receipts, evaluator versions, or internal workflow terminology."
        ),
    }


def publication_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep selection facts; experiment IDs, hashes and freeze timestamps stay in audit files."""
    allowed = {"primary_metric", "validation_value", "maximize"}
    return {key: candidate[key] for key in allowed if key in candidate}


def publication_research_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Expose pre-specified scientific requirements, never approval internals."""
    interventions = [
        {key: item[key] for key in ("id", "name", "description", "required") if key in item}
        for item in contract.get("interventions", []) if isinstance(item, dict)
    ]
    return {
        "research_question": contract.get("research_question"),
        "baseline": {key: contract.get("baseline", {}).get(key) for key in ("name", "description")},
        "interventions": interventions,
        "comparisons": contract.get("comparisons", []),
        "metrics": contract.get("metrics", {}),
        "subgroup_policy": contract.get("subgroup_policy", {}),
        "repeat_plan": contract.get("repeat_plan", {}),
        "statistical_plan": contract.get("statistical_plan", {}),
        "success_criteria": contract.get("success_criteria", []),
        "required_ablations": contract.get("required_ablations", []),
        "claim_boundary": contract.get("claim_boundary"),
    }


def publication_contract_results(fulfillment: dict[str, Any], test_comparison: dict[str, Any]) -> dict[str, Any]:
    """Select claim-bearing aggregate results without code/path/hash evidence."""
    return {
        "validation_contract_fulfilled": fulfillment.get("passed"),
        "hypothesis_supported_on_validation": fulfillment.get("hypothesis_supported"),
        "positive_claim_threshold": fulfillment.get("positive_claim_threshold"),
        "locked_confusion_pair": fulfillment.get("locked_confusion_pair"),
        "primary_metric": fulfillment.get("primary_metric"),
        "validation_statistics": fulfillment.get("statistics"),
        "validation_guardrails": fulfillment.get("guardrails"),
        "held_out_statistics": test_comparison.get("statistics"),
        "held_out_primary_metric": test_comparison.get("primary_metric"),
        "held_out_locked_confusion_pair": test_comparison.get("locked_confusion_pair"),
    }


def normalize_publication_language(markdown: str) -> str:
    """Replace workflow jargon when it leaks from a model response."""
    replacements = (
        (r"\bfrozen[- ]candidate\b", "final model"),
        (r"\bcandidate freezing\b", "final model selection"),
        (r"\bcandidate freeze\b", "final model selection"),
        (r"\bsealed test (?:split|set|arrays?)\b", "held-out test set"),
        (r"\bsealed test\b", "held-out test"),
        (r"\btrusted evaluator\b", "independent metric implementation"),
    )
    result = markdown
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result
