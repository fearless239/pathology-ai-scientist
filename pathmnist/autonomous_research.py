from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .literature import (
    LiteratureError,
    canonical_reference_key,
    filter_relevant_literature,
    search_verified_literature,
)
from .research_contract import contract_from_extraction, contract_sha256, extraction_schema, generate_contract, write_contract


def _generate_contract(
    project_root: Path | None,
    task_root: Path,
    task: dict,
    direction: str,
    profile: dict,
    revision_feedback: str,
) -> dict:
    split_seed = int(task.get("seed", 7))
    if project_root is None or not os.getenv("PARATERA_API_KEY", "").strip():
        return generate_contract(direction, profile, split_seed=split_seed, revision_feedback=revision_feedback)
    from gate_a.budget import BudgetLedger
    from gate_a.config import load_config
    from gate_a.pipeline import select_live_models
    from gate_a.provider import ZhipuProvider

    config = load_config(project_root.resolve() / "configs/gate_a_llm.yaml")
    limit = float(task.get("budget_limit_usd", 8.0))
    ledger = BudgetLedger.open_or_upgrade(task_root / "budget.json", limit)
    provider = ZhipuProvider(config, select_live_models(config), ledger, task_root / "research/responses")
    prompt = (
        "Convert the submitted research direction into a machine-checkable experimental contract for the currently supported "
        "supervised image-classification runtime. Preserve every explicit baseline, intervention, metric scope, guardrail, "
        "repeat count, threshold, and ablation. Mark unsupported if it requires segmentation, WSI, survival, or multimodal inputs. "
        "implementation_signals must be short Python identifier tokens that should genuinely occur in an implementation; do not "
        "use dataset names or generic words such as model/train/loss. A missing success threshold is represented by "
        "has_improvement_threshold=false and minimum_improvement_delta=0.\n\n"
        f"DIRECTION:\n{direction}\n\nREVISION FEEDBACK:\n{revision_feedback or '(none)'}\n\n"
        f"DATASET FACTS:\n{json.dumps({'classes': profile.get('classes'), 'split_counts': profile.get('split_counts'), 'recommended_metrics': profile.get('recommended_metrics')}, ensure_ascii=False)}"
    )
    fingerprint = hashlib.sha256((direction + "\n" + revision_feedback).encode("utf-8")).hexdigest()[:16]
    extraction, _ = provider.call_json(
        "ideation", f"{task['task_id']}-research-contract-v1-{fingerprint}",
        "You are a scientific protocol designer. Return only the requested JSON object.",
        prompt, "research_contract_extraction", extraction_schema(),
    )
    return contract_from_extraction(
        direction, profile, extraction, split_seed=split_seed, revision_feedback=revision_feedback
    )


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _queries(direction: str) -> list[str]:
    english = re.sub(r"[^A-Za-z0-9 -]+", " ", direction).strip()
    queries = [
        "PathMNIST histopathology image classification benchmark",
        "adaptive resolution dynamic routing computational pathology",
        "conditional computation image classification early exit",
    ]
    if len(english.split()) >= 3:
        queries.insert(0, english[:180])
    return queries[:4]


def prepare_research(
    state_root: Path,
    task_id: str,
    *,
    search: Callable[[str], list[dict]] = search_verified_literature,
    revision_feedback: str = "",
    project_root: Path | None = None,
) -> dict:
    root = state_root.resolve() / task_id
    task_path = root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("schema_version") != 2 or task.get("stages", {}).get("dataset_validated") != "completed":
        raise RuntimeError("Research preparation requires a validated research task")
    direction = str(task.get("research_direction", "")).strip()
    if not direction:
        raise RuntimeError("Research direction is empty")
    now = datetime.now(timezone.utc).isoformat()
    profile = json.loads((root / "dataset/dataset_profile.json").read_text(encoding="utf-8"))

    understanding = {
        "schema_version": 1,
        "research_direction": direction,
        "research_question": direction,
        "claim_boundary": "benchmark patch classification only; no clinical or patient-level claims",
        "primary_metrics": profile.get("recommended_metrics", ["macro_f1", "accuracy"]),
        "created_at": now,
    }
    _write(root / "research/research_understanding.json", understanding)

    results: list[dict] = []
    failures: list[dict] = []
    seen: set[str] = set()
    rejected: list[dict] = []
    for query in _queries(direction):
        try:
            papers = search(query)
        except LiteratureError as exc:
            failures.append({"query": query, "error": str(exc)})
            continue
        accepted, rejected_for_query = filter_relevant_literature(papers)
        rejected.extend({"query": query, **paper} for paper in rejected_for_query)
        for paper in accepted:
            key = canonical_reference_key(paper)
            if not key or key in seen:
                rejected.append({"query": query, **paper, "relevance_status": "rejected", "rejection_reason": "duplicate_or_version"})
                continue
            seen.add(key)
            results.append({"query": query, **paper})
    literature = {
        "schema_version": 2,
        "status": "verified" if results else "unavailable",
        "references": results[:20],
        "failures": failures,
        "relevance_report": {
            "directly_relevant": [item["title"] for item in results if item.get("relevance_status") == "directly_relevant"],
            "background_only": [item["title"] for item in results if item.get("relevance_status") == "background_only"],
            "rejected": rejected,
        },
        "created_at": now,
    }
    _write(root / "research/literature.json", literature)
    if not results:
        raise RuntimeError(
            "No literature reference was verified; the task may continue as an experiment, "
            "but it cannot enter the formal-paper workflow"
        )

    idea = {
        "schema_version": 1,
        "title": "Agent-designed pathology image classification study",
        "hypothesis": direction,
        "novelty_context": [reference["title"] for reference in results[:8]],
        "claim_boundary": understanding["claim_boundary"],
        "created_at": now,
    }
    contract = _generate_contract(project_root, root, task, direction, profile, revision_feedback)
    write_contract(root, contract)
    experiment_spec = {
        "schema_version": 2,
        "dataset_sha256": profile.get("content_sha256"),
        "allowed_splits": ["train", "validation"],
        "sealed_split": "test",
        "selection_policy": "select on validation, freeze code, evaluate test once",
        "primary_metric": contract["metrics"]["primary"]["name"],
        "required_metrics": [
            contract["metrics"]["primary"]["name"],
            *[item["name"] for item in contract["metrics"].get("guardrails", [])],
            *[item["name"] for item in contract["metrics"].get("secondary", [])],
        ],
        "research_contract": "research/research_contract.json",
        "research_contract_sha256": contract["contract_sha256"],
        "research_direction": direction,
        "created_at": now,
    }
    _write(root / "research/idea.json", idea)
    _write(root / "research/experiment_spec.json", experiment_spec)
    for stage in (
        "research_understood",
        "literature_collected",
        "idea_proposed",
        "research_contract_generated",
    ):
        task["stages"][stage] = "completed"
    task["completed_stage"] = "research_contract_generated"
    task["control"] = "paused"
    task["updated_at"] = now
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "task_id": task_id,
        "completed_stage": "research_contract_generated",
        "contract_supported": contract["capability"]["supported"],
        "contract": str(root / "research/research_contract.json"),
        "verified_references": len(results),
        "failed_queries": len(failures),
    }


def regenerate_contract(state_root: Path, task_id: str, revision_feedback: str, *, project_root: Path | None = None) -> dict:
    root = state_root.resolve() / task_id
    task_path = root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("schema_version") != 2 or task.get("completed_stage") not in {
        "research_contract_generated", "experiment_spec_validated", "sandbox_prechecked"
    }:
        raise RuntimeError("Only a generated, not-yet-executed contract can be revised")
    profile = json.loads((root / "dataset/dataset_profile.json").read_text(encoding="utf-8"))
    previous_path = root / "research/research_contract.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    contract = _generate_contract(
        project_root, root, task, str(task.get("research_direction", "")), profile, revision_feedback
    )
    contract["contract_revision"] = int(previous.get("contract_revision", 1)) + 1
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = contract_sha256(contract)
    write_contract(root, contract)
    approval = root / "research/research_contract_approval.json"
    approval.unlink(missing_ok=True)
    spec_path = root / "research/experiment_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["primary_metric"] = contract["metrics"]["primary"]["name"]
    spec["research_contract_sha256"] = contract["contract_sha256"]
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    task["stages"]["research_contract_generated"] = "completed"
    task["stages"]["research_contract_approved"] = "waiting"
    task["stages"]["experiment_spec_validated"] = "waiting"
    task["stages"]["sandbox_prechecked"] = "waiting"
    task["completed_stage"] = "research_contract_generated"
    task["control"] = "paused"
    task["updated_at"] = datetime.now(timezone.utc).isoformat()
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"task_id": task_id, "contract": str(root / "research/research_contract.json"), "contract_sha256": contract["contract_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare verified research evidence")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_research(args.state_root, args.task_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
