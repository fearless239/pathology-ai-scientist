from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gate_a.budget import BudgetLedger
from gate_a.config import load_config
from gate_a.pipeline import select_live_models
from gate_a.provider import ZhipuProvider
from .autonomous_acceptance import require_task, validate_task
from .experiment_contract import ExperimentResult
from .experiment_manifest import require_manifest
from .figures import generate_template_figures
from .paper_disclosure import ensure_disclosure
from .artifact_cache import cached_artifact
from .execution_control import task_operation
from .publication import (
    normalize_publication_language,
    publication_candidate,
    publication_dataset_profile,
    publication_integrity_summary,
    publication_contract_results,
    publication_research_contract,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _commit_stage(path, stage):
    from .research_stages import RESEARCH_STAGES
    task = _read(path)
    if RESEARCH_STAGES.index(task['completed_stage']) >= RESEARCH_STAGES.index(stage):
        return
    previous = path.read_bytes()
    task['stages'][stage] = 'completed'
    task['completed_stage'] = stage
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    try:
        require_task(path.parent, stage)
    except Exception:
        temporary.write_bytes(previous)
        temporary.replace(path)
        raise


def _paper_literature(literature: dict) -> dict:
    references = [
        reference for reference in literature.get("references", [])
        if reference.get("relevance_status") in {"directly_relevant", "background_only"}
    ]
    report = literature.get("relevance_report") or {}
    return {
        "schema_version": literature.get("schema_version"),
        "status": literature.get("status"),
        "references": references,
        "rejected_reference_count": len(report.get("rejected", [])),
    }


def _analysis_claims(candidate: dict, test_metrics: dict) -> tuple[str, list[str]]:
    selected_metric = str(candidate.get("primary_metric", "primary_metric"))
    metric = re.sub(r"^(validation|val)_", "test_", selected_metric)
    candidates = (metric, f"test_{selected_metric}", selected_metric)
    value = next((test_metrics.get(name) for name in candidates if isinstance(test_metrics.get(name), (int, float))), None)
    finding = (
        f"The final model obtained {metric}={value:.6f}; this is a descriptive "
        "one-time evaluation, not evidence of superiority."
        if isinstance(value, (int, float))
        else "The final model completed a one-time descriptive held-out test evaluation."
    )
    limitations = [
        "The held-out test result was not used for tuning or model selection.",
        "Any comparison without paired uncertainty estimates is descriptive only.",
    ]
    dynamic = test_metrics.get("dynamic_inference_seconds")
    fixed_high = test_metrics.get("fixed_high_inference_seconds")
    if isinstance(dynamic, (int, float)) and isinstance(fixed_high, (int, float)):
        relation = "not lower than" if dynamic >= fixed_high else "lower than"
        limitations.append(
            f"Measured dynamic latency ({dynamic:.6f}s) was {relation} fixed-high latency "
            f"({fixed_high:.6f}s); repeated synchronized trials are required for an efficiency claim."
        )
    return finding, limitations


def _contract_finding(fulfillment: dict, test_comparison: dict, fallback: str) -> str:
    validation = fulfillment.get("statistics")
    held_out = test_comparison.get("statistics")
    if not isinstance(validation, dict):
        return fallback
    delta = validation.get("mean_difference")
    supported = fulfillment.get("hypothesis_supported")
    if supported is True:
        conclusion = "The pre-specified validation improvement threshold and guardrails were met."
    elif supported is False:
        conclusion = "The completed experiment did not meet the pre-specified improvement threshold and is reported as a negative result."
    else:
        conclusion = "No positive-claim threshold was pre-specified, so the completed comparison is descriptive."
    held_out_text = ""
    if isinstance(held_out, dict) and isinstance(held_out.get("mean_difference"), (int, float)):
        held_out_text = f" The once-executed held-out comparison had a paired mean difference of {float(held_out['mean_difference']):.6f}."
    return f"{conclusion} The paired validation mean difference was {float(delta):.6f}.{held_out_text}" if isinstance(delta, (int, float)) else conclusion + held_out_text


@task_operation
def run_postprocess(project_root: Path, state_root: Path, task_id: str) -> dict[str, object]:
    project_root, task_root = project_root.resolve(), state_root.resolve() / task_id
    task_path = task_root / "task.json"
    task = _read(task_path)
    allowed_stages = {
        "test_evaluated", "analysis_completed", "figures_generated", "paper_written",
        "review_completed", "revision_completed", "translation_completed",
    }
    if task.get("completed_stage") not in allowed_stages:
        raise RuntimeError("Postprocess requires a completed one-time test evaluation")
    integrity_acceptance = validate_task(task_root, "test_evaluated")
    if not integrity_acceptance.passed:
        diagnosis_root = task_root / "paper/failure_diagnosis"
        diagnosis_root.mkdir(parents=True, exist_ok=True)
        diagnosis = {
            "schema_version": 1,
            "publication_mode": "failure_diagnosis",
            "errors": list(integrity_acceptance.errors),
            "warnings": list(integrity_acceptance.warnings),
            "repair_guidance": [
                "Export complete predictions, targets, and unique sample_ids from the mounted split.",
                "Recompute metrics through the trusted evaluator; never edit sealed-test outputs.",
                "Do not repeat a sealed test attempt; repair only from the already captured evidence.",
            ],
        }
        (diagnosis_root / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines = ["# Failure Diagnosis", "", "Formal paper generation was blocked by scientific-integrity checks.", "", "## Blocking errors", ""]
        lines.extend(f"- {error}" for error in integrity_acceptance.errors)
        (diagnosis_root / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        task["publication_mode"] = "failure_diagnosis"
        task["control"] = "paused"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"task_id": task_id, "completed_stage": task.get("completed_stage"), "publication_mode": "failure_diagnosis", "diagnosis": str(diagnosis_root / "diagnosis.json")}
    candidate = _read(task_root / "candidate_frozen/candidate.json")
    test_result = ExperimentResult.read(
        task_root / "final_evaluation/experiment_result.json", allow_test=True
    )
    validation = ExperimentResult.read(task_root / "candidate_frozen/validation_result.json")
    manifest = require_manifest(task_root)
    dataset_profile = _read(task_root / "dataset/dataset_profile.json")
    dataset_receipt = _read(task_root / "candidate_frozen/dataset_execution_receipt.json")
    test_receipt = _read(task_root / "final_evaluation/integrity/dataset_execution_receipt.json")
    trusted_metrics_record = _read(task_root / "final_evaluation/integrity/trusted_metrics.json")
    metric_provenance = _read(task_root / "final_evaluation/integrity/metric_provenance.json")
    literature = _read(task_root / "research/literature.json")
    research_contract = _read(task_root / "research/research_contract.json")
    fulfillment = _read(task_root / "research/contract_fulfillment.json")
    test_comparison = _read(task_root / "final_evaluation/comparison_results.json")
    evidence = {
        "schema_version": 1,
        "candidate_selection": publication_candidate(candidate),
        "dataset": publication_dataset_profile(dataset_profile),
        "experiment_manifest": manifest,
        "literature": _paper_literature(literature),
        "validation_metrics": validation.metrics,
        "test_metrics": trusted_metrics_record["metrics"],
        "research_contract": publication_research_contract(research_contract),
        "contract_results": publication_contract_results(fulfillment, test_comparison),
        "integrity": publication_integrity_summary(dataset_receipt, test_receipt, metric_provenance),
        "test_metric_semantics": "Values in test_metrics are independently recomputed held-out test results and were not used for model selection.",
    }
    (task_root / "paper/paper_evidence.json").parent.mkdir(parents=True, exist_ok=True)
    (task_root / "paper/paper_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    analysis_root = task_root / "paper/analysis_completed"
    analysis_root.mkdir(parents=True, exist_ok=True)
    finding, limitations = _analysis_claims(candidate, test_result.metrics)
    finding = _contract_finding(fulfillment, test_comparison, finding)
    analysis = {
        "schema_version": 2,
        "task_id": task_id,
        "primary_metric": candidate.get("primary_metric"),
        "finding": finding,
        "limitations": limitations,
        "evidence": evidence,
    }
    (analysis_root / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if task.get("completed_stage") == "test_evaluated":
        task["stages"]["analysis_completed"] = "completed"
        task["completed_stage"] = "analysis_completed"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if task.get("completed_stage") == "analysis_completed":
        generate_template_figures(task_root)
        task = _read(task_path)
    require_task(task_root, "figures_generated")
    figures = _read(task_root / "paper/figures_generated/figure_manifest.json")
    analysis["figures"] = figures

    config = load_config(project_root / "configs/gate_a_llm.yaml")
    ledger = BudgetLedger(task_root / "budget.json", float(task.get("budget_limit_usd", 8.0)))
    selected = select_live_models(config)
    provider = ZhipuProvider(config, selected, ledger, task_root / "research/responses")
    from .upstream_publication import backend, run
    if backend(task) == 'upstream_v2':
        return run(project_root,task_root,analysis,provider)
    context = json.dumps(analysis, ensure_ascii=False, indent=2)
    paper_prompt = (
        "Write a rigorous English research paper in Markdown for this completed pathology AI study. "
        "Use only the supplied evidence, distinguish validation from test, define every metric, and "
        "honor test_metric_semantics exactly; never claim that the sealed test was not evaluated. "
        "Use contract_results to state whether the pre-specified hypothesis was supported; a completed but unmet threshold is a valid negative result and must not be rewritten as success. "
        "explicitly report any supplied latency limitation. Include Dataset, Related Work, Methods, Experimental "
        "Protocol, Results, Discussion, Limitations, Conclusion, and References sections. Cite only "
        "the verified references using stable internal keys such as [R1]; they will be converted to standard numeric citations during export. Include the matching title, "
        "authors, year, venue, DOI/URL in References. Do not invent literature, metrics, or clinical claims.\n\n"
        "Write as a scientific manuscript, not an audit report. Do not expose hashes, artifact paths, runner/evaluator versions, timestamps, attempts, receipts, internal profile names, or workflow terms such as frozen candidate and sealed test. "
        "Describe the final model as selected on validation data and evaluated once on a held-out test set.\n\n"
        "Include useful verified figures with Markdown image syntax using exactly the paths in the figure "
        "manifest, and explain only what their source fields support.\n\n"
        + context
    )
    paper_root = task_root / "paper/paper_written"
    paper_root.mkdir(parents=True, exist_ok=True)
    paper_path = paper_root / "paper.md"
    paper = cached_artifact(paper_path, paper_prompt, lambda digest:
        provider.call_text("paper_writer", f"{task_id}-post-paper-v3-{digest[:16]}", "You are a careful scientific paper writer.", paper_prompt)[0])
    paper = ensure_disclosure(normalize_publication_language(paper), "en")
    paper_path.write_text(paper, encoding="utf-8")
    (paper_root / "paper.tex").write_text("\\documentclass{article}\n\\begin{document}\n" + paper.replace("&", "\\&") + "\n\\end{document}\n", encoding="utf-8")
    _commit_stage(task_path, 'paper_written')

    review_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 1200},
            "issues": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 600}},
            "checklist": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 400}},
        },
        "required": ["summary", "issues", "checklist"],
    }
    review_root = task_root / "paper/review_completed"
    review_root.mkdir(parents=True, exist_ok=True)
    review_path = review_root / "review.json"
    review_context = json.dumps(
        {
            "primary_metric": analysis.get("primary_metric"),
            "finding": analysis.get("finding"),
            "limitations": analysis.get("limitations"),
            "validation_metrics": evidence.get("validation_metrics"),
            "test_metrics": evidence.get("test_metrics"),
            "research_contract": evidence.get("research_contract"),
            "contract_results": evidence.get("contract_results"),
            "integrity": evidence.get("integrity"),
        },
        ensure_ascii=False,
        indent=2,
    )
    review_prompt = (
        "Review the manuscript against the compact trusted evidence below. Return a concise JSON object: "
        "one short summary, at most 8 non-overlapping actionable issues, and at most 8 checklist items. "
        "Prioritize scientific validity, task coverage, unsupported claims, missing limitations, and presentation defects. "
        "Do not restate the paper, reproduce tables, hashes, paths, or long metric lists.\n\nPAPER:\n"
        + paper
        + "\n\nCOMPACT TRUSTED EVIDENCE:\n"
        + review_context
    )
    review = json.loads(cached_artifact(review_path, review_prompt, lambda digest:
        json.dumps(provider.call_json("reviewer", f"{task_id}-post-review-v4-{digest[:16]}",
            "You are an independent scientific reviewer. Return only the requested concise JSON object.",
            review_prompt, "review", review_schema)[0], ensure_ascii=False)))
    _commit_stage(task_path, 'review_completed')

    revision_prompt = "Revise the paper using every checklist item below. Preserve verified scientific results but omit audit-only hashes, artifact paths, timestamps, receipts, and workflow terminology. Do not invent improvements. Return complete English Markdown.\n\nPAPER:\n" + paper + "\n\nREVIEW:\n" + json.dumps(review, ensure_ascii=False, indent=2)
    revision_root = task_root / "paper/revision_completed"
    revision_root.mkdir(parents=True, exist_ok=True)
    revision_path = revision_root / "final_paper.md"
    revision = cached_artifact(revision_path, revision_prompt, lambda digest:
        provider.call_text("paper_writer", f"{task_id}-post-revision-v3-{digest[:16]}", "You are a scientific revision editor.", revision_prompt)[0])
    revision = ensure_disclosure(normalize_publication_language(revision), "en")
    revision_path.write_text(revision, encoding="utf-8")
    _commit_stage(task_path, 'revision_completed')
    translation_root = task_root / "paper/translation_completed"
    translation_root.mkdir(parents=True, exist_ok=True)
    translation_path = translation_root / "translation.md"
    translation = cached_artifact(translation_path, revision, lambda digest:
        provider.call_text("paper_writer", f"{task_id}-post-translation-v3-{digest[:16]}", "You are a precise scientific translator.", "Translate this revised paper into Chinese. Preserve all scientific numbers, metric names, caveats, citations, and section structure. Do not add audit-only implementation details.\n\n" + revision)[0])
    translation = ensure_disclosure(normalize_publication_language(translation), "zh")
    translation_path.write_text(translation, encoding="utf-8")

    for stage in ("analysis_completed", "figures_generated", "paper_written", "review_completed", "revision_completed", "translation_completed"):
        task["stages"][stage] = "completed"
    task["completed_stage"] = "translation_completed"
    task["control"] = "paused"
    task["publication_mode"] = "research_paper"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "task_id": task_id,
        "completed_stage": "translation_completed",
        "next_required_action": "build and validate PDFs before archival",
        "budget": ledger.snapshot().__dict__,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_postprocess(args.project_root, args.state_root, args.task_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
