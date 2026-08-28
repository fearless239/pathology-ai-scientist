from __future__ import annotations

import json
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .research_stages import RESEARCH_STAGES


class AcceptanceError(RuntimeError):
    """The task cannot truthfully advance to the requested stage."""


STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "dataset_discovered": ("dataset/dataset_profile.json",),
    "dataset_validated": ("dataset/research_view/dataset_profile.json",),
    "research_understood": ("research/research_understanding.json",),
    "literature_collected": ("research/literature.json",),
    "idea_proposed": ("research/idea.json",),
    "research_contract_generated": ("research/research_contract.json",),
    "research_contract_approved": (
        "research/research_contract.json",
        "research/research_contract_approval.json",
    ),
    "experiment_spec_validated": ("research/experiment_spec.json",),
    "sandbox_prechecked": ("research/preflight.json",),
    "candidate_selected": (
        "candidates/selection.json",
        "research/semantic_review.json",
        "research/contract_fulfillment.json",
    ),
    "candidate_frozen": (
        "candidate_frozen/candidate.json",
        "candidate_frozen/run.py",
        "candidate_frozen/validation_result.json",
        "candidate_frozen/experiment_manifest.json",
        "candidate_frozen/comparison_bundle.json",
        "candidate_frozen/contract_fulfillment.json",
    ),
    "test_evaluation_approved": ("final_evaluation/approval.json",),
    "test_evaluated": (
        "final_evaluation/experiment_result.json",
        "final_evaluation/comparison_results.json",
    ),
    "analysis_completed": ("paper/analysis_completed/analysis.json",),
    "figures_generated": (
        "paper/figures_generated/figure_plan.json",
        "paper/figures_generated/figure_manifest.json",
    ),
    "paper_written": ("paper/paper_written/paper.md",),
    "review_completed": ("paper/review_completed/review.json",),
    "revision_completed": ("paper/revision_completed/final_paper.md",),
    "translation_completed": ("paper/translation_completed/translation.md",),
    "archived": ("paper/archived/archive.json",),
}


@dataclass(frozen=True)
class AcceptanceReport:
    task_id: str
    target_stage: str
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    data_integrity: str = "not_evaluated"
    sealed_test_integrity: str = "not_evaluated"
    metric_integrity: str = "not_evaluated"
    research_contract_integrity: str = "not_evaluated"
    comparison_integrity: str = "not_evaluated"
    repeat_integrity: str = "not_evaluated"
    statistical_integrity: str = "not_evaluated"
    publication_mode: str = "not_applicable"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "target_stage": self.target_stage,
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "data_integrity": self.data_integrity,
            "sealed_test_integrity": self.sealed_test_integrity,
            "metric_integrity": self.metric_integrity,
            "research_contract_integrity": self.research_contract_integrity,
            "comparison_integrity": self.comparison_integrity,
            "repeat_integrity": self.repeat_integrity,
            "statistical_integrity": self.statistical_integrity,
            "publication_mode": self.publication_mode,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise AcceptanceError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise AcceptanceError(f"JSON artifact must be an object: {path}")
    return value


def _valid_receipt(receipt: dict[str, Any], split: str) -> bool:
    return (
        receipt.get("schema_version") == 1
        and receipt.get("recorded_by") == "trusted-runner"
        and receipt.get("split") == split
        and isinstance(receipt.get("samples_consumed"), int)
        and receipt.get("samples_consumed", 0) > 0
        and isinstance(receipt.get("sample_ids_consumed"), list)
        and len(receipt["sample_ids_consumed"]) == receipt["samples_consumed"]
        and len(str(receipt.get("dataset_profile_sha256", ""))) == 64
        and len(str(receipt.get("code_sha256", ""))) == 64
    )


def validate_task(task_root: Path, target_stage: str, *, require_pdf: bool = False) -> AcceptanceReport:
    task_root = task_root.resolve()
    task = _read_json(task_root / "task.json")
    try:
        target_index = RESEARCH_STAGES.index(target_stage)
    except ValueError as exc:
        raise AcceptanceError(f"Unknown autonomous stage: {target_stage}") from exc
    required = RESEARCH_STAGES[: target_index + 1]
    errors: list[str] = []
    warnings: list[str] = []
    data_integrity = sealed_test_integrity = metric_integrity = "not_evaluated"
    research_contract_integrity = comparison_integrity = repeat_integrity = statistical_integrity = "not_evaluated"
    stages = task.get("stages")
    if not isinstance(stages, dict):
        raise AcceptanceError("task.json has no stage map")

    for stage in required:
        if stages.get(stage) == 'not_applicable' and stage in ('baseline_tuning_completed', 'ablations_completed'):
            from .research_contract import load_contract
            policy = load_contract(task_root, require_approved=True).get('experiment_policy', {})
            key = 'tuning' if stage == 'baseline_tuning_completed' else 'ablation'
            if policy.get(key) is False:
                continue
        if stages.get(stage) != "completed":
            errors.append(f"stage {stage!r} is {stages.get(stage)!r}, not completed")
        from .upstream_publication import backend, artifacts, STAGES
        relatives = STAGE_ARTIFACTS.get(stage, ())
        if backend(task) == 'upstream_v2' and stage in STAGES:
            try:
                relatives = [row['source'] for row in artifacts(task_root,stage)]
            except (OSError,ValueError,KeyError) as error:
                errors.append(f'Invalid upstream publication artifact: {error}')
                relatives = ()
        for relative in relatives:
            path = task_root / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"stage {stage!r} is missing artifact {relative}")
    if "dataset_validated" in required:
        view = task_root / "dataset/research_view"
        if not any((view / name).is_file() for name in ("dataset.npz", "manifest.json")):
            errors.append("Research view lacks dataset.npz or manifest.json")

    completed_stage = task.get("completed_stage")
    if completed_stage not in RESEARCH_STAGES:
        errors.append(f"completed_stage is invalid: {completed_stage!r}")
    elif RESEARCH_STAGES.index(completed_stage) < target_index:
        errors.append(f"completed_stage {completed_stage!r} precedes target {target_stage!r}")

    if "research_contract_approved" in required:
        from .research_contract import ResearchContractError, load_contract

        try:
            load_contract(task_root, require_approved=True)
            research_contract_integrity = "passed"
        except ResearchContractError as exc:
            research_contract_integrity = "failed"
            errors.append(str(exc))

    if "experiment_spec_validated" in required:
        spec_path = task_root / "research/experiment_spec.json"
        if spec_path.is_file():
            spec = _read_json(spec_path)
            try:
                approved = load_contract(task_root, require_approved=True)
                if spec.get("research_contract_sha256") != approved["contract_sha256"] or spec.get("primary_metric") != approved["metrics"]["primary"]["name"]:
                    errors.append("experiment specification is not bound to the approved research contract")
            except Exception:
                pass

    if "candidate_selected" in required:
        semantic_path = task_root / "research/semantic_review.json"
        if semantic_path.is_file():
            semantic = _read_json(semantic_path)
            try:
                approved_contract = load_contract(task_root, require_approved=True)
                semantic_ok = semantic.get("passed") is True and semantic.get("contract_sha256") == approved_contract["contract_sha256"]
            except Exception:
                semantic_ok = False
            if not semantic_ok:
                comparison_integrity = "failed"
                errors.append("independent implementation semantic review is invalid or did not pass")
        fulfillment_path = task_root / "research/contract_fulfillment.json"
        if not fulfillment_path.is_file():
            comparison_integrity = repeat_integrity = statistical_integrity = "failed"
            errors.append("research contract fulfillment evidence is missing")
        else:
            fulfillment = _read_json(fulfillment_path)
            checks = fulfillment.get("checks", {})
            comparison_integrity = str(checks.get("comparison_integrity", "failed"))
            repeat_integrity = str(checks.get("repeat_integrity", "failed"))
            statistical_integrity = str(checks.get("statistical_integrity", "failed"))
            if not fulfillment.get("passed"):
                errors.extend(str(item) for item in fulfillment.get("errors", []))

    if "test_evaluated" in required:
        validation_receipt = task_root / "candidate_frozen/dataset_execution_receipt.json"
        test_receipt = task_root / "final_evaluation/integrity/dataset_execution_receipt.json"
        trusted_metrics = task_root / "final_evaluation/integrity/trusted_metrics.json"
        provenance = task_root / "final_evaluation/integrity/metric_provenance.json"
        data_integrity = "passed" if validation_receipt.is_file() else "failed"
        sealed_test_integrity = "passed" if test_receipt.is_file() else "failed"
        metric_integrity = "passed" if trusted_metrics.is_file() and provenance.is_file() else "failed"
        for label, status, relative in (
            ("data integrity", data_integrity, "candidate_frozen/dataset_execution_receipt.json"),
            ("sealed test integrity", sealed_test_integrity, "final_evaluation/integrity/dataset_execution_receipt.json"),
            ("trusted metric integrity", metric_integrity, "final_evaluation/integrity/trusted_metrics.json and metric_provenance.json"),
        ):
            if status != "passed":
                errors.append(f"{label} evidence is missing: {relative}")
        if validation_receipt.is_file():
            receipt = _read_json(validation_receipt)
            if not _valid_receipt(receipt, "validation"):
                data_integrity = "failed"
                errors.append("validation data receipt is not a trusted-runner validation receipt")
        if test_receipt.is_file():
            receipt = _read_json(test_receipt)
            if not _valid_receipt(receipt, "test"):
                sealed_test_integrity = "failed"
                errors.append("sealed test receipt is not a trusted-runner test receipt")
        if trusted_metrics.is_file() and provenance.is_file():
            trusted = _read_json(trusted_metrics)
            metric_record = _read_json(provenance)
            if trusted.get("evaluator_version") != metric_record.get("evaluator_version") or not isinstance(trusted.get("metrics"), dict):
                metric_integrity = "failed"
                errors.append("trusted metric artifacts have inconsistent evaluator provenance")
        comparison_path = task_root / "final_evaluation/comparison_results.json"
        if comparison_path.is_file():
            comparison = _read_json(comparison_path)
            arms = comparison.get("arms")
            statistics = comparison.get("statistics")
            if comparison.get("attempt_count") != 1 or comparison.get("split") != "test":
                comparison_integrity = "failed"
                errors.append("held-out comparison is not a single test attempt")
            if not isinstance(arms, list):
                comparison_integrity = repeat_integrity = "failed"
                errors.append("held-out comparison has no arm evidence")
            else:
                role_seeds: dict[str, set[int]] = {"baseline": set(), "proposed_method": set()}
                for arm in arms:
                    if not isinstance(arm, dict) or arm.get("role_id") not in role_seeds or not isinstance(arm.get("seed"), int) or not isinstance(arm.get("trusted_metrics"), dict):
                        comparison_integrity = "failed"
                        errors.append("held-out comparison contains an invalid arm")
                        continue
                    role_seeds[arm["role_id"]].add(arm["seed"])
                try:
                    contract = load_contract(task_root, require_approved=True)
                    expected = set(map(int, contract["repeat_plan"]["seeds"]))
                except Exception:
                    expected = set()
                if not expected or any(seeds != expected for seeds in role_seeds.values()):
                    repeat_integrity = "failed"
                    errors.append("held-out comparison does not cover every approved arm and repeat seed")
            if not isinstance(statistics, dict) or statistics.get("n") is None:
                statistical_integrity = "failed"
                errors.append("held-out comparison lacks trusted paired statistics")

    literature_path = task_root / "research/literature.json"
    if "literature_collected" in required and literature_path.is_file():
        literature = _read_json(literature_path)
        references = literature.get("references")
        if literature.get("status") != "verified" or not isinstance(references, list) or not references:
            errors.append("formal-paper workflow requires at least one verified literature reference")
        else:
            for index, reference in enumerate(references):
                if not isinstance(reference, dict) or not reference.get("title"):
                    errors.append(f"literature reference {index} has no title")
                elif not any(reference.get(key) for key in ("doi", "corpus_id", "pmid", "url")):
                    errors.append(f"literature reference {index} has no stable identifier or URL")
                if literature.get("schema_version", 1) >= 2 and reference.get("relevance_status") not in {"directly_relevant", "background_only"}:
                    errors.append(f"literature reference {index} lacks an accepted relevance classification")

    if require_pdf:
        pdf_relatives = (
            "paper/revision_completed/final_paper.pdf",
            "paper/translation_completed/translation.pdf",
        )
        if backend(task) == 'upstream_v2':
            try:
                pdf_relatives = [artifacts(task_root,s)[0]['pdf'] for s in ('revision_completed','translation_completed')]
            except (OSError,ValueError,KeyError) as error:
                errors.append(f'Invalid publication PDFs: {error}')
                pdf_relatives = ()
        for relative in pdf_relatives:
            path = task_root / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"required final PDF is missing: {relative}")

    if target_stage == "archived":
        record_path = task_root / "paper/archived/archive.json"
        if record_path.is_file():
            record = _read_json(record_path)
            archive = task_root / str(record.get("archive", ""))
            if not archive.is_file() or archive.stat().st_size == 0:
                errors.append("archive record does not point to an existing evidence ZIP")

    figure_manifest_path = task_root / "paper/figures_generated/figure_manifest.json"
    if "figures_generated" in required and figure_manifest_path.is_file():
        manifest = _read_json(figure_manifest_path)
        figures = manifest.get("figures")
        if not isinstance(figures, list) or not figures:
            errors.append("formal-paper workflow requires at least one evidence-backed figure")
        else:
            if not any(isinstance(figure, dict) and figure.get("evidence_category") == "experimental_result" for figure in figures):
                errors.append("formal-paper workflow requires at least one experimental-result figure")
            for index, figure in enumerate(figures):
                if not isinstance(figure, dict):
                    errors.append(f"figure manifest entry {index} is invalid")
                    continue
                relative = figure.get("path")
                sources = figure.get("source_artifacts")
                if not isinstance(relative, str) or not (task_root / relative).is_file():
                    errors.append(f"figure manifest entry {index} points to a missing file")
                elif (task_root / relative).suffix.casefold() == ".png":
                    header = (task_root / relative).read_bytes()[:24]
                    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
                        errors.append(f"figure manifest entry {index} is not a decodable PNG")
                    elif int.from_bytes(header[16:20], "big") < 1 or int.from_bytes(header[20:24], "big") < 1:
                        errors.append(f"figure manifest entry {index} has invalid dimensions")
                if not isinstance(sources, list) or not sources:
                    errors.append(f"figure manifest entry {index} has no evidence source")

    paper_path = task_root / "paper/revision_completed/final_paper.md"
    if paper_path.is_file():
        text = paper_path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\b(Note:|TODO|FIXME|not exhaustively detailed)\b", text, re.I):
            warnings.append("final paper contains placeholder or missing-detail language")

    return AcceptanceReport(
        task_id=str(task.get("task_id", task_root.name)),
        target_stage=target_stage,
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        data_integrity=data_integrity,
        sealed_test_integrity=sealed_test_integrity,
        metric_integrity=metric_integrity,
        research_contract_integrity=research_contract_integrity,
        comparison_integrity=comparison_integrity,
        repeat_integrity=repeat_integrity,
        statistical_integrity=statistical_integrity,
        publication_mode=("research_paper" if not errors and "analysis_completed" in required else "failure_diagnosis" if "test_evaluated" in required else "not_applicable"),
    )


def require_task(task_root: Path, target_stage: str, *, require_pdf: bool = False) -> AcceptanceReport:
    report = validate_task(task_root, target_stage, require_pdf=require_pdf)
    if not report.passed:
        raise AcceptanceError("; ".join(report.errors))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an autonomous task stage")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--target-stage", choices=RESEARCH_STAGES, required=True)
    parser.add_argument("--require-pdf", action="store_true")
    args = parser.parse_args()
    report = validate_task(args.task_root, args.target_stage, require_pdf=args.require_pdf)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
