from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from .experiment_contract import ExperimentResult
from .research_contract import code_semantic_evidence, load_contract


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _metric_dict(node: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    value = getattr(getattr(node, "metric", None), "value", None)
    if isinstance(value, dict) and isinstance(value.get("metric_names"), list):
        for metric in value["metric_names"]:
            name = _slug(str(metric.get("metric_name", "metric")))
            data = metric.get("data") or []
            if data and isinstance(data[0].get("final_value"), (int, float)):
                metrics[name] = float(data[0]["final_value"])
    elif isinstance(value, (int, float)):
        metrics["primary_metric"] = float(value)
    elif isinstance(value, dict):
        metrics.update({str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))})
    return metrics


def _augment_compute_metrics(metrics: dict[str, Any], data_path: Path) -> None:
    if not data_path.is_file():
        return
    try:
        import numpy as np

        # Object-array NPY files are executable pickle containers. They are
        # accepted only as legacy, agent-generated files inside this task's own
        # experiment workspace; external datasets never pass through here.
        payload = np.load(data_path, allow_pickle=True).item()
        if not isinstance(payload, dict) or not payload:
            return
        def scalar(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return value.item()
                if value.size == 1:
                    return value.reshape(-1)[0].item()
            return value

        def walk(value: Any):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield str(key), scalar(item)
                    yield from walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from walk(item)

        # Generated programs commonly nest evidence under method/dataset keys.
        # Preserve only scalar audit/compute fields; ignore predictions and plots.
        aliases = {
            "high_resolution_fraction": "high_resolution_fraction",
            "high_res_fraction": "high_resolution_fraction",
            "executed_high_resolution_samples": "executed_high_resolution_samples",
            "executed_low_resolution_samples": "executed_low_resolution_samples",
            "both_branches_executed_samples": "both_branches_executed_samples",
            "validation_inference_seconds": "validation_inference_seconds",
            "dynamic_inference_seconds": "dynamic_inference_seconds",
            "fixed_low_inference_seconds": "fixed_low_inference_seconds",
            "fixed_high_inference_seconds": "fixed_high_inference_seconds",
        }
        for key, value in walk(payload):
            target = aliases.get(key.casefold())
            if target and isinstance(value, (int, float)):
                metrics.setdefault(target, float(value))
        dataset = next(iter(payload.values()))
        if not isinstance(dataset, dict):
            return
        routing = dataset.get("routing_ratio")
        if routing is None:
            routing = dataset.get("upgrade_ratio")
        if isinstance(routing, np.ndarray):
            routing = routing.tolist()
        if isinstance(routing, (list, tuple)) and len(routing) > 0:
            last = float(routing[-1])
            # Agents commonly record either high-resolution fraction directly or
            # expected relative FLOPs with low=1/high=4. Preserve the source and
            # derive upgrade ratio only when the latter convention is evident.
            metrics.setdefault("routing_statistic", last)
            if 1.0 <= last <= 4.0:
                metrics.setdefault("upgrade_ratio", max(0.0, min(1.0, (last - 1.0) / 3.0)))
            elif 0.0 <= last <= 1.0:
                metrics.setdefault("upgrade_ratio", last)
        avg_flops = dataset.get("avg_flops")
        if isinstance(avg_flops, dict):
            values = avg_flops.get("val")
            if values is None:
                values = avg_flops.get("validation")
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if isinstance(values, (list, tuple)) and len(values) > 0:
                metrics.setdefault("average_flops", float(values[-1]))
    except Exception:
        return


def _select_complete_result(source_dir: Path, code_hash: str) -> tuple[dict[str, Any] | None, Path | None]:
    """Select host-normalized per-sample evidence, never a scalar upstream summary."""
    candidates = [
        source_dir / "working/experiment_result.json",
        source_dir / "experiment_result.json",
    ]
    if source_dir.is_dir():
        candidates.extend(source_dir.rglob("experiment_result.json"))
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        recorded_hash = payload.get("code_sha256")
        if recorded_hash and recorded_hash != code_hash:
            continue
        if isinstance(payload.get("metrics"), dict) and all(
            isinstance(payload.get(field), list)
            for field in ("predictions", "targets", "sample_ids")
        ):
            return payload, path
    return None, None


def export_journals(project_root: Path, state_root: Path, task_id: str) -> dict[str, Any]:
    project_root, state_root = project_root.resolve(), state_root.resolve()
    vendor = project_root / "vendor/AI-Scientist-v2"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    task_root = state_root / task_id
    contract = load_contract(task_root, require_approved=True)
    checkpoint = task_root / "experiment_logs/manager.pkl"
    with checkpoint.open("rb") as handle:
        state = pickle.load(handle)
    journals = state["journals"]
    output_root = task_root / "experiment_logs/results"
    output_root.mkdir(parents=True, exist_ok=True)
    exported = []
    seen_node_ids = set()
    stage_summary: dict[str, dict[str, int]] = {}
    for stage_name, journal in journals.items():
        successful = 0
        for node in journal.nodes:
            if node.is_buggy is not False:
                continue
            if node.id in seen_node_ids:
                continue
            seen_node_ids.add(node.id)
            metrics = _metric_dict(node)
            if not metrics:
                continue
            export_id = f"{_slug(stage_name)}__{node.id}"
            node_dir = output_root / export_id
            node_dir.mkdir(parents=True, exist_ok=True)
            code_path = node_dir / "run.py"
            code_path.write_text(node.code, encoding="utf-8")
            code_hash = hashlib.sha256(node.code.encode("utf-8")).hexdigest()
            source_dir = None
            if node.exp_results_dir:
                candidate = Path(node.exp_results_dir)
                source_dir = candidate if candidate.is_absolute() else project_root / candidate
            data_path = source_dir / "experiment_data.npy" if source_dir else Path()
            # Prefer the generated program's explicit result contract when it
            # exists. Upstream metric aggregation may retain only a subset of
            # arbitrary audit fields such as route counts and measured latency.
            raw_result = None
            raw_result_path = None
            if source_dir:
                raw_result, raw_result_path = _select_complete_result(source_dir, code_hash)
            evidence_dir = task_root / "experiment_logs/evidence" / code_hash
            if evidence_dir.is_dir():
                raw_result, raw_result_path = _select_complete_result(evidence_dir, code_hash)
            if raw_result is None:
                # Checkpoints can retain a container-internal ``exp_results_dir``
                # that is not meaningful in a later export container. Recover
                # only evidence whose recorded code hash matches this node.
                raw_result, raw_result_path = _select_complete_result(
                    task_root / "experiment_workspace", code_hash
                )
            if raw_result:
                metrics.update(
                    {str(key): value for key, value in raw_result["metrics"].items()}
                )
            _augment_compute_metrics(metrics, data_path)
            if source_dir and source_dir.is_dir():
                for artifact in source_dir.iterdir():
                    if artifact.is_file() and artifact.name != "experiment_code.py":
                        shutil.copy2(artifact, node_dir / artifact.name)
            if raw_result_path is not None:
                for name in ("dataset_execution_receipt.json", "trusted_metrics.json", "metric_provenance.json", "model_checkpoint.pt"):
                    source = raw_result_path.parent / name
                    if source.is_file():
                        shutil.copy2(source, node_dir / name)
            manifest_snapshot = (
                task_root
                / "experiment_logs/generated_code"
                / f"{code_hash}.manifest.json"
            )
            if manifest_snapshot.is_file():
                shutil.copy2(manifest_snapshot, node_dir / "experiment_manifest.json")
            method = node.ablation_name or node.hyperparam_name or f"agent_{stage_name}"
            manifest_value = {}
            manifest_file = node_dir / "experiment_manifest.json"
            if manifest_file.is_file():
                try:
                    manifest_value = json.loads(manifest_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError):
                    manifest_value = {}
            injected_seed = re.search(r"# Set random seed[\s\S]{0,240}?\bseed\s*=\s*(\d+)", node.code)
            if injected_seed:
                seed = int(injected_seed.group(1))
            else:
                seed = int(raw_result.get("seed", manifest_value.get("seed", 7))) if raw_result else int(manifest_value.get("seed", 7))
            if str(stage_name).startswith(("1_", "2_")):
                role_id = "baseline"
                signals: list[str] = []
            elif str(stage_name).startswith("3_") or "hard_routing_repair" in str(stage_name):
                role_id = "proposed_method"
                signals = list(contract["interventions"][0].get("implementation_signals", []))
            else:
                role_id = "component_ablation"
                signals = []
            semantic = code_semantic_evidence(node.code, signals)
            if role_id == "proposed_method" and not semantic["passed"]:
                role_id = "unbound_exploration"
            binding = {
                "schema_version": 1,
                "recorded_by": "host-exporter",
                "contract_sha256": contract["contract_sha256"],
                "role_id": role_id,
                "comparison_id": "baseline_vs_proposed" if role_id in {"baseline", "proposed_method"} else None,
                "seed": seed,
                "code_sha256": code_hash,
                "semantic_evidence": semantic,
                "upstream_stage": stage_name,
                "upstream_node_id": node.id,
                "is_seed_node": bool(getattr(node, "is_seed_node", False)),
                "is_seed_aggregation": bool(getattr(node, "is_seed_agg_node", False)),
                "execution_controls": {
                    key: manifest_value.get(key)
                    for key in (
                        "dataset",
                        "model",
                        "optimizer",
                        "learning_rate",
                        "epochs",
                        "max_epochs",
                        "early_stopping",
                        "batch_size",
                        "input_resolutions",
                        "selection_metric",
                    ) if key in manifest_value
                },
            }
            (node_dir / "contract_execution.json").write_text(
                json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            result = ExperimentResult(
                method_name=str(method),
                parent_experiment_id=node.parent.id if node.parent else None,
                code_sha256=code_hash,
                seed=seed,
                split="validation",
                metrics=metrics,
                resource_usage={"elapsed_seconds": float(node.exec_time or 0.0)},
                artifacts={
                    "code": "run.py",
                    "experiment_data": "experiment_data.npy",
                    "stage": stage_name,
                    "upstream_node_id": node.id,
                    "contract_execution": "contract_execution.json",
                },
                test_data_accessed=False,
                predictions=raw_result.get("predictions") if raw_result else None,
                probabilities=raw_result.get("probabilities") if raw_result else None,
                targets=raw_result.get("targets") if raw_result else None,
                sample_ids=raw_result.get("sample_ids") if raw_result else None,
                class_names=raw_result.get("class_names") if raw_result else None,
            )
            result_path = result.write(node_dir / "experiment_result.json")
            exported.append(
                {
                    "experiment_id": export_id,
                    "upstream_node_id": node.id,
                    "stage": stage_name,
                    "method_name": method,
                    "result": str(result_path.relative_to(task_root)),
                    "metrics": metrics,
                    "code_sha256": code_hash,
                    "contract_role": role_id,
                    "seed": seed,
                }
            )
            successful += 1
        stage_summary[stage_name] = {
            "nodes": len(journal.nodes),
            "successful_exported": successful,
            "buggy": len(journal.buggy_nodes),
        }
    manifest = {
        "schema_version": 2,
        "task_id": task_id,
        "checkpoint": str(checkpoint.relative_to(task_root)),
        "stages": stage_summary,
        "experiments": exported,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    print(json.dumps(export_journals(args.project_root, args.state_root, args.task_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
