from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .autonomous_acceptance import require_task


class FigureError(RuntimeError):
    pass


@dataclass(frozen=True)
class FigureSpec:
    id: str
    type: str
    path: str
    title: str
    source_artifacts: list[str]
    source_fields: list[str]
    evidence_category: str
    generator: str = "pathmnist-template-v1"
    script_sha256: str = ""
    sha256: str = ""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FigureError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise FigureError("Template figures require matplotlib") from exc
    return plt


def _save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 0.7), 4.2))
    ax.bar(range(len(values)), values, color="#4472C4")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_confusion(path: Path, matrix: list[list[float]], labels: list[str]) -> None:
    plt = _pyplot()
    normalized = []
    for row in matrix:
        total = float(sum(row))
        normalized.append([float(value) / total if total else 0.0 for value in row])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized confusion matrix")
    fig.colorbar(image, ax=ax, label="Row proportion")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_history(path: Path, history: list[dict[str, Any]]) -> None:
    plt = _pyplot()
    fields = [field for field in ("loss", "validation_loss", "accuracy", "macro_f1") if any(isinstance(row.get(field), (int, float)) for row in history)]
    if not fields:
        raise FigureError("Training history has no supported numeric series")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(history)]
    for field in fields:
        ax.plot(epochs, [row.get(field, float("nan")) for row in history], label=field.replace("_", " "))
    ax.set_xlabel("Epoch")
    ax.set_title("Training and validation history")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_per_class(path: Path, matrix: list[list[float]], labels: list[str]) -> None:
    plt = _pyplot()
    precision, recall, f1 = [], [], []
    for index, row in enumerate(matrix):
        tp = float(row[index])
        predicted = float(sum(value[index] for value in matrix))
        actual = float(sum(row))
        p = tp / predicted if predicted else 0.0
        r = tp / actual if actual else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.8), 4.4))
    width = 0.25
    ax.bar([x - width for x in positions], precision, width, label="precision")
    ax.bar(positions, recall, width, label="recall")
    ax.bar([x + width for x in positions], f1, width, label="F1")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_seed_scores(path: Path, per_seed: list[dict[str, Any]]) -> None:
    fields = [field for field in ("accuracy", "macro_f1") if any(isinstance(row.get(field), (int, float)) for row in per_seed)]
    if not fields:
        raise FigureError("Per-seed evidence has no supported score")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for field in fields:
        values = [float(row[field]) for row in per_seed if isinstance(row.get(field), (int, float))]
        ax.scatter([field.replace("_", " ")] * len(values), values, label=field.replace("_", " "))
        ax.errorbar(field.replace("_", " "), sum(values) / len(values), yerr=(max(values) - min(values)) / 2 if len(values) > 1 else 0, fmt="o", color="black")
    ax.set_ylim(0, 1)
    ax.set_title("Independent-seed score distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_calibration(path: Path, probabilities: list[list[float]], targets: list[int]) -> None:
    if len(probabilities) != len(targets) or not probabilities:
        raise FigureError("Calibration evidence is missing or misaligned")
    bins: list[list[float]] = [[] for _ in range(10)]
    outcomes: list[list[float]] = [[] for _ in range(10)]
    for scores, target in zip(probabilities, targets, strict=True):
        confidence = max(float(value) for value in scores)
        predicted = max(range(len(scores)), key=lambda index: scores[index])
        index = min(9, int(confidence * 10))
        bins[index].append(confidence)
        outcomes[index].append(float(predicted == target))
    points = [(sum(conf) / len(conf), sum(correct) / len(correct)) for conf, correct in zip(bins, outcomes, strict=True) if conf]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label="model")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Observed accuracy", title="Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_roc_pr(path: Path, probabilities: list[list[float]], targets: list[int]) -> None:
    if len(probabilities) != len(targets) or not probabilities:
        raise FigureError("ROC/PR evidence is missing or misaligned")
    pairs = []
    for scores, target in zip(probabilities, targets, strict=True):
        for class_index, score in enumerate(scores):
            pairs.append((float(score), int(class_index == target)))
    pairs.sort(reverse=True)
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise FigureError("ROC/PR requires both positive and negative labels")
    tp = fp = 0
    roc_x, roc_y, recall, precision = [0.0], [0.0], [0.0], [1.0]
    for _, label in pairs:
        tp += label
        fp += 1 - label
        roc_x.append(fp / negatives)
        roc_y.append(tp / positives)
        recall.append(tp / positives)
        precision.append(tp / (tp + fp))
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].plot(roc_x, roc_y)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Micro-averaged ROC")
    axes[1].plot(recall, precision)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Micro-averaged precision-recall")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _labels(profile: dict[str, Any], size: int) -> list[str]:
    mapping = profile.get("label_mapping")
    if isinstance(mapping, dict):
        reverse = {int(value): str(key) for key, value in mapping.items() if isinstance(value, int)}
        if reverse:
            return [reverse.get(index, str(index)) for index in range(size)]
    classes = profile.get("classes")
    if isinstance(classes, list) and len(classes) == size:
        return [str(value) for value in classes]
    return [str(index) for index in range(size)]


def _score_comparison(metrics: dict[str, Any]) -> dict[str, float]:
    """Return explicit scalar classification scores without interpreting them."""
    supported = ("accuracy", "macro_f1", "f1_score")
    values = {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
        and any(token in str(key).casefold() for token in supported)
        and "improvement" not in str(key).casefold()
    }
    final = {key: value for key, value in values.items() if "final" in key.casefold()}
    return final or values


def generate_template_figures(task_root: Path) -> dict[str, Any]:
    task_root = task_root.resolve()
    require_task(task_root, "analysis_completed")
    task_path = task_root / "task.json"
    task = _read(task_path)
    output_root = task_root / "paper/figures_generated"
    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    profile = _read(task_root / "dataset/dataset_profile.json")
    validation = _read(task_root / "candidate_frozen/validation_result.json")
    test = _read(task_root / "final_evaluation/experiment_result.json")
    comparison_path = task_root / "final_evaluation/comparison_results.json"
    contract_comparison = _read(comparison_path) if comparison_path.is_file() else None
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    generated: list[FigureSpec] = []

    def record(kind: str, title: str, sources: list[str], fields: list[str], render, *, evidence_category: str = "experimental_result") -> None:
        figure_id = f"fig_{kind}"
        relative = f"paper/figures_generated/figures/{kind}.png"
        planned.append({"id": figure_id, "type": kind, "title": title, "source_artifacts": sources, "source_fields": fields})
        path = task_root / relative
        try:
            render(path)
        except (FigureError, TypeError, ValueError) as exc:
            skipped.append({"type": kind, "reason": str(exc)})
            return
        generated.append(FigureSpec(figure_id, kind, relative, title, sources, fields, evidence_category, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sha256=_sha256(path)))

    split_counts = profile.get("split_counts")
    if isinstance(split_counts, dict) and split_counts:
        record("dataset_splits", "Dataset split sizes", ["dataset/dataset_profile.json"], ["split_counts"], lambda path: _save_bar(path, [str(key) for key in split_counts], [float(value) for value in split_counts.values()], "Dataset split sizes", "Samples"), evidence_category="dataset_description")

    class_counts = profile.get("class_counts")
    if isinstance(class_counts, dict) and class_counts:
        totals: dict[str, float] = {}
        for counts in class_counts.values():
            if isinstance(counts, dict):
                for label, value in counts.items():
                    totals[str(label)] = totals.get(str(label), 0.0) + float(value)
        if totals:
            record("class_distribution", "Class distribution", ["dataset/dataset_profile.json"], ["class_counts"], lambda path: _save_bar(path, list(totals), list(totals.values()), "Class distribution", "Samples"), evidence_category="dataset_description")

    history = validation.get("training_history")
    if isinstance(history, list) and history:
        record("training_history", "Training and validation history", ["candidate_frozen/validation_result.json"], ["training_history"], lambda path: _save_history(path, history))

    metrics = test.get("metrics", {})
    matrix = metrics.get("confusion_matrix") or metrics.get("aggregate_confusion_matrix")
    if isinstance(matrix, list) and matrix and all(isinstance(row, list) for row in matrix):
        labels = _labels(profile, len(matrix))
        record("confusion_matrix", "Normalized confusion matrix", ["final_evaluation/integrity/trusted_metrics.json", "dataset/dataset_profile.json"], ["metrics.confusion_matrix", "label_mapping"], lambda path: _save_confusion(path, matrix, labels))
        record("per_class_metrics", "Per-class precision, recall, and F1", ["final_evaluation/integrity/trusted_metrics.json", "dataset/dataset_profile.json"], ["metrics.confusion_matrix", "label_mapping"], lambda path: _save_per_class(path, matrix, labels))

    per_seed = test.get("per_seed") or metrics.get("per_seed")
    if isinstance(per_seed, list) and per_seed:
        record("seed_distribution", "Independent-seed score distribution", ["final_evaluation/experiment_result.json"], ["per_seed"], lambda path: _save_seed_scores(path, per_seed))

    probabilities = test.get("probabilities")
    targets = test.get("targets")
    if isinstance(probabilities, list) and isinstance(targets, list) and probabilities and len(probabilities) == len(targets):
        record("calibration", "Reliability diagram", ["final_evaluation/experiment_result.json"], ["probabilities", "targets"], lambda path: _save_calibration(path, probabilities, targets))
        record("roc_pr", "Micro-averaged ROC and precision-recall", ["final_evaluation/experiment_result.json"], ["probabilities", "targets"], lambda path: _save_roc_pr(path, probabilities, targets))
    else:
        skipped.append({"type": "roc_pr_calibration", "reason": "aligned per-sample probabilities and targets are unavailable"})

    comparison = _score_comparison(metrics)
    if comparison:
        record("test_metrics", "One-time test metrics", ["final_evaluation/integrity/trusted_metrics.json"], [f"metrics.{key}" for key in comparison], lambda path: _save_bar(path, [key.replace("_", " ") for key in comparison], [float(value) for value in comparison.values()], "One-time test metrics", "Score"))

    if isinstance(contract_comparison, dict):
        statistics = contract_comparison.get("statistics")
        if isinstance(statistics, dict) and all(isinstance(statistics.get(key), (int, float)) for key in ("mean_baseline", "mean_intervention")):
            metric_name = str(contract_comparison.get("primary_metric", "primary metric"))
            record(
                "contract_comparison",
                "Pre-specified baseline and intervention comparison",
                ["final_evaluation/comparison_results.json", "research/research_contract.json"],
                ["statistics.mean_baseline", "statistics.mean_intervention", "statistics.confidence_interval_95"],
                lambda path: _save_bar(
                    path, ["baseline", "proposed method"],
                    [float(statistics["mean_baseline"]), float(statistics["mean_intervention"])],
                    f"Held-out {metric_name} across frozen repeats", metric_name,
                ),
            )

    plan = {"schema_version": 1, "generator": "pathmnist-template-v1", "planned": planned, "skipped": skipped, "llm_extension": {"enabled": False, "upstream_component": "ai_scientist.perform_plotting", "reason": "optional; deterministic templates are the beta baseline"}}
    manifest = {
        "schema_version": 1,
        "generated_at": task.get("created_at") or "deterministic-fixture",
        "figures": [asdict(item) for item in generated],
        "skipped": skipped,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figure_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not generated:
        raise FigureError("No evidence-backed template figure could be generated")
    task["stages"]["figures_generated"] = "completed"
    task["completed_stage"] = "figures_generated"
    task["control"] = "paused"
    task["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = task_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(task_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evidence-backed pathology figures")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    print(json.dumps(generate_template_figures(args.state_root / args.task_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
