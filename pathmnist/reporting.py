from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev


class ReportError(ValueError):
    pass


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReportError(f"Expected JSON object: {path}")
    return payload


def _best_macro_f1(payload: dict) -> float:
    return float(payload["epochs"][payload["best_epoch"] - 1]["macro_f1"])


def _variant_table(root: Path, phase: str, variants: tuple[str, ...]) -> list[dict]:
    rows = []
    for variant in variants:
        directory = root / phase / variant
        run_paths = sorted(directory.glob("seed_*/run.json"))
        if len(run_paths) != 3:
            raise ReportError(f"Expected three runs for {directory}")
        scores = [_best_macro_f1(_load(path)) for path in run_paths]
        rows.append(
            {
                "phase": phase,
                "variant": variant,
                "runs": len(run_paths),
                "macro_f1_mean": mean(scores),
                "macro_f1_std": stdev(scores),
            }
        )
    return rows


def _markdown_table(rows: list[dict], highlight: str | None = None) -> list[str]:
    lines = [
        "| Phase | Variant | Runs | Macro-F1 mean | Macro-F1 std |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        name = row["variant"]
        if highlight and name == highlight and row["phase"] == "main":
            name = f"**{name}**"
        lines.append(
            f"| {row['phase']} | {name} | {row['runs']} | "
            f"{row['macro_f1_mean']:.6f} | {row['macro_f1_std']:.6f} |"
        )
    return lines


def generate_m4_report(
    run_root: Path,
    candidate_path: Path,
    output_path: Path,
) -> Path:
    final_test = _load(run_root / "test_evaluation.json")
    candidate = _load(candidate_path)
    if final_test.get("evaluation_count") != 1:
        raise ReportError("Final report requires exactly one test evaluation")
    if final_test.get("split") != "test":
        raise ReportError("Final test evaluation has an unexpected split")
    tuning = _load(run_root / "tuning" / "result.json")

    rows = _variant_table(
        run_root,
        "main",
        ("baseline", "augmentation", "optimization", "multiscale", "combined"),
    )
    rows.extend(
        _variant_table(
            run_root,
            "ablations",
            ("combined_no_augmentation", "combined_no_multiscale"),
        )
    )
    test_scores = [float(item["macro_f1"]) for item in final_test["per_seed"]]
    test_mean = mean(test_scores)
    test_std = stdev(test_scores)
    validation = next(row for row in rows if row["phase"] == "main" and row["variant"] == "optimization")
    confusion = final_test["aggregate_confusion_matrix"]
    class_recall = [row[index] / sum(row) for index, row in enumerate(confusion)]
    worst_classes = sorted(range(len(class_recall)), key=lambda index: class_recall[index])[:3]

    lines = [
        "# PathMNIST M4 Final Report",
        "",
        "## Scope and discipline",
        "",
        "- Training and tuning used train/validation splits only.",
        "- The frozen candidate was evaluated on the test split exactly once.",
        "- Test results were not used for tuning, model selection, or retraining.",
        f"- Dataset SHA-256: `{candidate['dataset_sha256']}`.",
        "",
        "## Frozen candidate",
        "",
        f"- Model: `{candidate['model']}`",
        f"- Variant: `{candidate['variant']}`",
        "- Seeds: `7, 17, 27`",
        f"- Learning rate: `{candidate['hyperparameters']['learning_rate']:g}`",
        f"- Weight decay: `{candidate['hyperparameters']['weight_decay']:g}`",
        f"- OneCycle: `{str(candidate['hyperparameters']['one_cycle']).lower()}`",
        f"- Label smoothing: `{candidate['hyperparameters']['label_smoothing']:g}`",
        f"- Augmentation: `{str(candidate['hyperparameters']['augmentation']).lower()}`",
        f"- Multiscale: `{str(candidate['hyperparameters']['multiscale']).lower()}`",
        "",
        "## Tuning",
        "",
        "- Six pre-freeze grid settings were completed.",
        f"- Best learning rate: `{tuning['best']['learning_rate']:g}`",
        f"- Best weight decay: `{tuning['best']['weight_decay']:g}`",
        f"- Best validation Macro-F1: `{tuning['best']['macro_f1']:.6f}`",
        "",
        "## Train/validation results",
        "",
        *_markdown_table(rows, highlight="optimization"),
        "",
        "## Final one-time test result",
        "",
        f"- Test Macro-F1 mean: `{test_mean:.6f}`",
        f"- Test Macro-F1 std: `{test_std:.6f}`",
        f"- Validation-to-test Macro-F1 gap: `{validation['macro_f1_mean'] - test_mean:.6f}`",
        f"- Evaluation count: `{final_test['evaluation_count']}`",
        "",
        "| Seed | Best epoch | Accuracy | Macro-F1 | Loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item['seed']} | {item['best_epoch']} | {item['accuracy']:.6f} | "
        f"{item['macro_f1']:.6f} | {item['loss']:.6f} |"
        for item in final_test["per_seed"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The optimization-only candidate transfers from validation to test with a material gap.",
            "- Seed 27 is unstable on test and reduces the mean while increasing variance.",
            "- The three lowest-recall classes are "
            + ", ".join(f"`{index}` ({class_recall[index]:.3f})" for index in worst_classes)
            + " in the aggregate confusion matrix.",
            "- These observations are final reporting only and did not trigger further tuning.",
            "",
            "## Runtime environment",
            "",
            f"- PyTorch: `{final_test['environment']['pytorch']}`",
            f"- CUDA: `{final_test['environment']['cuda']}`",
            f"- GPU: `{final_test['environment']['device']}`",
            "",
            "## Artifacts",
        ]
    )
    artifact_paths = [
        run_root / "tuning" / "result.json",
        run_root / "final" / "optimization" / "seed_7" / "checkpoint.pt",
        run_root / "final" / "optimization" / "seed_17" / "checkpoint.pt",
        run_root / "final" / "optimization" / "seed_27" / "checkpoint.pt",
        run_root / "test_evaluation.json",
    ]
    lines.extend(f"- `{path}`" for path in artifact_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
