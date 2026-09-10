from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    pass


MANIFEST_NORMALIZATION_MARKER = (
    "# PATH_AI_MANIFEST_NORMALIZATION: primary_metric derived from approved contract"
)


def normalize_generated_manifest_fields(code: str, primary_metric: str) -> str:
    """Normalize unambiguous generated manifest literals before any execution.

    The approved contract is the sole source of the inserted value. A different
    selection metric or a dynamic expression remains a hard failure rather than
    being guessed or rewritten after training.
    """
    if not isinstance(primary_metric, str) or not primary_metric.strip():
        raise ManifestError("Approved primary metric is missing")
    tree = ast.parse(code)
    insertions: list[int] = []
    lines = code.splitlines(keepends=True)
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and "manifest" in target.id.casefold()
            for target in targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        fields = {
            key.value: value
            for key, value in zip(node.value.keys, node.value.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        selection = fields.get("selection_metric")
        if selection is None:
            continue
        if not isinstance(selection, ast.Constant) or selection.value != primary_metric:
            raise ManifestError(
                "Generated experiment_manifest selection_metric must be the approved "
                f"primary metric {primary_metric!r}"
            )
        declared = fields.get("primary_metric")
        if declared is not None:
            if not isinstance(declared, ast.Constant) or declared.value != primary_metric:
                raise ManifestError(
                    "Generated experiment_manifest primary_metric must equal the approved "
                    f"metric {primary_metric!r}"
                )
            continue
        start = offsets[node.value.lineno - 1] + node.value.col_offset
        if start >= len(code) or code[start] != "{":
            raise ManifestError("Cannot safely normalize generated experiment_manifest")
        insertions.append(start + 1)

    if not insertions:
        return code
    field = f'"primary_metric": {json.dumps(primary_metric)}, '
    normalized = code
    for position in sorted(insertions, reverse=True):
        normalized = normalized[:position] + field + normalized[position:]
    if MANIFEST_NORMALIZATION_MARKER not in normalized:
        normalized = MANIFEST_NORMALIZATION_MARKER + "\n" + normalized
    return normalized


def validate_generated_manifest_fields(code: str) -> None:
    """Reject a literal generated manifest missing the approved metric label."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and "manifest" in target.id.casefold()
                   for target in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys = {
            key.value for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "selection_metric" in keys and "primary_metric" not in keys:
            raise ManifestError(
                "Generated experiment_manifest is missing primary_metric; fix metadata before training"
            )


REQUIRED_FIELDS = {
    "dataset",
    "model",
    "optimizer",
    "learning_rate",
    "epochs",
    "batch_size",
    "seed",
    "input_resolutions",
    "selection_metric",
    "hardware",
}


TRAINING_POLICY_PROMPT = (
    "MANDATORY MANIFEST SCHEMA: every working/experiment_manifest.json dictionary must contain the two "
    "separate literal fields primary_metric and selection_metric, and both must name the approved research "
    "metric (e.g. accuracy), never validation_metric or validation_loss. checkpoint_selection must be "
    "{metric:'validation_loss',mode:'min'} or {metric:<approved primary metric>,mode:'max'}. "
    "It controls which epoch's weights are restored, independently of early_stopping.monitor. "
    "Preserve the baseline checkpoint_selection and early_stopping during tuning; a legacy baseline's "
    "selection_metric=validation_loss means its checkpoint_selection is validation_loss/min, NOT its research metric. "
    "Also write training_runs=[{max_epochs: configured cap, epochs: completed count}, ...] covering EVERY "
    "training launch in this execution, including search candidates and final fits; do not count epoch helpers "
    "as launches. In inference-only mode this list is empty. "
    "Export selected_parameters as a JSON object for any method search. When globals().get('PATH_AI_REPEAT') "
    "is present, skip ALL parameter searches, train exactly once using its learning_rate and parameters, and "
    "write selected_parameters exactly from PATH_AI_REPEAT['parameters']; "
    "do not redefine PATH_AI_REPEAT. Repeats change only the host-injected seed. "
    "For every new experiment manifest, write max_epochs as the configured positive integer "
    "training limit per candidate and epochs as the actual completed epoch count for the "
    "selected candidate (not its best checkpoint epoch, the maximum across candidates, or the total). "
    "Write early_stopping as either {enabled:false} or {enabled:true, "
    "monitor:'validation_loss' or 'validation_metric', mode:'min' or 'max', patience:positive integer, "
    "min_delta:nonnegative number}. Use only this patience-based stopping rule; do not add "
    "unreported convergence, divergence, or time-based success exits. Tuning must copy the "
    "baseline max_epochs and entire early_stopping policy unchanged, including when the baseline "
    "stopped early. Actual epochs may differ. Never infer a legacy baseline's limit from its "
    "completed epochs; missing policy requires regenerating baseline evidence."
)


def metric_policy(manifest, primary):
    """Resolve legacy checkpoint semantics without rewriting historical records."""
    explicit = 'primary_metric' in manifest or 'checkpoint_selection' in manifest
    if explicit:
        if manifest.get('primary_metric') != primary or manifest.get('selection_metric') != primary:
            raise ManifestError(f'primary_metric and selection_metric must equal approved metric {primary!r}')
        checkpoint = manifest.get('checkpoint_selection')
    else:
        name = manifest.get('selection_metric')
        if name not in (primary, 'validation_loss'):
            raise ManifestError(f'Ambiguous legacy selection_metric {name!r}; expected {primary!r} or validation_loss; explicit migration required')
        checkpoint = {'metric': name, 'mode': 'min' if name == 'validation_loss' else 'max'}
    if not isinstance(checkpoint, dict) or checkpoint not in (
        {'metric': primary, 'mode': 'max'}, {'metric': 'validation_loss', 'mode': 'min'}
    ):
        raise ManifestError('Invalid checkpoint_selection metric/mode')
    return {'primary_metric': primary, 'checkpoint_selection': checkpoint}


def training_policy(value: dict[str, Any]) -> dict[str, Any] | None:
    """Validate explicit controls without guessing the meaning of legacy epochs."""
    if not ({"max_epochs", "early_stopping"} & value.keys()):
        return None
    maximum = value.get("max_epochs")
    completed = value.get("epochs")
    if type(maximum) is not int or maximum < 1:
        raise ManifestError("max_epochs must be a positive integer")
    if type(completed) is not int or not 0 <= completed <= maximum:
        raise ManifestError("epochs must be the completed count between zero and max_epochs")
    stop = value.get("early_stopping")
    if not isinstance(stop, dict) or type(stop.get("enabled")) is not bool:
        raise ManifestError("early_stopping must declare enabled as a boolean")
    fields = {"enabled"}
    if stop["enabled"]:
        fields |= {"monitor", "mode", "patience", "min_delta"}
        if stop.get("monitor") not in ("validation_loss", "validation_metric"):
            raise ManifestError("Invalid early_stopping monitor")
        if stop.get("mode") not in ("min", "max"):
            raise ManifestError("Invalid early_stopping mode")
        if type(stop.get("patience")) is not int or stop["patience"] < 1:
            raise ManifestError("early_stopping patience must be a positive integer")
        delta = stop.get("min_delta")
        if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
            raise ManifestError("early_stopping min_delta must be finite and nonnegative")
    if set(stop) != fields:
        raise ManifestError("early_stopping has missing or unsupported fields")
    return {"max_epochs": maximum, "early_stopping": stop}


def validate_execution_budget(manifest, budget, *, repeat=False):
    runs = manifest.get('training_runs')
    if not isinstance(runs, list) or not runs:
        raise ManifestError('New training executions require explicit training_runs')
    if len(runs) > budget.max_conditions or repeat and len(runs) != 1:
        raise ManifestError('Training launch budget exceeded or repeat reran parameter search')
    for run in runs:
        if not isinstance(run, dict) or any(type(run.get(k)) is not int for k in ('max_epochs', 'epochs')):
            raise ManifestError('Training run epochs must be integers')
        if not 1 <= run['epochs'] <= run['max_epochs']:
            raise ManifestError('Training run completed epochs exceed configured limit')
    if sum(run['max_epochs'] for run in runs) > budget.max_total_epochs:
        raise ManifestError('Declared training execution exceeds total epoch budget')


def load_manifest(path: Path, *, require_training_policy: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ManifestError(f"Invalid experiment manifest: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ManifestError("Experiment manifest must be a schema-v1 JSON object")
    missing = sorted(REQUIRED_FIELDS - value.keys())
    if missing:
        raise ManifestError(f"Experiment manifest is missing required fields: {missing}")
    for key in REQUIRED_FIELDS:
        if value[key] in (None, "", [], {}):
            raise ManifestError(f"Experiment manifest field is empty: {key}")
    if training_policy(value) is None and require_training_policy:
        raise ManifestError("New executions require explicit max_epochs and early_stopping")
    return value


def require_manifest(task_root: Path) -> dict[str, Any]:
    return load_manifest(task_root / "candidate_frozen/experiment_manifest.json")
