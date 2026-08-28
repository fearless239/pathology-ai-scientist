"""Deterministic stage-2 acceptance using immutable validation artifacts."""
import ast
import hashlib
import json
import math
from pathlib import Path

from .autonomous_evidence import verified_metrics
from .experiment_manifest import ManifestError, training_policy, metric_policy
from .scientific_integrity import IntegrityError


def validate_tuning_record(record, manifest, metric_value, *, primary=None):
    try:
        policy = training_policy(manifest)
        primary = primary or manifest.get('primary_metric') or manifest['selection_metric']
        semantics = metric_policy(manifest, primary)
    except ManifestError as error:
        raise IntegrityError(str(error)) from error
    epoch_limit = policy['max_epochs'] if policy else manifest['epochs']
    if not isinstance(record, dict):
        raise IntegrityError('Tuning evidence must be a JSON object')
    if record.get('schema_version') != 1 or record.get('complete') is not True:
        raise IntegrityError('Tuning evidence is incomplete')
    if record.get('seed') != manifest['seed'] or record.get('selection_metric') != primary:
        raise IntegrityError(f'Tuning seed or selection metric mismatch: selection_metric must be approved primary {primary!r}; checkpoint loss belongs in checkpoint_selection')
    candidates = record.get('candidates', [])
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise IntegrityError('Tuning evidence requires exactly two completed candidates')
    rates = []
    values = []
    for row in candidates:
        if not isinstance(row, dict):
            raise IntegrityError('Invalid tuning candidate')
        lr, value = row.get('learning_rate'), row.get('validation_metric')
        if any(type(n) not in (int, float) or not math.isfinite(n) for n in (lr, value)) or lr <= 0:
            raise IntegrityError('Invalid tuning rate or validation metric')
        history = row.get('history', [])
        if not isinstance(history, list) or not history or len(history) > epoch_limit:
            raise IntegrityError('Missing or excessive tuning training history')
        if policy and not policy['early_stopping']['enabled'] and len(history) != epoch_limit:
            raise IntegrityError('Incomplete training with early_stopping disabled')
        for index, epoch in enumerate(history, 1):
            if not isinstance(epoch, dict) or epoch.get('epoch') != index or any(
                type(epoch.get(key)) not in (int, float) or not math.isfinite(epoch[key])
                for key in ('train_loss', 'validation_loss', 'validation_metric')
            ):
                raise IntegrityError('Invalid tuning training history')
        checkpoint = semantics['checkpoint_selection']
        if 'checkpoint_selection' in manifest or checkpoint['metric'] == 'validation_loss':
            selected_epoch = row.get('selected_epoch')
            if type(selected_epoch) is not int or not 1 <= selected_epoch <= len(history):
                raise IntegrityError('Checkpoint selection requires explicit selected_epoch; legacy evidence needs migration')
            key = 'validation_loss' if checkpoint['metric'] == 'validation_loss' else 'validation_metric'
            best = (min if checkpoint['mode'] == 'min' else max)(e[key] for e in history)
            selected_history = history[selected_epoch - 1]
            if not math.isclose(selected_history[key], best, abs_tol=1e-6):
                raise IntegrityError('Selected epoch does not follow checkpoint_selection')
            expected_score = selected_history['validation_metric']
        else:
            expected_score = max(e['validation_metric'] for e in history)
        if not math.isclose(expected_score, value, abs_tol=1e-6):
            raise IntegrityError('Candidate score must match best validation epoch')
        rates.append(lr)
        values.append(value)
    if len(set(rates)) != 2 or record.get('selected_learning_rate') != manifest['learning_rate']:
        raise IntegrityError('Selected learning rate mismatch')
    if manifest['learning_rate'] not in rates:
        raise IntegrityError('Selected learning rate was not evaluated')
    selected = values[rates.index(manifest['learning_rate'])]
    if policy and manifest['epochs'] != len(candidates[rates.index(manifest['learning_rate'])]['history']):
        raise IntegrityError('Selected completed epochs do not match tuning history')
    if not math.isclose(selected, max(values), abs_tol=1e-6) or not math.isclose(selected, metric_value, abs_tol=1e-6):
        raise IntegrityError('Selected score does not match trusted validation predictions')
    return rates


def model_signature(code):
    tree = ast.parse(code)
    # Documentation does not affect the executed architecture.
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
               and any(ast.unparse(base).endswith('.Module') for base in node.bases)]
    if not classes:
        raise IntegrityError('Model architecture cannot be verified from Module definitions')
    names = {node.name for node in classes}
    constructors = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id in names]
    # Tuning may instantiate the same architecture again to restore the selected
    # checkpoint. Call-site count/order is not an architectural property. Retain
    # every distinct constructor expression so changed arguments still fail.
    return (
        tuple(ast.dump(node, include_attributes=False) for node in classes),
        tuple(sorted({ast.dump(node, include_attributes=False) for node in constructors})),
    )


def select_verified_tuning(journal, root: Path, metric: str):
    """Never consult live workspaces or reconstruct absent historical evidence."""
    if not journal.nodes:
        return None, ['No baseline in tuning journal']
    profile = root / 'dataset/research_view/dataset_profile.json'
    def evidence(node):
        digest = hashlib.sha256(node.code.encode('utf-8')).hexdigest()
        directory = root / 'experiment_logs/evidence' / digest
        metrics = verified_metrics(directory, profile, digest)
        manifest = json.loads((directory / 'experiment_manifest.json').read_text(encoding='utf-8'))
        return directory, manifest, metrics
    try:
        _, baseline, _ = evidence(journal.nodes[0])
        architecture = model_signature(journal.nodes[0].code)
    except (OSError, ValueError, KeyError, TypeError, IntegrityError, ManifestError) as error:
        return None, [f'Baseline evidence: {error}']
    accepted, errors = [], []
    for node in journal.nodes[1:]:
        if node.is_buggy is not False:
            errors.append(f'{node.id}: {getattr(node, "analysis", "")}; {str(getattr(node, "term_out", ""))[-600:]}')
            continue
        if node.is_seed_node or node.is_seed_agg_node:
            continue
        try:
            directory, manifest, metrics = evidence(node)
            hashes = json.loads((directory / 'artifact_hashes.json').read_text(encoding='utf-8'))
            if 'tuning_evidence.json' not in hashes:
                raise IntegrityError('Missing immutable tuning_evidence.json; legacy metric alone is insufficient')
            record = json.loads((directory / 'tuning_evidence.json').read_text(encoding='utf-8'))
            baseline_policy, candidate_policy = training_policy(baseline), training_policy(manifest)
            if baseline_policy != candidate_policy:
                raise IntegrityError('Baseline training policy changed or missing: max_epochs/early_stopping')
            # Legacy evidence has no unambiguous configured limit: retain strict comparison.
            if baseline_policy is None and manifest.get('epochs') != baseline.get('epochs'):
                raise IntegrityError('Baseline control changed: epochs (legacy evidence lacks training policy)')
            for field in ('model', 'dataset', 'optimizer', 'batch_size', 'input_resolutions', 'seed'):
                if manifest.get(field) != baseline.get(field):
                    raise IntegrityError(f'Baseline control changed: {field}')
            if metric_policy(baseline, metric) != metric_policy(manifest, metric):
                raise IntegrityError('Baseline checkpoint selection policy changed')
            if model_signature(node.code) != architecture:
                raise IntegrityError('Model architecture changed during tuning')
            rates = validate_tuning_record(record, manifest, metrics[metric], primary=metric)
            if baseline['learning_rate'] not in rates:
                raise IntegrityError('Tuning did not include baseline learning rate')
            accepted.append((metrics[metric], node))
        except (OSError, ValueError, KeyError, TypeError, IntegrityError, ManifestError) as error:
            errors.append(f'{node.id}: {error}')
    return (max(accepted, key=lambda row: row[0])[1] if accepted else None), errors
