"""One immutable baseline policy shared by generation and experiment acceptance."""
import ast
import hashlib
import json

from .autonomous_evidence import verified_metrics
from .experiment_manifest import training_policy
from .scientific_integrity import IntegrityError


def bind_policy(root, baseline_code):
    digest = hashlib.sha256(baseline_code.encode('utf-8')).hexdigest()
    evidence = root / 'experiment_logs/evidence' / digest
    verified_metrics(evidence, root / 'dataset/research_view/dataset_profile.json', digest)
    manifest = json.loads((evidence / 'experiment_manifest.json').read_text(encoding='utf-8'))
    policy = training_policy(manifest)
    if policy is None:
        raise IntegrityError('Comparison baseline lacks an explicit training policy')
    record = {'baseline_code_sha256': digest, 'policy': policy}
    path = root / 'research/comparison_training_policy.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != record:
        raise IntegrityError('Comparison baseline policy changed; explicit review required')
    if not path.exists():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        temporary.replace(path)
    return policy


def read_policy(root):
    path = root / 'research/comparison_training_policy.json'
    return json.loads(path.read_text(encoding='utf-8'))['policy'] if path.exists() else None


def validate_final_plan(code, policy, budget):
    """Require an explicit pre-execution plan; output evidence is checked again."""
    tree = ast.parse(code)
    plans = [node.value for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'FINAL_TRAINING_PLAN' for t in node.targets)]
    try:
        plan = ast.literal_eval(plans[0]) if len(plans) == 1 else None
    except (ValueError, TypeError):
        plan = None
    if not isinstance(plan, dict):
        raise IntegrityError('Declare one literal FINAL_TRAINING_PLAN with max_epochs, early_stopping, search_epochs (list of per-candidate caps)')
    if {k: plan.get(k) for k in policy} != policy:
        raise IntegrityError(f'Final training policy must equal trusted baseline: {policy}; reduce search, never final training')
    search = plan.get('search_epochs')
    if (not isinstance(search, list) or any(type(n) is not int or n < 1 for n in search)
            or len(search) + 1 > budget.max_conditions
            or sum(search) + policy['max_epochs'] > budget.max_total_epochs):
        raise IntegrityError('Reserve baseline final max_epochs first; search_epochs exceed remaining execution budget')
    # Catch common stale assignments even when the declaration is correct.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(isinstance(t, ast.Name) and t.id in {'final_train_epochs', 'final_max_epochs'} for t in node.targets):
                if node.value.value != policy['max_epochs']:
                    raise IntegrityError('Final training epoch assignment contradicts trusted baseline policy')


def validate_final_manifest(manifest, policy):
    if training_policy(manifest) != policy:
        raise IntegrityError('Final training max_epochs/early_stopping differs from trusted baseline; preserve outputs for review')
