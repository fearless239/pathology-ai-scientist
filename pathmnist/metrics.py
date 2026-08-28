"""One resolver for scalar and preselected subgroup metrics."""
import math
import json
import os
import uuid


def metric_value(metrics, name, pair=None, class_id=None):
    value = metrics.get(name)
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    indices = pair if name == 'confusion_pair_mean_f1' else [class_id] if name == 'class_f1' else None
    rows = metrics.get('per_class', [])
    if indices and all(type(i) is int and 0 <= i < len(rows) for i in indices):
        values = [rows[i].get('f1') for i in indices]
        if all(type(v) in (int, float) and math.isfinite(v) for v in values):
            return sum(values) / len(values)
    return None


def most_confused_pair(metrics):
    matrix = metrics.get('confusion_matrix', [])
    pairs = [(matrix[a][b] + matrix[b][a], -a, -b)
             for a in range(len(matrix)) for b in range(a + 1, len(matrix))]
    if not pairs:
        return None
    _, a, b = max(pairs)
    return [-a, -b]


def add_contract_metric(root, metrics, *, baseline_hash=None):
    """Lock a validation subgroup once; never derive it from the intervention/test."""
    path = root / 'research/research_contract.json'
    if not path.is_file():
        return metrics
    from .research_contract import load_contract
    from .scientific_integrity import IntegrityError
    contract = load_contract(root, require_approved=True)
    primary = contract['metrics']['primary']
    name = primary['name']
    pair = None
    if name == 'confusion_pair_mean_f1':
        lock = root / 'research/locked_metric.json'
        if not lock.exists() and baseline_hash:
            record = {'contract_sha256': contract['contract_sha256'],
                      'baseline_code_sha256': baseline_hash, 'pair': most_confused_pair(metrics)}
            temporary = lock.with_name(lock.name + '.' + uuid.uuid4().hex + '.tmp')
            temporary.write_text(json.dumps(record), encoding='utf-8')
            try:
                os.link(temporary, lock)  # Publish complete bytes exclusively, never a partial JSON lock.
            except FileExistsError:
                pass
            finally:
                temporary.unlink()
        if not lock.is_file():
            raise IntegrityError('Baseline subgroup is not locked; explicit baseline recovery required')
        record = json.loads(lock.read_text(encoding='utf-8'))
        if record['contract_sha256'] != contract['contract_sha256']:
            raise IntegrityError('Metric subgroup belongs to another contract')
        pair = record['pair']
    value = metric_value(metrics, name, pair, primary.get('class_id'))
    if value is None:
        raise IntegrityError(f'Trusted metric cannot be resolved: {name}')
    return {**metrics, name: value}
