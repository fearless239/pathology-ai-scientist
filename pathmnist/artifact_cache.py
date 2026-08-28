"""Input-bound durable cache for generated publication artifacts."""
import hashlib
import json


def cached_artifact(path, inputs, generate):
    fingerprint = hashlib.sha256(inputs.encode('utf-8')).hexdigest()
    receipt = path.with_suffix(path.suffix + '.cache.json')
    value = None
    if receipt.exists():
        record = json.loads(receipt.read_text(encoding='utf-8'))
        if record.get('input_sha256') == fingerprint:
            value = record['value']
            if hashlib.sha256(value.encode('utf-8')).hexdigest() != record['value_sha256']:
                raise RuntimeError('Publication cache hash mismatch')
    elif path.exists():
        raise RuntimeError(f'Legacy publication artifact has no input binding: {path.name}; explicit migration required')
    if value is None:
        value = generate(fingerprint)
        record = {'schema_version': 1, 'input_sha256': fingerprint, 'value': value,
                  'value_sha256': hashlib.sha256(value.encode('utf-8')).hexdigest()}
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
        temporary.replace(receipt)
    # The receipt commits first; a crash before projection is harmless.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)
    return value
