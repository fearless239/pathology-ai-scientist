"""Host-derived requirements for generated image-classification programs."""
import json
import re
from pathlib import Path


def input_sizes(contract):
    explicit = contract.get('execution_requirements', {}).get('input_sizes')
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit or any(
            not isinstance(size, list) or len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size) for size in explicit
        ):
            raise ValueError('Invalid contract execution_requirements.input_sizes')
        return explicit
    # Compatibility for previously approved contracts; never rewrite their hashes.
    sizes = sorted({(int(a), int(b)) for a, b in re.findall(
        r'(?<!\d)(\d{1,4})\s*[×xX]\s*(\d{1,4})(?!\d)',
        contract.get('research_question', ''),
    )})
    if len(sizes) > 1:
        raise ValueError('Ambiguous input sizes: explicitly approve execution_requirements.input_sizes')
    return [list(size) for size in sizes]


def requirements_for_dataset(dataset: Path | None):
    if dataset is None:
        return {}
    import numpy as np
    dataset = dataset.resolve()
    inference = False
    if (dataset / 'dataset.npz').is_file():
        with np.load(dataset / 'dataset.npz', allow_pickle=False) as arrays:
            inference = 'validation_images' in arrays.files and 'train_images' not in arrays.files
    elif (dataset / 'manifest.json').is_file():
        rows = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
        inference = bool(rows) and not any(row.get('split') == 'train' for row in rows)
    contract = None
    for parent in (dataset, *dataset.parents):
        if (parent / 'task.json').is_file():
            if (parent / 'research/research_contract.json').is_file():
                from pathmnist.research_contract import load_contract
                contract = load_contract(parent, require_approved=True)
            break
    sizes = input_sizes(contract) if contract else []
    return {'input_sizes': sizes, 'inference_only': inference} if sizes or inference else {}
