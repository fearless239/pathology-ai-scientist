"""Run in the pinned torch image as well as normal CI (torch optional locally)."""
import subprocess
import sys
import shutil

import pytest

from gate_a.import_preflight import sandbox_launcher


torch = pytest.importorskip('torch')


def run(tmp_path, source, inference=False, custom_losses=None, standard_required=False):
    script = tmp_path / 'experiment.py'
    script.write_text(source, encoding='utf-8')
    launcher = sandbox_launcher(source, str(script), {'input_sizes': [[28, 28]], 'inference_only': inference,
                                                     'custom_smoothing_classes': custom_losses or [],
                                                     'standard_smoothing_required': standard_required})
    # Isolated test filesystem substitute for the fixed production mount.
    launcher = launcher.replace("Path('/workspace/model_checkpoint.pt')", f'Path({str(tmp_path / "model_checkpoint.pt")!r})')
    return subprocess.run([sys.executable, '-c', launcher], cwd=tmp_path,
                          capture_output=True, text=True, timeout=40)


MODEL = '''import torch
from pathlib import Path
model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2))
'''


@pytest.mark.parametrize('variant', ['valid', 'zero', 'unused', 'caught'])
def test_dynamic_standard_smoothing_is_checked_on_backward(tmp_path, variant):
    source = MODEL + '''
def train_candidate(smoothing_factor):
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing_factor)
    return criterion(model(torch.randn(4,3,28,28)), torch.tensor([0,1,0,1]))
'''
    if variant == 'caught':
        source += 'try:\n    train_candidate(0.0)\nexcept RuntimeError:\n    pass\ntrain_candidate(0.1).backward()\n'
    else:
        source += f'loss = train_candidate({0.0 if variant == "zero" else 0.1})\n'
        if variant != 'unused':
            source += 'loss.backward()\n'
    result = run(tmp_path, source, standard_required=True)
    assert (result.returncode == 0) == (variant == 'valid'), result.stderr
    if variant != 'valid':
        assert 'MODEL_CONTRACT_FAILED' in result.stderr


@pytest.mark.parametrize('variant', ['valid', 'wrong_value', 'wrong_gradient', 'unused'])
def test_real_generated_custom_loss_value_gradient_and_backward(tmp_path, variant):
    import ast
    from pathlib import Path
    fixture = Path(__file__).parent / 'fixtures/semantic_recovery/proposed.py'
    tree = ast.parse(fixture.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LabelSmoothingCrossEntropy')
    source = ast.unparse(node)
    if variant == 'wrong_value':
        source = source.replace('return loss.mean()', 'return loss.mean() + 1')
    if variant == 'wrong_gradient':
        source = source.replace('return loss.mean()', 'return loss.mean().detach() + 0 * pred.sum()')
    source = MODEL + '\nimport torch.nn as nn\nimport torch.nn.functional as F\n' + source
    source += '\nlogits=model(torch.randn(4,3,28,28))\ncriterion=LabelSmoothingCrossEntropy(0.1)\nloss=criterion(logits, torch.tensor([0,1,0,1]))\n'
    if variant != 'unused':
        source += 'loss.backward()\n'
    result = run(tmp_path, source, custom_losses=['LabelSmoothingCrossEntropy'])
    assert (result.returncode == 0) == (variant == 'valid'), result.stderr
    if variant != 'valid':
        assert 'MODEL_CONTRACT_FAILED' in result.stderr


def test_short_train_save_and_independent_reload(tmp_path):
    trained = run(tmp_path, MODEL + '''
torch.manual_seed(0)
x = torch.randn(4, 3, 28, 28)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0,1,0,1]))
loss.backward()
optimizer.step()
torch.save(model.state_dict(), 'trained.pt')
model.eval()
torch.save({'x': x, 'expected': model(x).detach()}, 'reference.pt')
''')
    assert trained.returncode == 0, trained.stderr
    shutil.copy2(tmp_path / 'trained.pt', tmp_path / 'model_checkpoint.pt')
    evaluated = run(tmp_path, MODEL + '''
model.load_state_dict(torch.load(Path('model_checkpoint.pt').resolve(), weights_only=True))
model.eval()
reference = torch.load('reference.pt', weights_only=True)
assert torch.allclose(model(reference['x']), reference['expected'])
''', inference=True)
    assert evaluated.returncode == 0, evaluated.stderr
    assert '"checkpoint_restored": true' in evaluated.stdout


def test_wrong_actual_size_fails_even_if_manifest_claims_28(tmp_path):
    result = run(tmp_path, MODEL + "input_resolutions = [28]\nmodel(torch.randn(2,3,64,64))")
    assert result.returncode != 0
    assert 'actual model input (64, 64)' in result.stderr


def test_missing_checkpoint_fails_before_random_fallback(tmp_path):
    result = run(tmp_path, MODEL + "model(torch.randn(2,3,28,28))", inference=True)
    assert result.returncode != 0
    assert 'inference requires' in result.stderr


@pytest.mark.parametrize('body', [
    "model(torch.randn(2,3,28,28))",
    "state=torch.load('model_checkpoint.pt', weights_only=True)\nmodel(torch.randn(2,3,28,28))",
    "model.load_state_dict(model.state_dict())\nmodel(torch.randn(2,3,28,28))",
])
def test_checkpoint_file_or_torch_load_alone_is_not_restoration(tmp_path, body):
    torch.save({'other': torch.ones(1)}, tmp_path / 'model_checkpoint.pt')
    result = run(tmp_path, MODEL + body, inference=True)
    assert result.returncode != 0
    assert 'MODEL_CONTRACT_FAILED' in result.stderr


def test_partial_checkpoint_cannot_be_silently_accepted(tmp_path):
    torch.save({'0.weight': torch.ones(4, 3, 3, 3)}, tmp_path / 'model_checkpoint.pt')
    result = run(tmp_path, MODEL + "model.load_state_dict(torch.load('model_checkpoint.pt', weights_only=True), strict=False)", inference=True)
    assert result.returncode != 0
    assert 'partial loading is forbidden' in result.stderr


def test_catching_shape_error_cannot_turn_run_into_success(tmp_path):
    result = run(tmp_path, MODEL + '''
try:
    model(torch.randn(1,3,64,64))
except RuntimeError:
    pass
model(torch.randn(1,3,28,28))
''')
    assert result.returncode != 0
    assert 'MODEL_CONTRACT_FAILED' in result.stderr
