import pytest
import json
from pathlib import Path

from pathmnist.method_spec import (
    attach_method_spec,
    extract_method_spec,
    normalize_symbol,
    parse_method_spec,
    semantic_report,
)


TRAINING = """
import torchvision.transforms as transforms
train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
])
train_dataset = PathologyDataset(train_images, train_labels, transform=train_transform)
loss.backward()
optimizer.step()
"""


def test_latest_rejected_program_passes_real_stage_preflight_with_runtime_requirement():
    fixture = Path(__file__).parent / 'fixtures/semantic_recovery'
    code = (fixture / 'proposed_metadata.py').read_text(encoding='utf-8')
    signals = json.loads((fixture / 'requirements.json').read_text(encoding='utf-8'))['implementation_signals']
    report = semantic_report(code, signals, extract_method_spec(code))
    assert report['passed'], report
    assert report['runtime_checks']['standard_smoothing_required']
    from pathmnist.autonomous import _validate_stage_semantics
    _validate_stage_semantics(code, '3_creative_research_1_first_attempt', signals)


@pytest.mark.parametrize('category', ['cross_entropy', 'CrossEntropy', 'label_smoothing_tuning',
                                    'LabelSmoothingTuning', 'learning_rate_tuning', 'rotation_tuning'])
def test_recognized_metadata_composition_still_requires_actual_intervention(category):
    spec = {'components': [{'category': category}]}
    report = semantic_report('loss.backward()', ['label_smoothing'], spec)
    assert not report['unknown']
    assert report['missing'] == ['label_smoothing']
    assert not report['passed']


@pytest.mark.parametrize('category', ['novel_stain_tuning', 'novel_label_smoothing_tuning'])
def test_unknown_tuning_is_not_approved_by_suffix(category):
    report = semantic_report(TRAINING, ['rotation'], {'components': [{'category': category}]})
    assert report['status'] == 'needs_review'



def test_actual_failed_contract_and_program_require_runtime_loss_verification():
    fixture = Path(__file__).parent / 'fixtures/semantic_recovery'
    code = (fixture / 'proposed.py').read_text(encoding='utf-8')
    signals = json.loads((fixture / 'requirements.json').read_text(encoding='utf-8'))['implementation_signals']
    report = semantic_report(code, signals, extract_method_spec(code))
    assert report['passed']
    assert report['required'] == ['label_smoothing']
    assert report['requirement_groups']['evaluation'] == ['test_accuracy']
    assert report['requirement_groups']['data'] == ['train_subset_fraction']
    assert report['runtime_checks']['custom_smoothing_classes'] == ['LabelSmoothingCrossEntropy']
    from pathmnist.research_contract import code_semantic_evidence
    assert code_semantic_evidence(code, signals)['passed']
    from pathmnist.autonomous import _validate_stage_semantics
    _validate_stage_semantics(code, '3_creative_research_1_first_attempt', signals)


@pytest.mark.parametrize('category', ['cnn_architecture', 'data_loading', 'classification_metrics', 'sgd', 'supervised_training', 'smoothing_factor_tuning'])
def test_normal_research_metadata_does_not_need_false_augmentation_labels(category):
    report = semantic_report('loss=nn.CrossEntropyLoss(label_smoothing=0.1)\nloss.backward()',
                             ['label_smoothing'], {'components': [{'category': category}]})
    assert report['passed']


def test_camel_case_torchvision_calls_match_contract_concepts():
    report = semantic_report(
        TRAINING, ["transform", "color_jitter", "rotation", "flip"]
    )
    assert report["passed"] is True
    assert report["missing"] == []
    assert set(report["detected"]) == {
        "color_perturbation",
        "rotation",
        "flip",
    }


def test_import_without_call_is_not_semantic_evidence():
    code = "from torchvision.transforms import ColorJitter\nloss.backward()\n"
    report = semantic_report(code, ["color_jitter"])
    assert report["passed"] is False
    assert report["missing"] == ["color_perturbation"]


def test_augmentation_only_in_validation_pipeline_is_rejected():
    code = """
import torchvision.transforms as transforms
train_transform = transforms.Compose([transforms.Resize((224, 224))])
validation_transform = transforms.Compose([transforms.ColorJitter(0.2)])
train_dataset = PathologyDataset(train_images, train_labels, transform=train_transform)
validation_dataset = PathologyDataset(val_images, val_labels, transform=validation_transform)
loss.backward()
optimizer.step()
"""
    report = semantic_report(code, ["color_jitter"])
    assert report["passed"] is False
    assert report["missing"] == ["color_perturbation"]


def test_transform_forwarded_through_training_helper_is_traced():
    code = """
import torchvision.transforms as transforms
proposed_transform = transforms.Compose([
    transforms.ColorJitter(0.2),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
])
def run_experiment(train_images, train_labels, train_transform):
    train_dataset = PathologyDataset(train_images, train_labels, transform=train_transform)
run_experiment(train_images, train_labels, train_transform=proposed_transform)
loss.backward()
optimizer.step()
"""
    report = semantic_report(code, ["color_jitter", "rotation", "flip"])
    assert report["passed"] is True


def test_unknown_declared_component_needs_review():
    spec = {
        "components": [
            {
                "id": "novel",
                "category": "novel_stain_policy",
                "implementation_symbols": ["NovelStainPolicy"],
            }
        ]
    }
    report = semantic_report(TRAINING, ["color_jitter"], spec)
    assert report["status"] == "needs_review"
    assert report["unknown"] == ["novel_stain_policy"]


def test_generic_augmentation_group_does_not_force_manual_review():
    spec = {
        "components": [
            {
                "id": "augmentation",
                "category": "image_augmentation",
                "implementation_symbols": ["ColorJitter"],
            }
        ]
    }
    report = semantic_report(TRAINING, ["color_jitter", "rotation", "flip"], spec)
    assert report["status"] == "passed"


def test_method_spec_round_trip_in_free_form_code():
    spec = {
        "hypothesis": "augmentation helps",
        "components": [
            {
                "id": "color",
                "category": "color_perturbation",
                "implementation_symbols": ["ColorJitter"],
            }
        ],
        "changes": ["training_transform"],
        "preserved": ["validation_transform"],
    }
    code = attach_method_spec("print('free form')\n", spec)
    assert extract_method_spec(code) == {"schema_version": 1, **spec}


def test_method_spec_can_be_parsed_from_agent_plan():
    plan = (
        'METHOD_SPEC: {"hypothesis":"x","components":[],"changes":[],"preserved":[]}\n'
        "I will freely implement the method."
    )
    assert parse_method_spec(plan)["hypothesis"] == "x"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("ColorJitter", "colorjitter"), ("color_jitter", "colorjitter"), ("color-jitter", "colorjitter")],
)
def test_symbol_normalization(raw, expected):
    assert normalize_symbol(raw) == expected

@pytest.mark.parametrize('category', ['architecture','model_architecture','loss','loss_function','optimizer','optimization','regularization'])
def test_standard_component_categories_are_metadata(category):
    code = 'criterion = nn.CrossEntropyLoss(label_smoothing=0.1)\nloss.backward()\n'
    report = semantic_report(code, ['label_smoothing'], {'components':[{'category':category}]})
    assert report['passed']

@pytest.mark.parametrize('amount', ['0', '-0.1', '1.0', 'unknown'])
def test_label_smoothing_requires_positive_known_parameter(amount):
    report = semantic_report(f'criterion = nn.CrossEntropyLoss(label_smoothing={amount})\nloss.backward()', ['label_smoothing'])
    assert not report['passed']


def test_loss_and_structure_are_not_filtered_as_dataset_transforms():
    code = '''
train_dataset = TensorDataset(train_images, train_labels)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
model = Net(num_classes=9)
x = self.conv1(x)
loss.backward()
'''
    report = semantic_report(code, ['label_smoothing','smoothing_alpha','smoothed_targets','cross_entropy','num_classes','conv1'])
    assert report['passed']
    assert report['required'].count('label_smoothing') == 1


def test_generic_category_does_not_satisfy_missing_intervention():
    assert not semantic_report('loss.backward()', ['label_smoothing'], {'components':[{'category':'loss'}]})['passed']
