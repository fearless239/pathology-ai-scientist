# Dataset adapters

Pathology-AI-Scientist separates trusted dataset ingestion from LLM-generated experiment code. The
built-in `generic` adapter supports supervised image-classification data in three forms:

- an NPZ with `train_images/train_labels`, `val_images/val_labels`, and
  `test_images/test_labels` (the validation prefix may also be `valid` or `validation`);
- an image directory arranged as `split/class/image` or `class/image`;
- a CSV or JSON manifest with `path` and `label`, plus optional `split`, `id`,
  `patient_id`, or `group_id` fields.

The source dataset is never mounted directly into an experiment. Initialization creates a physical
research view containing only train and validation data. The sealed test view is materialized only by
the approved test-evaluation stage.

## Built-in adapter

```powershell
$datasetPath = Join-Path $env:USERPROFILE "Desktop\autoresearch\datasets\pneumoniamnist.npz"
path-ai-scientist init `
  --task-id pneumoniatest-001 `
  --dataset-adapter generic `
  --dataset-path $datasetPath `
  --budget-limit-usd 10 `
  --seed 7 `
  --direction "在 PneumoniaMNIST 官方划分上，从零训练相同的轻量级 CNN，比较普通交叉熵与仅加入训练集逆频率类别权重的加权交叉熵。以 macro-F1 为主指标，accuracy 和 weighted-F1 为次指标；固定 3 个随机种子进行配对训练，除 loss weighting 外保持模型、数据、优化器、学习率、batch size、epoch 上限和 early stopping 完全一致。不访问额外数据或预训练权重，在所有候选冻结后仅执行一次获批的密封测试，并如实报告正负结果与统计限制。"
```

PneumoniaMNIST 28x28 is distributed by MedMNIST as a 4.2 MB NPZ. Its official split contains
4,708 training, 524 validation, and 624 test images. Download it only from the official MedMNIST
Zenodo record: <https://zenodo.org/records/10519652>.

## Trusted custom adapter

Pass an import reference when a source cannot be represented by the generic formats:

```powershell
path-ai-scientist init `
  --task-id custom-001 `
  --dataset-adapter my_project.dataset_adapter:CustomDatasetAdapter `
  --dataset-path C:\datasets\custom `
  --direction "..."
```

The class constructor receives `seed=<int>` and `discover(source, profile_path)` must return a
`DatasetSpec`. Custom adapters execute in the host process and must therefore be reviewed and trusted;
never point this option at LLM-generated experiment code.

Copy `examples/custom_dataset_adapter` as a starting point. Before creating a paid task, run its
conformance test against disposable output:

```python
from pathlib import Path
from path_ai_scientist.adapters import load_dataset_adapter, validate_dataset_adapter

adapter = load_dataset_adapter("my_project.dataset_adapter:CustomDatasetAdapter", seed=7)
report = validate_dataset_adapter(adapter, Path("dataset"), Path(".adapter-check"))
assert report["passed"]
```

Conformance checks cover deterministic fingerprints and labels, required splits, duplicate IDs,
cross-split duplicate content, patient/group leakage, the train/validation research view, and physical
separation of the sealed test view. Passing conformance validates the data boundary, not the scientific
quality or clinical fitness of a dataset.
