# PathMNIST M4 Final Report

## Scope and discipline

- Training and tuning used train/validation splits only.
- The frozen candidate was evaluated on the test split exactly once.
- Test results were not used for tuning, model selection, or retraining.
- Dataset SHA-256: `1e7fc200dd5aac79f39f0da26178be801ffac8e6d59c8533e51acebae953745a`.

## Frozen candidate

- Model: `SmallResNet`
- Variant: `optimization`
- Seeds: `7, 17, 27`
- Learning rate: `0.001`
- Weight decay: `1e-05`
- OneCycle: `true`
- Label smoothing: `0.1`
- Augmentation: `false`
- Multiscale: `false`

## Tuning

- Six pre-freeze grid settings were completed.
- Best learning rate: `0.001`
- Best weight decay: `1e-05`
- Best validation Macro-F1: `0.997589`

## Train/validation results

| Phase | Variant | Runs | Macro-F1 mean | Macro-F1 std |
|---|---|---:|---:|---:|
| main | baseline | 3 | 0.956763 | 0.027502 |
| main | augmentation | 3 | 0.971311 | 0.011846 |
| main | **optimization** | 3 | 0.992459 | 0.008029 |
| main | multiscale | 3 | 0.978622 | 0.003913 |
| main | combined | 3 | 0.968218 | 0.024861 |
| ablations | combined_no_augmentation | 3 | 0.930088 | 0.036204 |
| ablations | combined_no_multiscale | 3 | 0.949365 | 0.020949 |

## Final one-time test result

- Test Macro-F1 mean: `0.834035`
- Test Macro-F1 std: `0.075947`
- Validation-to-test Macro-F1 gap: `0.158425`
- Evaluation count: `1`

| Seed | Best epoch | Accuracy | Macro-F1 | Loss |
|---:|---:|---:|---:|---:|
| 7 | 39 | 0.906128 | 0.875924 | 0.397548 |
| 17 | 39 | 0.906267 | 0.879813 | 0.412215 |
| 27 | 18 | 0.779666 | 0.746367 | 0.722006 |

## Interpretation

- The optimization-only candidate transfers from validation to test with a material gap.
- Seed 27 is unstable on test and reduces the mean while increasing variance.
- The three lowest-recall classes are `7` (0.358), `0` (0.724), `6` (0.858) in the aggregate confusion matrix.
- These observations are final reporting only and did not trigger further tuning.

## Runtime environment

- PyTorch: `2.13.0+cu132`
- CUDA: `13.2`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`

## Artifacts
- `runs\pathmnist-m4\tuning\result.json`
- `runs\pathmnist-m4\final\optimization\seed_7\checkpoint.pt`
- `runs\pathmnist-m4\final\optimization\seed_17\checkpoint.pt`
- `runs\pathmnist-m4\final\optimization\seed_27\checkpoint.pt`
- `runs\pathmnist-m4\test_evaluation.json`
