# A Rigorous Single-Evaluation Test-Set Assessment of a SmallResNet Optimization Variant on PathMNIST

## Abstract

We report a frozen-candidate evaluation of a `SmallResNet` model (variant: `optimization`) on the PathMNIST dataset (SHA-256: `1e7fc200dd5aac79f39f0da26178be801ffac8e6d59c8533e51acebae953745a`). Training and tuning used only train/validation splits. After candidate freezing, the test split was evaluated exactly once; test results were not used for tuning, model selection, or retraining. The frozen candidate used learning rate `0.001`, weight decay `1e-05`, OneCycle scheduling, label smoothing `0.1`, and neither augmentation nor multiscale processing, across seeds `7`, `17`, and `27`. The best validation Macro-F1 during tuning was `0.997589`. On the single one-time test evaluation, the Macro-F1 mean was `0.834035` with standard deviation `0.075947`, yielding a validation-to-test Macro-F1 gap of `0.158425`. Seed `27` was unstable on test (Macro-F1 `0.746367`, best epoch `18`), whereas seeds `7` and `17` reached Macro-F1 values of `0.875924` and `0.879813`, respectively, both at best epoch `39`. The three lowest-recall classes in the aggregate confusion matrix were class `7` (`0.358`), class `0` (`0.724`), and class `6` (`0.858`). We make no clinical use or diagnostic claims.

## 1. Introduction

This document reports a strictly controlled evaluation of a frozen model candidate on the PathMNIST dataset. The experimental protocol was designed to prevent any information leakage from the test split into model development. Specifically, training and tuning used only train/validation splits, and the frozen candidate was evaluated on the test split exactly once. Test results were not used for tuning, model selection, or retraining.

The frozen candidate is a `SmallResNet` model with the `optimization` variant. The candidate was approved and frozen on `2026-08-18` by the project owner, with status `approved_frozen`. The dataset SHA-256 is `1e7fc200dd5aac79f39f0da26178be801ffac8e6d59c8533e51acebae953745a`. The primary metric is `macro_f1`. The test policy permits exactly one evaluation on the test split, with tuning feedback disallowed.

The contributions of this report are: (i) a transparent record of the tuning grid and train/validation results; (ii) a single, post-freeze test evaluation; and (iii) a discussion of the observed validation-to-test gap and per-seed instability. We do not propose clinical use or diagnostic claims.

## 2. Related Work

This report is constrained to the supplied frozen artifacts and does not cite external literature. Within the supplied artifacts, the relevant prior context is the set of pre-freeze experiments on PathMNIST under the `pathmnist-m4` run group. These include the `main` phase variants (`baseline`, `augmentation`, `optimization`, `multiscale`, `combined`) and the `ablations` phase variants (`combined_no_augmentation`, `combined_no_multiscale`). The `optimization` variant was selected as the frozen candidate based on validation Macro-F1.

## 3. Materials and Methods

### 3.1 Dataset

The dataset is PathMNIST, identified by SHA-256 `1e7fc200dd5aac79f39f0da26178be801ffac8e6d59c8533e51acebae953745a`. The dataset was partitioned into train, validation, and test splits. Training and tuning used only the train and validation splits. The test split was held out until after candidate freezing.

### 3.2 Model and Variant

The frozen candidate is a `SmallResNet` model with the `optimization` variant. The candidate status is `approved_frozen`, approved by the project owner on `2026-08-18`.

### 3.3 Hyperparameters

The frozen hyperparameters are:

- Learning rate: `0.001`
- Weight decay: `1e-05`
- OneCycle: `true`
- Label smoothing: `0.1`
- Augmentation: `false`
- Multiscale: `false`

### 3.4 Seeds

Three seeds were used: `7`, `17`, and `27`.

### 3.5 Primary Metric

The primary metric is `macro_f1`.

### 3.6 Test Policy

The test policy specifies:

- Split: `test`
- Evaluations allowed: `1`
- Tuning feedback allowed: `false`

The frozen candidate was evaluated on the test split exactly once after candidate freezing. Test results were not used for tuning, model selection, or retraining.

### 3.7 Runtime Environment

- PyTorch: `2.13.0+cu132`
- CUDA: `13.2`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`

## 4. Experiments

### 4.1 Tuning

Six pre-freeze grid settings were completed. The best learning rate identified was `0.001`, and the best weight decay was `1e-05`. The best validation Macro-F1 achieved during tuning was `0.997589`.

### 4.2 Train/Validation Results

The train/validation results across phases and variants are summarized below. Each variant was run with three runs.

| Phase | Variant | Runs | Macro-F1 mean | Macro-F1 std |
|---|---|---:|---:|---:|
| main | baseline | 3 | 0.956763 | 0.027502 |
| main | augmentation | 3 | 0.971311 | 0.011846 |
| main | optimization | 3 | 0.992459 | 0.008029 |
| main | multiscale | 3 | 0.978622 | 0.003913 |
| main | combined | 3 | 0.968218 | 0.024861 |
| ablations | combined_no_augmentation | 3 | 0.930088 | 0.036204 |
| ablations | combined_no_multiscale | 3 | 0.949365 | 0.020949 |

Among the `main` variants, `optimization` achieved the highest Macro-F1 mean (`0.992459`) with the lowest standard deviation (`0.008029`) among the non-multiscale variants. The `multiscale` variant had the lowest standard deviation (`0.003913`) but a lower mean (`0.978622`) than `optimization`.

### 4.3 Final One-Time Test Evaluation

After candidate freezing, the test split was evaluated exactly once. The evaluation count is `1`. The test Macro-F1 mean was `0.834035`, with a standard deviation of `0.075947`. The validation-to-test Macro-F1 gap was `0.158425`.

Per-seed test results are:

| Seed | Best epoch | Accuracy | Macro-F1 | Loss |
|---:|---:|---:|---:|---:|
| 7 | 39 | 0.906128 | 0.875924 | 0.397548 |
| 17 | 39 | 0.906267 | 0.879813 | 0.412215 |
| 27 | 18 | 0.779666 | 0.746367 | 0.722006 |

## 5. Results

### 5.1 Aggregate Test Performance

The aggregate test Macro-F1 mean was `0.834035` with standard deviation `0.075947`. The evaluation count was `1`. The validation-to-test Macro-F1 gap was `0.158425`.

### 5.2 Per-Seed Test Performance

Seeds `7` and `17` both achieved best epoch `39`, with accuracies of `0.906128` and `0.906267`, Macro-F1 values of `0.875924` and `0.879813`, and losses of `0.397548` and `0.412215`, respectively. Seed `27` achieved best epoch `18`, with accuracy `0.779666`, Macro-F1 `0.746367`, and loss `0.722006`.

### 5.3 Aggregate Confusion Matrix

The aggregate confusion matrix over the three seeds is:

| True \ Pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2905 | 335 | 1 | 34 | 18 | 683 | 29 | 0 | 9 |
| 1 | 12 | 2529 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 2 | 958 | 0 | 8 | 28 | 0 | 2 | 19 |
| 3 | 0 | 0 | 3 | 1898 | 0 | 0 | 0 | 0 | 1 |
| 4 | 50 | 2 | 3 | 0 | 2873 | 76 | 44 | 16 | 41 |
| 5 | 2 | 120 | 68 | 0 | 0 | 1567 | 0 | 18 | 1 |
| 6 | 0 | 0 | 23 | 104 | 6 | 52 | 1907 | 29 | 102 |
| 7 | 0 | 6 | 108 | 2 | 18 | 476 | 20 | 452 | 181 |
| 8 | 0 | 1 | 26 | 34 | 5 | 5 | 106 | 0 | 3522 |

### 5.4 Lowest-Recall Classes

The three lowest-recall classes in the aggregate confusion matrix are:

- Class `7`: recall `0.358`
- Class `0`: recall `0.724`
- Class `6`: recall `0.858`

## 6. Discussion

The frozen `optimization` candidate achieved a high validation Macro-F1 mean (`0.992459`) during the train/validation phase, and the best validation Macro-F1 during tuning was `0.997589`. However, the single one-time test evaluation produced a Macro-F1 mean of `0.834035`, yielding a material validation-to-test gap of `0.158425`.

Seed-level analysis indicates that the gap is not uniform across seeds. Seeds `7` and `17` produced comparable test Macro-F1 values (`0.875924` and `0.879813`) and accuracies (`0.906128` and `0.906267`), both at best epoch `39`. Seed `27`, however, was unstable on test, achieving a Macro-F1 of only `0.746367` at best epoch `18`, with a substantially higher loss (`0.722006`) compared to seeds `7` (`0.397548`) and `17` (`0.412215`). This instability reduces the mean and increases the variance of the aggregate test result.

The aggregate confusion matrix indicates that class `7` has particularly low recall (`0.358`), with notable confusion into classes `5` (`476` samples), `8` (`181` samples), and `2` (`108` samples). Class `0` recall is `0.724`, with notable confusion into class `5` (`683` samples) and class `1` (`335` samples). Class `6` recall is `0.858`, with notable confusion into class `3` (`104` samples) and class `8` (`102` samples).

These observations are final reporting only and did not trigger further tuning, in accordance with the test policy that disallows tuning feedback and permits exactly one test evaluation.

## 7. Limitations

Several limitations follow directly from the supplied artifacts:

1. **Single test evaluation.** The test split was evaluated exactly once after candidate freezing. No repeated test evaluations were performed, and test results were not used for tuning, model selection, or retraining.
2. **Material validation-to-test gap.** The validation-to-test Macro-F1 gap was `0.158425`, indicating a substantial transfer gap from validation to test.
3. **Seed instability.** Seed `27` was unstable on test, reducing the mean Macro-F1 and increasing the standard deviation. The small number of seeds (three) limits the robustness of the aggregate estimate.
4. **Class-level imbalance in errors.** Class `7` recall was `0.358`, indicating severe under-performance on that class. Classes `0` and `6` also had reduced recall (`0.724` and `0.858`).
5. **No external benchmarking.** This report is constrained to the supplied frozen artifacts and does not compare against external models or literature.
6. **No clinical or diagnostic claims.** This report makes no clinical use or diagnostic claims. The results are research reporting only.

## 8. Conclusion

We presented a frozen-candidate evaluation of a `SmallResNet` `optimization` variant on PathMNIST. The candidate was trained and tuned using only train/validation splits, and the test split was evaluated exactly once after candidate freezing. The best validation Macro-F1 during tuning was `0.997589`. The single one-time test evaluation produced a Macro-F1 mean of `0.834035` with standard deviation `0.075947`, and a validation-to-test gap of `0.158425`. Seed `27` was unstable on test, while seeds `7` and `17` produced comparable and higher Macro-F1 values. The three lowest-recall classes were `7` (`0.358`), `0` (`0.724`), and `6` (`0.858`). These observations are final reporting only and did not trigger further tuning. We do not propose clinical use or diagnostic claims.

## 9. Reproducibility Statement

### 9.1 Dataset

- Dataset: PathMNIST
- SHA-256: `1e7fc200dd5aac79f39f0da26178be801ffac8e6d59c8533e51acebae953745a`

### 9.2 Frozen Candidate

- Model: `SmallResNet`
- Variant: `optimization`
- Status: `approved_frozen`
- Approved by: `project_owner`
- Approved at: `2026-08-18`
- Seeds: `7, 17, 27`
- Learning rate: `0.001`
- Weight decay: `1e-05`
- OneCycle: `true`
- Label smoothing: `0.1`
- Augmentation: `false`
- Multiscale: `false`
- Primary metric: `macro_f1`

### 9.3 Test Policy

- Split: `test`
- Evaluations allowed: `1`
- Tuning feedback allowed: `false`
- Evaluation count: `1`

The test set was evaluated exactly once after candidate freezing. Test results were not used for tuning, model selection, or retraining.

### 9.4 Runtime Environment

- PyTorch: `2.13.0+cu132`
- CUDA: `13.2`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`

### 9.5 Artifacts

- `runs\pathmnist-m4\tuning\result.json`
- `runs\pathmnist-m4\final\optimization\seed_7\checkpoint.pt`
- `runs\pathmnist-m4\final\optimization\seed_17\checkpoint.pt`
- `runs\pathmnist-m4\final\optimization\seed_27\checkpoint.pt`
- `runs\pathmnist-m4\test_evaluation.json`
