# `pneumoniatest-001`: PneumoniaMNIST Adapter Run

> **Evidence boundary:** This is a completed end-to-end workflow artifact produced on 2026-09-10.
> It is published to demonstrate the generic dataset-adapter and controlled research workflow, not
> as peer-reviewed scientific or clinical evidence.

## Study

The task used the official PneumoniaMNIST splits to compare standard cross-entropy with
inverse-frequency class-weighted cross-entropy in the same lightweight CNN trained from scratch.
The comparison used paired seeds 0, 1, and 2, macro-F1 as the primary metric, and a single approved
sealed-test evaluation after candidate freezing.

The held-out mean macro-F1 difference was positive (`+0.0743`), but the three-seed comparison was
underpowered and did not establish a statistically reliable improvement. The result must therefore
be presented as a limited, non-confirmatory experiment rather than a successful performance claim.

## Paper

- [English generated paper](papers/final_paper_en.pdf)

The paper is AI-generated and has not been peer reviewed. The workflow's independent draft review
returned `Reject`; the artifact is retained transparently as evidence of system execution and still
requires human scientific and editorial revision.

## Deliberately excluded

The dataset, task state, checkpoints, model weights, raw predictions, LLM requests and responses,
logs, compilation intermediates, Chinese translation, and private local paths are not published.
