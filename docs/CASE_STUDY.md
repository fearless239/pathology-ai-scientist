# Case Study: Reproducible PathMNIST Evaluation

## Question

Can a research agent preserve split discipline, freeze a candidate, consume a sealed test once, and
produce a traceable evaluation report without turning generated prose into unsupported claims?

## Evidence

The PathMNIST reference run selected an optimization variant on validation evidence. The frozen
candidate used three seeds and was evaluated on the test set once. Test Macro-F1 was
`0.834035 ± 0.075947`, versus validation Macro-F1 `0.992459 ± 0.008029`. Seed 27 fell to `0.746367`.
The comparison is retained so the evaluation can be interpreted and reproduced accurately.

The public, de-identified bundle is under `examples/pathmnist-case-study`. It contains only curated
JSON and prose. Dataset arrays, weights, provider replies, checkpoints, machine paths, and task state
remain excluded.

## What this demonstrates

- Transactional stage acceptance is separate from “a tool returned successfully.”
- Human approval sits before research-contract execution and sealed-test consumption.
- Metrics are trusted only when their execution receipt and provenance validate.
- Validation/test comparisons and per-seed variation survive into the report.
- The manuscript is a downstream artifact, not the source of truth.

## What this does not demonstrate

This is a small patch-classification benchmark with three seeds. It is not a whole-slide workflow,
external replication, clinical validation, a state-of-the-art result, or evidence that generated papers
are publication ready.
