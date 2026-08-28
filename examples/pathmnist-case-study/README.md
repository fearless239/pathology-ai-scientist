# PathMNIST evidence case study

This is a curated, de-identified evidence summary—not a raw run directory and not a clinical result.
No dataset, model weight, API response, checkpoint, secret, or absolute local path is included.

The frozen SmallResNet candidate was selected using train/validation evidence and evaluated on the
test split once. Across seeds 7, 17, and 27, test Macro-F1 was `0.834035 ± 0.075947`. The bundle records
the validation/test comparison and per-seed variation for reproducible interpretation. See
[the full technical case study](../../docs/CASE_STUDY.md).

Artifacts:

- `task_config.json`: de-identified intent and control boundaries.
- `timeline.json`: simplified evaluation transaction history.
- `experiment_manifest.json`: curated aggregate results and provenance pointers.
- `acceptance_report.json`: what is and is not accepted for public claims.
- `generated_manuscript_sample.md`: deliberately short, visibly non-peer-reviewed output sample.
