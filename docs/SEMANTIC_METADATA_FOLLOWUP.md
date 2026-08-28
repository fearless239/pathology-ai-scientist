# Semantic preflight follow-up (2026-08-28)

## Observed failure

The stage-3 node `8e8b78eaf50a40789ac58e746739cec3` was rejected before execution:
`SEMANTIC_REVIEW_REQUIRED`, unknown categories `cross_entropy` and
`label_smoothing_tuning`. A later resume encountered the same saved failure;
restarting alone does not clear the review stop. No successful stage-3 training
was lost. This was an incomplete metadata classification fix, not a GPU failure.

The exact failed program is retained in
`tests/fixtures/semantic_recovery/proposed_metadata.py`. Inspection also exposed
a second preflight rejection: the built-in loss receives `smoothing_factor`
as a function parameter, whereas the previous detector required a literal or
module-level constant. Merely accepting the two categories would not fix this.

## Project changes

- Share requirement ownership with metadata classification. Compose the tuning
  suffix only with a recognized method or supported ordinary hyperparameter;
  arbitrary unknown methods still require review. Metadata never proves that
  an intervention is implemented.
- Recognize standard cross-entropy calls using parameters of called functions
  as candidates requiring runtime verification, rather than proof of a positive
  smoothing coefficient. Undefined free variables remain rejected.
- The trusted stage interpreter requests the runtime requirement for an
  approved label-smoothing intervention. It is not imposed on ordinary baseline
  or ablation runs by guessing their role from generated metadata.
- Wrap the existing PyTorch functional cross-entropy entry used by the built-in
  module. Check active coefficients on differentiable calls and require a loss
  backward hook before accepting the run. Caught violations remain failures.
  Inference skips training-loss requirements. Existing custom-loss value and
  gradient checks remain in place.

## Verification and limits

The real failed program is tested through semantic classification and the stage
preflight/interpreter entry, with the external runner replaced for offline tests.
Runner tests verify that the trusted requirement reaches the launcher. Eight
tiny CPU tests in the existing PyTorch image, with network disabled and no GPU
or credentials, cover standard valid/zero/unused/caught cases and custom
valid/wrong-value/wrong-gradient/unused cases. All eight passed.

Use a fresh project-local `--basetemp` for the full suite. Using Windows' default
external temporary directory exposes an existing upstream `relative_to` path
assumption in 32 integration cases. The project-local run passes; that is not
evidence that external workspace roots are supported.

This is not a claim that all generated programs or the complete paid research
workflow now succeed. Parameter recognition is intentionally limited, and the
runtime checks do not establish every optimizer step's scientific correctness.
Recovery preparation uses the explicit stage-3 backup script; it does not start
paid execution, alter approved contracts, change metrics, or raise the budget.
