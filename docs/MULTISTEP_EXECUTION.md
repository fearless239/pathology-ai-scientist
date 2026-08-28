# Multi-step execution migration status

## Implemented

`pathmnist.execution_plan` defines versioned train/select DAGs. Steps have safe,
unique IDs, topologically ordered dependencies, explicit seeds and parameters,
training fractions, epoch and timeout limits, and persistent retry budgets.
Worst-case epochs include retries. Host selection uses finite validation metrics,
keeps deterministic ties, and locks selected parameters for dependent training.
No test-set selection is accepted.

`StepExecutor` is a callback-based scheduling core. It stores attempts before
calling the executor, preserves failed attempt directories, verifies completed
steps on resume, binds checkpoints to plan/contract/dataset identities, and checks
dependency evidence before reuse. Callers must hold the task lock. The execution
callback must enforce the per-step timeout and physical dataset isolation; the
verification callback must recompute metrics and check artifact hashes, code,
seed, training controls, and parameter use. Scheduler unit tests use fake
callbacks and do not establish GPU or scientific correctness.

An optional `execution_plan` is covered by contract validation and the existing
contract hash/approval mechanism. Approved legacy contracts are never rewritten.
Both legacy preflight and paid execution reject explicitly multi-step research
before provider setup. Unsupported DAGs cannot silently fall back to the old
single-stage executor. Detection of legacy multi-step prose is conservative and
limited, not a general natural-language plan compiler.

## Not yet implemented — do not enable paid DAG execution

- Generation and user review of structured plans from research requirements.
- The production training adapter: per-step isolated datasets, generated programs,
  runtime budgets, verified results, and selected-parameter handoff.
- Compatibility of final artifacts with existing seed pairing, contract
  fulfillment, candidate freezing, and sealed-test evaluation.
- Frontend approval/progress/recovery controls for this execution backend.
- Audited migration of old journals, preserving baseline and historical evidence.
- Offline integration using the real sandbox followed by separately authorized
  live generation/training verification.

Current `pathtest-01` has no approved structured plan. It remains stopped. Do not
increase retry limits, remove the compatibility guard, or mark it completed to
work around the missing integration.
