# Contributing

Thank you for helping improve Path-AI Scientist. This project is a research workflow, not a
clinical system. Contributions must preserve its test-set isolation, evidence traceability,
restricted-use license, and human-review requirements.

## Before opening a pull request

1. Do not commit datasets, model weights, checkpoints, task state, API responses, logs, archives,
   credentials, or generated papers from real runs.
2. Add or update tests for workflow contracts, state transitions, security boundaries, and schema
   migrations affected by the change.
3. Run `python -m ruff check path_ai_scientist pathmnist gate_a app.py` and
   `python -m pytest -q --basetemp .test-tmp/pytest` in the pinned
   environment.
4. Run `python -m pathmnist.release_check --repo .` against the commit being proposed.
5. Document any paid-provider, GPU, dataset, or operating-system requirement. Paid and GPU tests
   must remain manual and must not run automatically on pull requests.

The upstream journal requires experiment paths below the current repository directory. Run tests
from the repository root and keep `--basetemp` inside `.test-tmp/`; pytest's system temporary
directory is not compatible with those integration fixtures. The selected temporary directory
is disposable and cleared by pytest: never point it at real task state or existing evidence.
Before the first test run, create the parent directory with
`python -c "from pathlib import Path; Path('.test-tmp').mkdir(exist_ok=True)"`.

## Repository scope

Keep runtime code, required upstream templates, configuration, launch/build helpers, and framework
regression tests. Keep documentation focused on installation, operation, architecture, provenance,
and maintenance. Personal experiment drivers, recovery attempts, development diaries, case-study
outputs, and generated manuscripts belong outside the tracked tree.

`tests/fixtures/semantic_recovery` contains inputs used by evidence, method-spec, and runtime
regression tests. Some inputs intentionally represent invalid generated code. They are not standalone
experiments and should not be deleted or "fixed" without reviewing the tests that consume them.

## Safety and scientific integrity

- Generated experiment code must run only in the configured sandbox.
- Candidate selection must use validation data only. The sealed test may be evaluated once after
  explicit approval and must never trigger reselection or tuning.
- Every numerical or literature claim in a formal paper must map to a frozen artifact.
- Machine-generated manuscripts must retain the prominent AI-generation disclosure.
- Do not present outputs as clinical advice, diagnosis, or evidence of statistical superiority
  beyond the recorded experiment design.

## Licensing

By contributing, you agree that your contribution may be distributed under the repository's
`LICENSE`, including its restricted-use and scientific-disclosure provisions. Retain third-party
notices and identify newly introduced third-party code or assets in `THIRD_PARTY_NOTICES.md`.
