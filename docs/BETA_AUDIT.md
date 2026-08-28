# Public Beta Audit

## Decision

The repository is an advanced research prototype with one passing real P0 end-to-end acceptance
task (`p0-beta-e2e-20260820T215055Z`). That run completed verified research, generated experiments,
candidate freeze, one-time sealed test, evidence-bound writing, independent review, revision,
translation, PDF QA, and archival. Public release still requires committing and reviewing the
release boundary; a clean-environment reproduction was deliberately deferred by the project owner.

The historical `pathmnist-dynamic-resolution-v2` task is retained as regression evidence but is
not an accepted formal-paper run: its research, literature, idea, and experiment-spec stages are
still `waiting`, its test metrics use legacy validation-prefixed names, and its paper lacks verified
references and a complete experiment manifest. The new acceptance validator rejects this state.

The repository now preserves the AI Scientist Source Code License at its root and documents its
source-available (not OSI open-source) status. `THIRD_PARTY_NOTICES.md` records the upstream,
MedMNIST, AIDE, and document-asset boundaries. Formal-paper output now requires a prominent,
deterministically inserted AI-generation disclosure.

## Updated assessment

| Area | Score | Evidence-based assessment |
|---|---:|---|
| Core engineering loop | 75% | Experiment-to-paper path exists; formal path is now gated |
| Pathology specialization | 60% | Split/claim controls exist; validation remains PathMNIST-heavy |
| Scientific reliability | 60% | Freeze/test discipline is strong; statistical protocol remains P1 |
| Paper generation | 55% | Evidence, literature, manifest, references and PDF gates are now required |
| Maintainability | 55% | Contracts added; legacy and v2 orchestrators still coexist |
| GitHub readiness | 80% | License, notices, CI and release scan exist; clean-clone validation is deferred |

## Implemented P0 controls

- A single autonomous acceptance validator enforces stage order and durable artifacts.
- Formal papers require verified literature with a stable identifier or source URL.
- Schema-v2 test output forbids validation-prefixed test metrics; schema-v1 evidence migrates in memory.
- Generated experiments must emit a reproducibility manifest before they can become autonomous candidates.
- Formal postprocessing stops at translation; PDF QA is the only component allowed to archive.
- PDF QA rejects missing references, unresolved placeholder language, undefined citations/references,
  material overfull boxes, failed compilation, or implausibly small output.
- NPY header inspection uses `ast.literal_eval`; object-pickle handling is explicitly confined to
  legacy task-owned experiment output.
- A release checker rejects tracked runtime state, datasets, secrets, and oversized files.
- The release checker requires license, third-party, contribution, and security documents and
  rejects tracked checkpoints, archives, model weights, local absolute paths, and data binaries.
- Paper postprocessing and PDF QA enforce the upstream license's machine-generation disclosure.

## Remaining P1 work

- Replace trusted local `manager.pkl` checkpoints with a non-executable state representation. Pickle
  remains isolated to task-owned checkpoints because upstream AgentManager objects are not presently
  serializable as JSON.
- The current research control plane is unified behind `path-ai-scientist`; the legacy 24-stage
  demonstrator is compatibility/regression code and is absent from the main UI. Physical deletion is
  deferred until old-task migration evidence is no longer needed.
- Deterministic, evidence-manifested pathology figures now gate formal paper writing. The vendored
  AI-Scientist-v2 plotting prompt/code extraction is exposed only as an optional extension; template
  figures remain the offline and CI baseline.
- Add paired predictions, bootstrap confidence intervals, multiple independent seeds, per-class metrics,
  GPU warm-up/synchronization, repeated latency trials, throughput and memory measurements.
- A clean-clone reproduction is now a mandatory pending release gate. The real P0 task and prior
  pinned Docker test suite provide evidence but do not replace that final release-hardening step.

## Beta-candidate verification status

- Ruff and the complete pytest suite pass in the pinned Linux runner with the real local dataset;
  the clean source snapshot also passes, with real-data/GPU integration cases explicitly skipped.
- The deterministic illustrated-paper fixture compiles successfully in the pinned LaTeX runner and
  embeds its manifest-backed PNG.
- A clean source snapshot passes editable installation, Ruff, pytest, and the release checker with no
  tracked data, state, secret, oversized artifact, or local path violation.
- Fresh image rebuild is currently blocked before project layers by Docker Hub IPv6 authentication
  timeouts; existing pinned local images were used for verification. GPU evaluation and paid-provider
  smoke remain manual gates and were not run.

## Claim boundary for the existing paper

The existing result supports only a feasibility statement: hard routing executed one branch per sample
and used the high-resolution branch for a minority of samples. It does not establish an accuracy gain over
fixed-high resolution or a wall-clock efficiency gain. Dynamic latency was slower in the recorded single run.
