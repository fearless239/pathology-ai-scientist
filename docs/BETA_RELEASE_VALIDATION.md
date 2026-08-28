# Beta release verification — 2026-08-29

Status: **local preservation-release gates passed; remote publication is not performed**.

## Scope

The candidate preserves the existing application, upstream patches, dependencies, configuration,
test fixtures, and public interfaces byte for byte. Changes are restricted to release exclusions,
CI test-directory/compiler-image coverage, shell checkout line endings, and documentation.
No paid calls, GPU research, sealed-test evaluation,
task migration, or original-task resume are authorized by this release verification.

The existing `pathtest-001` archive passed the current acceptance validator with both PDFs required
before and after release preparation. Its 495 files remained unchanged. This is revalidation
of existing evidence, not a new live run. All 247 protected application, upstream, configuration,
dependency, fixture, script, Dockerfile, license, and interface files match the original baseline.
The current `upstream_v2` publication backend's real-service limitations remain as documented in
`UPSTREAM_PUBLICATION.md`.

## Required gates

- [x] Original and candidate full test results compared, with no unexplained regression or skip.
- [x] Existing Ruff check passes.
- [x] Original source and original task hashes unchanged; candidate business files unchanged.
- [x] Candidate acceptance of the original archive passes with both PDFs required.
- [x] Clean checkout installs with documented dependencies and passes offline tests.
- [x] Two deterministic Demo runs have identical artifact hashes.
- [x] Demo, orchestrator, runner, and publication images build from candidate files.
- [x] Demo container safety and UI health checks pass.
- [x] Illustrated fixture and English/Chinese publication templates compile.
- [x] All candidate files reviewed for release boundaries, secrets, and broken local README links.
- [x] Release checker passes against the candidate's tracked file set.

## Measured results

| Environment | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Original Windows Python 3.11 environment | 464 | 1 | 0 |
| Candidate in the same Windows environment, without private data | 457 | 8 | 0 |
| Independent Git clone in the same Windows environment | 457 | 8 | 0 |
| Fresh Python 3.11 virtualenv, native Linux container filesystem | 456 | 9 | 0 |

All four runs collected 465 outcomes. PyTorch is absent in these developer environments (one
module-level skip). Seven additional cases require the intentionally excluded real dataset.
One further Linux case skips construction of a live provider when no API key is supplied;
it is not a paid inference test. No test assertions or skip conditions were changed.

The clean Linux run installed `.[dev]` in a new virtualenv without system site packages, data,
task state, or provider credentials. Its Ruff and release checker passed, both Demo manifests
matched, and Git status remained clean. The existing compiler's log-quality gate passed for
the illustrated fixture, English ICML fixture, and the production Chinese preamble with BibTeX.
These fixtures prove compilation compatibility, not manuscript quality or new research success.
The README's Docker Compose build/start path also passed a local health check from a fresh Linux
clone; the temporary Compose project was stopped and removed afterward.

All 293 candidate files were scanned. Three credential-pattern matches were reviewed as two
documented placeholders and one shell environment-variable forwarding expression. The README
has no broken local file links; the Demo screenshot and curated example were reviewed separately.
This is a scoped release scan, not a claim of exhaustive security or legal certification.

## Environment corrections and remaining limitations

- The upstream journal requires temporary experiment paths below the repository root. CI and
  the developer commands now create `.test-tmp` and pass an explicit `--basetemp` there.
  The initial external-temp run failed this requirement; no application code was changed.
- A Linux run on a Windows bind mount produced one concurrent-read `Errno 61` failure (455
  passed, 9 skipped). The complete native-Linux rerun passed. Linux tests and new task state
  should use native Linux storage; the original archived task was not moved or resumed.
- Windows Git line-ending conversion broke bash before execution. `.gitattributes` now keeps
  shell scripts LF; the independent checkout's Demo verification passed without script edits.
- Docker images used separate validation tags. The publication build used the same Dockerfile
  instructions with only the parent image tag redirected to the newly built validation runner.
  No production image tag or original environment was replaced.
- Original upstream manifests describe baseline code, not the five pre-existing patched files.
  Provenance wording now distinguishes the two; no upstream patch was added by this release.

Full pytest may legitimately skip optional PyTorch and real-dataset tests when those inputs are
absent; counts and reasons must be recorded, not silently treated as full hardware coverage.
GitHub-hosted CI (including its Python 3.12 matrix entry) remains unexecuted until a separately
authorized push. No new real-service full research run, GPU training, or sealed-test evaluation
was performed. No remote or Release is created during local preparation.
