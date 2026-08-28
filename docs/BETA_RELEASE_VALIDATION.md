# Beta release verification — 2026-08-29

Status: **verification in progress; not approved for public release**.

## Scope

The candidate preserves the existing application, upstream patches, dependencies, configuration,
test fixtures, and public interfaces byte for byte. Changes are restricted to release exclusions,
CI compiler-image coverage, and documentation. No paid calls, GPU research, sealed-test evaluation,
task migration, or original-task resume are authorized by this release verification.

The existing `pathtest-001` archive passed the current acceptance validator with both PDFs required
before release preparation. This is revalidation of existing evidence, not a new live run.
The current `upstream_v2` publication backend's real-service limitations remain as documented in
`UPSTREAM_PUBLICATION.md`.

## Required gates

- [ ] Original and candidate full test results compared, with no unexplained regression or skip.
- [ ] Existing Ruff check passes.
- [ ] Original source and original task hashes unchanged; candidate business files unchanged.
- [ ] Candidate acceptance of the original archive passes with both PDFs required.
- [ ] Clean checkout installs with documented dependencies and passes offline tests.
- [ ] Two deterministic Demo runs have identical artifact hashes.
- [ ] Demo, orchestrator, runner, and publication images build from candidate files.
- [ ] Demo container safety and UI health checks pass.
- [ ] Illustrated fixture and English/Chinese publication templates compile.
- [ ] All candidate files reviewed for release boundaries, secrets, and broken local README links.
- [ ] Release checker passes against the candidate's tracked file set.

Full pytest may legitimately skip optional PyTorch and real-dataset tests when those inputs are
absent; counts and reasons must be recorded, not silently treated as full hardware coverage.
GitHub-hosted CI remains unexecuted until a separately authorized push. No remote or Release is
created during local preparation.
