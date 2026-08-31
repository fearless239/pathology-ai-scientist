# Release checklist

Use this checklist for each candidate; record actual results and commit identity in the release notes.
Do not inherit a previous version's passing result without checking the affected behavior.

## Source boundary

- [ ] Review the staged diff; retain only framework code, required assets, regression tests, and operational documentation.
- [ ] Exclude datasets, weights, task state, provider responses, credentials, experiment logs, and generated papers.
- [ ] Preserve the upstream license, notices, and patch provenance; describe the project as source-available.
- [ ] Run `python -m pathmnist.release_check --repo .` against the tracked candidate.
- [ ] Check README/documentation links and installation commands.

## Executable candidate

- [ ] Install `.[dev]` from a clean checkout and run the documented Ruff and complete pytest commands.
- [ ] Explain every skip and failure; use native Linux storage for Linux concurrency tests.
- [ ] Run the deterministic Demo twice and compare artifact hashes.
- [ ] Build the relevant Docker images; check Demo safety, Compose startup, and UI health.
- [ ] Compile the illustrated CI fixture and check supported publication templates when affected.
- [ ] Revalidate existing archived evidence read-only when execution or acceptance behavior could be affected.
- [ ] For substantive execution changes, obtain separate authorization for any needed real GPU/provider validation.

## Publication

- [ ] Record changes, supported environments, known limitations, and performed versus unperformed validation.
- [ ] Confirm GitHub CI passes; mark Beta releases as prereleases.
- [ ] Never repeat a sealed test, migrate an existing task, or relax evidence gates merely to validate a release.
- [ ] Keep historical validation logs and private experiment evidence outside the public repository.
