# Public Beta Release Checklist

- [ ] Run `python -m pathmnist.release_check --repo .` on the release commit.
- [ ] Run the deterministic Demo Mode twice and confirm identical artifact hashes.
- [ ] Build `docker/demo.Dockerfile`; run `scripts/verify-docker.sh IMAGE` and confirm every check passes.
- [ ] Launch `docker compose up --build` from a clean clone and verify the local demonstration workflow.
- [ ] On Windows/WSL, verify `scripts/build-demo-wsl.sh` from a Windows-mounted checkout and confirm Docker Desktop is not required.
- [ ] Build the pinned orchestrator and PathMNIST runner images from a clean clone.
- [ ] Run Ruff and the complete pytest suite in the pinned environment.
- [ ] Run one offline fixture workflow twice and compare immutable artifact hashes.
- [ ] Generate deterministic template figures and verify every manifest entry resolves to source evidence.
- [ ] Compile the paper fixture with at least one local figure and no unresolved figure references.
- [ ] Run one real PathMNIST task through research, preflight, experiment, freeze, approved test,
      postprocess, PDF QA, and archive.
- [ ] Confirm every reference resolves to its recorded DOI, PMID, Corpus ID, or URL.
- [ ] Confirm the acceptance report passes with both PDFs required.
- [ ] Confirm no dataset, checkpoint, task state, API response, key, cache, or local absolute path is tracked.
- [ ] Confirm root `LICENSE` exactly preserves the AI Scientist Source Code License.
- [ ] Review `THIRD_PARTY_NOTICES.md` and verify upstream, AIDE, MedMNIST, and document-asset attribution.
- [ ] Confirm the README describes the project as source-available, not MIT/Apache/OSI open source.
- [ ] Confirm both final manuscripts prominently retain the mandatory AI-generation disclosure.
- [ ] Confirm PathMNIST is downloaded from the official distribution and is absent from Git and release archives.
- [ ] Tag GPU and paid-provider smoke tests as manual; never run them on pull requests.
- [ ] From an empty directory, clone the release candidate and repeat install, tests, Docker build,
      offline fixture, illustrated PDF, and release checker commands exactly as documented.
- [ ] Publish known limitations: patch classification only, no clinical claims, no autonomous deployment.
- [ ] Review the curated case-study bundle; confirm it contains no raw provider reply, dataset, model weight,
      checkpoint, secret, task state, or absolute path.
- [ ] Keep the manuscript sample subordinate to the acceptance report and retain its non-peer-reviewed warning.
