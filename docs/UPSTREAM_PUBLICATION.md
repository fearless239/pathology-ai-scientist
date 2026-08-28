# AI Scientist v2 publication adapter

## Rollout

New autonomous tasks persist `publication_backend=upstream_v2`. Missing fields mean
`legacy_local`; initialization/resume preserves this interpretation. No existing task
is migrated and there is no failure fallback. In particular, pathtest-001 is unchanged.
The experiment runner, freezing and single-use test protocol are outside this patch.

Build the additional local compiler image before creating production tasks:

```sh
docker build -f docker/publication.Dockerfile -t path-scientist-publication-runner:0.1 .
```

It extends the existing runner with the Times fonts required by ICML. Legacy tasks
retain their old image. Compilation runs with networking and shell escape disabled,
a 300 second timeout, named-container cleanup, and BibTeX between engine passes.

## Reuse boundary

`ai_scientist.perform_writeup.writeup_step` and `reflection_step` contain the original
native prompts and response handling. Both the native entry point and the local adapter
call these functions. Review calls native `perform_review`, with one ensemble member,
no examples and no reviewer reflection. Query dependencies are per-call; there is no
global client monkeypatch. The local provider retains accounting and uncertain-response
barriers. No additional citation search, VLM review or generated plotting code is run.

`pathmnist/upstream_publication.py` owns evidence mapping, scoped model gateways,
artifact validation, translation, compilation and archive integration. English LaTeX
is authoritative. Chinese text nodes are translated while protected LaTeX nodes and
numeric sequences are preserved. Verified literature is prefilled into references.bib;
the ICML template and default eight-page writing target are retained. Page count alone
does not fail a task.

## Recovery and artifacts

`paper/publication_manifest.json` names the backend/version and each language/stage's
source and hashes. Final PDF links and acceptance use this manifest instead of legacy
Markdown filenames. Each input fingerprint has its own `paper/versions/<sha256>`
directory. Input identity includes evidence, template, native writing/review code,
adapter and model configuration. Changed inputs require explicit version review.

Draft, raw review, each reflection, final selection and translation have durable cache
receipts. Raw native review is retained beside a display projection; invalid JSON is
not replaced by default scores. There are at most two scientific reflection steps.
Compilation uses isolated copies and durable repair counters (at most two repairs),
rejecting content-changing repairs. PDF receipts avoid recompilation after successful
commit. Archive failures rebuild only the archive using already committed PDFs.

## Verification and limits

`tests/test_upstream_publication.py` executes actual native drafting, review and
reflection with external services replaced. It covers stage interruption, missing
response projection, archive-commit interruption, scoped histories, backend routing,
unknown figures/citations, decimal changes, malformed artifacts, budget failure and
timeout cleanup. The publication-to-archive integration stubs pre-publication evidence
acceptance; it does **not** claim a new real experiment or full real-service run.

Offline fixed English ICML and Chinese XeLaTeX manuscripts were compiled with BibTeX,
rendered with Poppler and inspected on every page (one page per language). This verifies
the compiler/template setup, not arbitrary generated manuscript quality.

Known validation boundary: decimal matching and protected translation tokens are
conservative checks, not semantic proof that all claims are supported. Long-form
translation quality, scientific conclusions, unusual LaTeX structures, real model
responses and recovery across real paid requests still need the separately authorized
frozen-evidence acceptance run. Compilation success is not publication-quality proof.
No paid model requests, training, or held-out evaluation were performed for this patch.

The corresponding baseline and patched hashes are in
`UPSTREAM_PUBLICATION_PATCHES.json`; no upstream version upgrade was performed.
