# Gate A status

Date: 2026-08-17 (final; supersedes the 2026-08-16 pre-paid draft)

## Outcome: Gate A (M3) passed

Gate A is complete. The offline fixture chain passed twice, and the one remaining paid
OpenRouter checkpoint then ran and passed every acceptance criterion.

Authoritative paid evidence:

- Acceptance report: `runs/gate-a-paid/20260816T211625Z_05482d1a/acceptance.json`
- `gate_a_passed: true`; all ten criteria true, including `real_openrouter`
- Budget: USD 0.0113 spent of the USD 2.00 hard cap; 5 requests, all settled
- Writer `z-ai/GLM-5.2` (via OpenRouter), reviewer `z-ai/GLM-5.1` - independent
- Paper PDF and independent review are archived in the same run directory

## Offline repeatability evidence (unchanged)

- Repeatability report: `runs/gate-a-image-contained-final/offline-repeatability.json`
- Runs: `20260816T062408Z_dd3b3432` and `20260816T062415Z_12b47069`
- Artifact manifest: 19 entries, 0 hash or byte-count mismatches
- Unit tests and formatting were clean at that revision (18 tests, Black 25.1.0)

## Immutable environment identifiers (offline evidence)

- Runner image: `sha256:9ab85babea1427c3bed976cece988a62493db25c3b2e695154c281dcb857f3ef`
- Orchestrator image: `sha256:5bfa4781898fcd8ec1b7527a85906f620096961822b6c35a7e6a8c6b4234e52a`
- Python base manifest: `sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822b6dfc60c317`
- Docker CLI base manifest: `sha256:851f91d242214e7c6db86513b270d58776379aacc5eb9c4a87a5b47115e3065c`

Python direct and transitive dependencies are pinned in
`docker/orchestrator-requirements.lock` and `docker/runner-requirements.lock`.

## Runner image 0.2 revalidation (2026-08-19)

`docker/runner.Dockerfile` gained `texlive-xetex` and `fonts-noto-cjk` so the pinned
image can compile the Chinese M5 paper with XeLaTeX and Noto Serif CJK SC. The runner
tag therefore moved from 0.1 to 0.2; the archived 0.1 evidence above is unchanged and
the old image remains available locally under its recorded digest.

Offline fixture acceptance on runner 0.2 (digest
`sha256:2cdf411b50934e0c4e39706a8ae2400b44210da4f59d02f1c976305b8d272868`) passed
twice:

- Runs: `runs/gate-a-offline/20260819T122319Z_02a21b85` and
  `runs/gate-a-offline/20260819T122327Z_b6a19cab`
- Every offline criterion is true in both runs (`offline_ready: true`);
  `real_paid_provider` is false by design in fixture mode
- Both compiled PDFs are byte-count identical (65,125 bytes); the remaining manifest
  hash differences are confined to files that embed run IDs or timestamps

## Provider transition (2026-08-17)

The user's OpenRouter balance is exhausted. Future LLM calls therefore move to the
Paratera LLM API gateway (`https://llmapi.paratera.com/v1`), which is
OpenAI-compatible.
This switch does not invalidate the archived Gate A evidence above: the M3 artifacts,
ledger, and acceptance report are local and immutable records of that run.

What changed in the code:

- A provider abstraction (`provider` config section) now selects `openrouter`,
  `zhipu`, or `openai_compatible`; the legacy `configs/gate_a.yaml` keeps working
  unchanged.
- New `configs/gate_a_llm.yaml` pins the gateway endpoint, `PARATERA_API_KEY` env var,
  model IDs (`GLM-5.2`, `GLM-5.1`), context lengths, and nominal CNY list prices,
  because the gateway publishes no priced machine-readable catalog. The USD ledger is
  an accounting proxy that keeps the reservation discipline. Verify the base path and
  model IDs against the gateway's `/v1/models` response for this account.
- `ZhipuProvider` keeps the same reservation/settle/response-cache ledger semantics and
  omits OpenRouter-specific headers and body fields. Structured output uses documented
  `tools` + `tool_choice="auto"` with a content-JSON fallback, and retries once without
  tools if the endpoint rejects tool parameters with HTTP 400.
- CLI: `--provider openai_compatible`; script:
  `./scripts/gate-a.sh llm-preflight` and `llm-paid`. Paid inference still requires
  `--confirm-paid-smoke`.
- Acceptance criterion generalized from `real_openrouter` to `real_paid_provider`
  (true for either live provider). Fixture/offline semantics are unchanged.

Before the first Zhipu paid smoke:

1. Rebuild images: `./scripts/gate-a.sh build`
2. Re-run offline acceptance: `./scripts/gate-a.sh offline`
3. Export `PARATERA_API_KEY` in WSL, then `./scripts/gate-a.sh llm-preflight`
4. Confirm the gateway model list matches `configs/gate_a_llm.yaml`
5. Only then, with explicit approval, run `./scripts/gate-a.sh llm-paid`

