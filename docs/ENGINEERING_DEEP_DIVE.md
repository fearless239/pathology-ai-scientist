# Engineering Deep Dive

## Control plane

`path-ai-scientist` is the public control plane. The research workflow advances through ordered,
validated transactions. Each transition follows `validate inputs → perform work → validate outputs →
commit task state`; a failed output validator prevents the stage marker from advancing.

Paid LLM work, research-contract approval, sealed-test evaluation, and PDF compilation are distinct
authorization boundaries. UI controls call the orchestrator rather than editing task state directly.

## Extension boundary

`pathmnist.framework` publishes three beta protocols:

- `DatasetAdapter`: discovery, fingerprinting, split validation, and research/test isolation.
- `ExperimentBackend`: sandbox preflight, generated experiments, candidate freeze, and sealed test.
- `ArtifactValidator`: manifest, trusted-metric, and publication validation.

`ResearchTaskConfig` carries provider-neutral intent, adapter name, paths, budget, model roles, seed,
and permissions. PathMNIST is the only reference implementation in beta; the interfaces are not yet
1.0-stable.

## Idempotency and recovery

Artifacts are written before a task transaction is committed. Valid immutable outputs may be reused
after restart. Provider request IDs and a persistent budget ledger prevent an accepted response from
being charged twice. A stage fails after bounded retries and requires an explicit resume/repair action;
completion flags are never synthesized to bypass missing evidence.

## Sandbox and secrets

Generated experiment code runs inside a non-root Docker boundary with networking disabled. API keys
are read from the orchestrator environment and are not mounted into experiment containers. The local
Demo image is additionally read-only, drops Linux capabilities, and enables `no-new-privileges`.

## Evidence provenance

Dataset fingerprints, generated-code hashes, execution receipts, experiment manifests, candidate
hashes, trusted statistics, figure manifests, and publication disclosures form the evidence chain.
Acceptance validates this chain independently of the writer model. Missing or contradictory evidence
downgrades the output to a failure diagnosis or blocks publication.

## Cost model

Every paid request reserves a worst-case amount before execution. Settled responses are cached by
request ID, unused reservations are released, and a run-wide USD ceiling remains authoritative. Demo
Mode makes no provider call and has a zero-dollar budget.
