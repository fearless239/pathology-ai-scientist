# Source provenance

## Input baseline

- Input supplied by the project author: an unpacked `AI-Scientist-v2` source snapshot
- Copied, unmodified baseline: `vendor/AI-Scientist-v2`
- Upstream project named by the supplied README: `SakanaAI/AI-Scientist_v2`
- License retained verbatim at `vendor/AI-Scientist-v2/LICENSE`
- The supplied directory did not contain Git metadata, so an upstream commit is not guessed.
  `UPSTREAM_MANIFEST.sha256` is the authoritative identity for this local baseline.

The Gate A implementation lives outside `vendor/AI-Scientist-v2`. No file in the vendored
directory is intentionally patched. This makes the source boundary and all local adaptations
auditable.

## Gate A scope

Gate A adds the minimum provider, budget, isolated execution, paper compilation, and independent
review adapters required to exercise the upstream engineering chain. Its
`gate_a.upstream_bridge.UpstreamMinimalBFTS` directly imports and uses the upstream
`ai_scientist.treesearch.parallel_agent.MinimalAgent` implementation.

## Pathology specialization

The `pathmnist` package lives outside the vendored baseline. It adds PathMNIST dataset validation,
split discipline, computational-pathology prompts, a supported-intervention registry, experiment
artifact contracts, claim-bounded analysis, and a reviewed-paper workflow. The
`pathmnist.upstream_adapter.PathologyAIScientistV2Adapter` imports the upstream prompt compiler,
`MinimalAgent`, and `AgentManager` in the pinned Python 3.11+ container runtime. Each workflow task records
the exact boundary and upstream manifest digest in `task_created/framework.json`.
