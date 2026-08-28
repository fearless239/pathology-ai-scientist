# Agent trace for pathology research workflows

Each workflow task writes an append-only `trace.jsonl` beside `task.json`,
`llm_usage.jsonl`, and the `artifacts/` directory. The trace records stage lifecycle,
latency, cost, retries, LLM usage metadata, failures, and artifact hashes. It intentionally
does not record prompts, model responses, images, dataset rows, or credentials.

## Quick use

Run a normal workflow. Then create a compact summary without another model or GPU call:

```bash
python -m pathmnist.trace state/workflow/<task-id>/trace.jsonl
python -m pathmnist.trace state/workflow/<task-id>/trace.jsonl \
  --output state/workflow/<task-id>/trace-summary.json
```

The summary reports completed and failed stages, total stage time, per-stage cost, and LLM
call count. The raw JSONL remains the source of truth and can later be exported to an
OpenTelemetry-compatible backend.

## Evaluation strategy

Use three test tiers so expensive end-to-end runs are the exception:

1. Component evaluation: replay fixed inputs against literature ranking, research-contract
   validation, experiment-plan validation, statistical checks, and claim/citation checks.
2. Trajectory evaluation: score a saved trace for stage order, retry loops, cost and latency
   budgets, forbidden test-set access, artifact completeness, and claim-to-evidence lineage.
3. End-to-end evaluation: run a small, fixed benchmark suite and review reproducibility,
   statistical validity, novelty, and manuscript quality.

Trace evaluation does not replace execution. It can prove that required controls ran and
identify where a workflow failed, but it cannot prove that a hypothesis is novel, code is
semantically correct, an experiment is scientifically valid, or a paper's conclusions are
true. Those properties need deterministic validators, selective reruns, and human or
independent-model review.

## Recommended next checks

- Assign every artifact a producer stage and every paper claim one or more artifact hashes.
- Record tool calls for literature search and sandbox execution, including input hashes and
  output references but not sensitive payloads.
- Add deterministic trajectory rules first: stage ordering, split discipline, one-time test
  access, budget limits, retry ceilings, and required artifacts.
- Build a 10-20 task frozen benchmark set with small datasets and cached literature results.
- Only after the local schema stabilizes, add OpenTelemetry export or a hosted trace viewer.

Keep high-cardinality or sensitive content in access-controlled artifacts. Store only hashes,
counts, safe identifiers, and relative artifact references in the trace.
