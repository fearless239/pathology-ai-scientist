# Architecture and Specialization Boundary

## Current research workflow

```text
direction + dataset
  -> dataset discovery/validation (test omitted from research view)
  -> research understanding + verified literature
  -> generated ResearchExecutionContract -> explicit user approval
  -> experiment specification bound to the approved contract
  -> upstream AI-Scientist-v2 AgentManager through pathology prompt/sandbox adapters
  -> contract-bound baseline/intervention/ablation repeats + reproducibility manifests
  -> trusted contract-fulfillment/statistics gate -> immutable comparison bundle
  -> explicit approval -> one sealed-test attempt covering every frozen arm/seed
  -> evidence bundle -> deterministic figures (optional upstream LLM extension)
  -> writer -> independent reviewer -> revision -> translation
  -> LaTeX compile + PDF QA -> acceptance validator -> evidence archive
```

`pathmnist.autonomous_acceptance` is the sole authority for formal stage acceptance. Individual
commands may write their own artifacts, but cannot make an incomplete task acceptable.

## Upstream and local responsibilities

| Concern | Upstream AI-Scientist-v2 | Local specialization |
|---|---|---|
| Search agent and journals | AgentManager, MinimalAgent, stages | Pathology goals, dataset contract, checkpoints |
| Research completion | Generic stages and good nodes | Approved execution contract and deterministic fulfillment gate |
| Generated experiment code | Prompted code search/debug loop | Network-disabled Docker execution and result validation |
| Dataset | Generic upstream assumptions | Discovered pathology arrays/groups, split and leakage checks |
| Model choice | Agent generated | Not fixed locally; manifest and claim constraints required |
| Literature | Upstream Semantic Scholar concept | Verified identifiers, durable evidence, formal-paper gate |
| Test split | No local sealed policy | Frozen comparison bundle, durable approval, irreversible one-time attempt |
| Figures/paper | Plot prompting, code extraction, writing concepts | Deterministic pathology templates, evidence manifest, independent review, PDF gates |
| Cost/provider | Upstream model calls | Full-mode $8 cap, environment-only key, one persistent task ledger |

## Compatibility path

The older `pathmnist.workflow` 24-stage demonstrator remains only for offline fixtures and prior-task
compatibility and is absent from the main UI. It is not the formal public-beta state machine. Its artifacts may be migrated into the current workflow only
through explicit validators; stage names alone do not prove that work occurred.

## Stage transaction rule

Every formal stage follows the behavioral contract `validate inputs -> perform work -> validate outputs
-> commit task state`. A crash before the final state write leaves the stage incomplete. Restart logic may
reuse immutable validated artifacts, but it may not synthesize completion flags or skip predecessors.

`path-ai-scientist` is the sole public control plane. It stops before paid LLM work, sealed-test
execution, and Docker PDF compilation until the corresponding explicit authorization is supplied.
The `figures_generated` stage emits a versioned plan and manifest; formal papers may reference only
manifest-backed local figures whose source artifacts and fields are recorded.

## Upstream-centered refactor (2026-08-28)

The production executor remains AI-Scientist-v2 `AgentManager.run`: one search
journal, one stage loop and one checkpoint authority. `stage_policy.py` centralizes
local roles, stage goals and execution budgets. New-run prompts and restored
manager goals consume that policy. The legacy compatibility budget imports remain
available; no task-name branch is used.

Stage 3 permits up to four training launches within a 30-total-epoch node. Thus a
requested subset search and final full-data fit can be one upstream experiment
node. `training_budget.py` statically counts bounded loops over ordinary training
helpers and explicit epoch arguments. Unknown helper loop bounds and unspecified
multi-launch epoch budgets fail preflight. This analysis is conservative and not a
proof for arbitrary Python; Docker still enforces time and resource limits.
Baseline retraining by the intervention node and full-method retraining by an
ablation node remain prohibited. The 30-epoch ceiling is shared with the prompt,
not an implicit permission to expand the experiment budget.

Local main/substage acceptance for stages 1–4 uses the same checks. Missing
optional VLM plots no longer causes an additional completion-model call. Reaching
a search limit is not scientific success. Loading checkpoints refreshes policy
but does not automatically grant a retry window. Completed artifact reuse still
requires the existing evidence gates.

The experimental `execution_plan.py` callback scheduler is NOT a production
backend and is not imported by the execution entry points. Explicit approved DAG
contracts fail with a migration-required diagnostic; ordinary multi-step research
prose uses upstream nodes rather than being rejected for not having a DAG.
`docs/MULTISTEP_EXECUTION.md` records its earlier experimental status, not the
current production roadmap.

Recovery granularity is an upstream node. This refactor does NOT promise automatic
resumption inside a generated training loop. Such reuse requires independently
validated training checkpoints; partial metrics alone are insufficient.

Validation: 124 scoped offline tests, including the real upstream run loop with
fake experiment execution, plus budget/role and prior evidence regression tests.
No paid generation or GPU training was performed for this refactor. The existing
failed task and its approved contract/checkpoint were not rewritten. Its last
program has four launches and 39 epochs and therefore still requires a bounded
repair before it can be run under the 30-epoch policy.

### Training limits and early stopping

New autonomous executions must include `max_epochs` (configured per-candidate
limit), `epochs` (actual epochs completed by the selected candidate, not the
best-checkpoint epoch), and `early_stopping` in the schema-v1 experiment manifest.
The stopping policy is either `{"enabled": false}` or an enabled policy with
`monitor` (`validation_loss` or `validation_metric`), `mode` (`min` or `max`),
positive integer `patience`, and finite nonnegative `min_delta`. Additional
unreported convergence or time-based success exits are not allowed.

Tuning compares limits and the entire stopping policy against the baseline;
actual completed counts may differ. Each candidate history is bounded by the
configured limit, and the selected history must match the manifest's completed
count. With stopping disabled, candidates must complete the full limit.
Architecture and other baseline-control checks remain in force.
Exported execution controls retain both fields and the stopping policy; final
baseline/proposed comparisons likewise compare configured limits rather than
actual completed counts. Policy metadata is declarative evidence, not a proof
that arbitrary generated Python implemented the declared stopping algorithm.

Historical manifests without these fields remain readable and retain strict
`epochs` equality during tuning acceptance. Mixed legacy/new policy evidence is
rejected; no limit is inferred from a historical completed count and immutable
evidence is never rewritten. Such runs need regenerated baseline/tuning evidence
before using the new policy. This change does not automatically resume jobs.
