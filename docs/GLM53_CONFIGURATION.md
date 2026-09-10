# GLM model configuration history and estimated budget

> Active configuration update, 2026-09-10: the live research roles were switched back
> to the previously stable GLM-5.2/GLM-5.1 combination after repeated GLM-5.3 HTTP 500
> responses. GLM-5.2 handles contract, experiment, plotting, and writing calls; GLM-5.1
> remains the independent reviewer. The GLM-5.3 notes below are retained as incident
> history and are not the active configuration.

User-authorized configuration, 2026-09-05. The active full-research file is
`configs/gate_a_llm.yaml`; the legacy paper-only configuration was not migrated.

| Role | Model | Input ceiling | Completion ceiling |
|---|---|---:|---:|
| Contract / ideation | GLM-5.3-Flash | 12000 | 8000 |
| Experiment code | GLM-5.3 | 24000 | 12000 |
| Plotting | GLM-5.3-Flash | 12000 | 8000 |
| Paper writing | GLM-5.3 | 32000 | 24000 |
| Review | GLM-5.3-Flash | 32000 | 24000 |

These are per-request limits, not separate stage dollar allocations. All stages share
the task ledger. New tasks receive a $50 **estimated-accounting** limit; existing tasks
retain their recorded limit (missing legacy fields mean $8). Experiment execution no
longer silently forces the limit to $8. Generated contracts bind the task's limit before
their hash is calculated. Existing tasks and outstanding reservations were not edited.

Both model rates are user-approved placeholders: CNY 10 per million input tokens and
CNY 40 per million completion tokens, using the existing 7.1 CNY/USD conversion.
They are NOT verified provider prices, NOT actual charges and NOT guaranteed upper
bounds on real cost. Historical `actual_cost_usd` response fields are calculated using
configured rates; the field name does not make them provider-reported charges.
The UI and active configuration identify the estimates. Replace them when billing
information becomes available. The configured 65536-token context is a local admission
ceiling, not a verified endpoint maximum. The experiment-code completion ceiling was
reduced to 12000 after repeated 600-second read timeouts at 24000; errors must not be
hidden by automatic model fallback.

Short connectivity checks using the updated WSL credential succeeded on
`https://llmapi.paratera.com/v1`: GLM-5.3 returned OK in 3.22 seconds (46 total tokens),
Flash returned OK in 2.01 seconds (19 total tokens). Paratera rejected disabled
thinking for both configured GLM-5.3 variants; their adapter now uses
`reasoning_effort=low` without the `thinking` parameter. Other models retain their prior
parameter handling. These checks
do not prove structured contracts, long code or full manuscripts work in production.

The active GLM-5.2 provider response timeout remains 600 seconds. A transport timeout
stops the current worker immediately, and its full reservation remains charged against
the task cap as `outcome_unknown`. If the user explicitly starts the task again, the
provider permits one new request with a distinct recovery ID; a second uncertain outcome
fails closed and requires manual provider reconciliation. Legacy `reserved` entries are
eligible for this migration only when a durable, prompt-free diagnostic records
`state=outcome_unknown`.

Restart the BAT-launched web service to load the current environment and code, then
create a new task. Verify the task.json limit is 50 and the generated contract's
resource_plan agrees. Writer and reviewer remain different models. No GPU, epochs,
search-node limits, timeout policy or automatic retries were increased by this change.
No research was started. The old pathtest-002 unknown-billing request still requires
reconciliation; a larger budget does not resolve it. No old task should be deleted to
conceal its usage.

Verification: targeted config/provider/budget/publication tests passed (63 tests);
final init/contract/provider checks passed (16 tests). Ruff and diff whitespace checks
passed. Full regression: 473 passed, 2 failed, 1 skipped. The failures are legacy
`test_experiment_summary_handlers` and `test_paper_workflow_handlers_use_llm_usage`:
the newly present dataset enables these integration tests, but saved baseline training
outputs are absent (`Incomplete main:baseline runs; found 0`). They were previously
skipped when the dataset was absent. The skipped module requires torch. No training
was run to manufacture missing outputs. Full-test log: `../tmp/budget50-full.txt`.

## Structured contract response fix

A real Flash response omitted the required `unsupported_reasons` array. Previously
the provider accepted any JSON object and contract conversion raised a KeyError.
The provider now validates fresh AND cached JSON against the supplied schema before
returning it. Its existing three-attempt limit remains unchanged; field-level feedback
and the schema accompany retries, each with the existing distinct request ID and cost
record. The original malformed response remains intact. No default fields are inserted.
Contract conversion also validates independently, producing ResearchContractError for
malformed input. Offline checks: 48 provider/contract/budget tests plus 38 publication,
request-ID and orchestration tests passed; Ruff and diff checks passed. No live retry
was performed. Restart the UI service and continue the interrupted task to let the user
initiate the remaining bounded attempts; deleting the cache or recreating the task is
not required for this missing-field error. Transport-outcome-unknown requests remain
blocked as before, and compilation/training behavior is unchanged.
## 请求诊断

新发起的 Paratera 请求会在任务响应目录的 `diagnostics/` 子目录写入一个同名 JSON 文件。例如：

`state/workflow/<task-id>/research/responses/diagnostics/<request-id>.json`

该文件记录模型角色、模型名、开始和结束时间、耗时、超时配置、HTTP 状态、服务端请求 ID、响应大小、异常分类以及服务端返回的 token 用量。超时请求没有服务端 usage，因此 `input_tokens` 保持为 `null`，同时保留输入字符数用于比较请求规模。诊断文件不保存 API Key、系统提示词或用户提示词。
