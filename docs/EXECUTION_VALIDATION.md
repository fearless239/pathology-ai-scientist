# Execution lifecycle validation

The paid adapter uses a 3600-second sandbox ceiling and a separate 5400-second
worker deadline, including generation and host validation. A worker deadline is
a fail-stop interruption: it does not silently schedule another candidate.
Cleanup stops worker processes, removes task-registered sandboxes, then releases
GPU ownership. Unconfirmed sandbox cleanup must not become a retryable node.

Task ownership uses a non-blocking OS file lock. Kernel locks are released when
the owning process exits; an existing lock file alone does not mean a live task.
Old processes started before this feature do not hold the new lock and must be
stopped before using the updated runner.

The approved contract fixes the search metric. Resuming removes duplicate node
IDs, and the previous-stage parent is only inserted when absent. Historical
experiment artifacts are not deleted by this operation.

Streaming output is enabled for paid runs. Redacted output is flushed to the
terminal and to `.active-sandboxes/*.log` under the experiment workspace.
The matching progress JSON contains elapsed time, last output time, and the last
line (including epoch/seed details when the generated program emits them).
These diagnostic files are not scientific evidence or epoch-resumable weights.

## Offline checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_execution_lifecycle.py tests/test_autonomous_evidence.py tests/test_policy_runner.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\path-ai-scientist-demo.exe --output .demo/lifecycle-validation-a
.\.venv\Scripts\path-ai-scientist-demo.exe --output .demo/lifecycle-validation-b
```

The lifecycle test invokes the actual upstream `ParallelAgent.step` and adapter
cleanup with controlled futures/process handles. It verifies that a running
future is not treated as completed and that GPU ownership is retained until
cleanup. A real local Python subprocess verifies incremental output retention
on timeout. These tests do not exercise Docker daemon failures or GPU timing.

Both synthetic demos passed with identical five-artifact hashes in local
validation. The demo is a separate fixture, not proof that a paid research tree
has completed.

Before another full research run, use a separate small PathMNIST smoke task:
fixed stratified training subset, native 64x64 input, ResNet-18, one seed and
1–2 epochs. Keep baseline/intervention controls identical, require no positive
effect, and do not access the official sealed test. This remains a separate
validation step, not performed by the offline checks above.

## Real GPU smoke result (2026-08-27)

The fixed-code smoke was subsequently run on the local RTX 5070 Ti Laptop GPU
with the existing runner/orchestrator Docker images. It used 32 training and 8
validation samples per class (288/72), one seed, native 64x64 inputs and two
epochs per arm. The official test arrays were neither loaded nor mounted.

Both arms passed the project Docker interpreter and host prediction verification
and were added to a smoke journal. Both reported UID 65532, CUDA available,
no API-key environment variables and no default network route. A real 3-second
container timeout was cleaned up. A second invocation reused both hash-verified
training snapshots, verified equal controls and rejected duplicate task ownership.
The final Docker/GPU process check was empty.

Local evidence: `state/smoke/gpu-lifecycle-20260827/smoke_report.json` (not tracked).
Reusable driver: `scripts/gpu-lifecycle-smoke.py`; fixed experiment:
`scripts/fixtures/gpu_smoke_experiment.py`.

This verifies the fixed-code interpreter → trusted metrics → journal boundary,
not LLM-generated tree search, formal multi-seed statistics, or full scientific
acceptance. The tiny-subset accuracies are intentionally not research claims.
No LLM calls or sealed-test evaluation were performed.

## LLM-controlled patch GPU smoke (2026-08-27)

`scripts/llm-gpu-smoke.py` subsequently made one GLM-5.2 request through the
configured Paratera adapter. It generated a torchvision augmentation expression
(color jitter, rotation, horizontal and vertical flips). The host validated its
AST and inserted it into the fixed training/evidence harness. No generated code
was executed on the host. The request had no automatic retries, a stable cache
ID, and a reservation below the additional USD 0.25 cap in the existing USD 8
task ledger. Costs use the configured pinned price estimate, not a billing receipt.

The candidate completed real GPU training on the same 288/72 subset, seed 0,
two epochs and native 64x64 inputs. The existing hash-verified baseline was
reused, not retrained. Host prediction verification, manifest control equality
and execution-result classification passed. The ledger recorded USD 0.0003876056
for the request. Baseline/candidate validation accuracies were 0.1667/0.1111;
these deliberately short runs do not test the scientific improvement hypothesis.

Local report: `state/smoke/llm-gpu-20260827/smoke_report.json` (not tracked).
This validates a constrained LLM patch → Docker/GPU → trusted evidence boundary,
not arbitrary whole-program generation, the complete parallel tree-search loop,
formal stage 3 recovery, multi-seed research acceptance, or PDF generation.
The formal task remained interrupted; the sealed test was not accessed.

## Stage-3 process-pool and checkpoint smoke (2026-08-27)

`scripts/stage3-recovery-smoke.py` exercised the real stage manager, parent
selection, `ParallelAgent.step`, process-pool worker, plan/code extraction,
MethodSpec attachment, Docker/GPU interpreter, trusted metric ingestion and
child journal insertion. Generation was replayed from the preceding smoke's
cached candidate; the progress summary was a fixture. No live provider or key
was supplied. Training used the same small subset and two epochs, not the
formal task's data volume or repeat plan.

The driver saved its own new checkpoint after a valid child was durable, raised
an intentional interruption, then reloaded through the project recovery loader.
Stage 3 resumed to the stage-4 entry without adding a node or rerunning training.
During resume, any call to the sandbox runner was configured to raise, making
the no-retraining check explicit. Execution stopped before any ablation worker.
The first attempt stopped before training because the replay fixture did not
account for the upstream summary query; that fixture was corrected, not the
production completion gate.

Local evidence: `state/smoke/stage3-recovery-20260827-c/smoke_report.json`.
This covers a durable-child interruption, not abrupt mid-epoch process death,
live free-form code generation, multi-seed recovery, or formal research acceptance.
The original research task was not resumed and no sealed-test data was used.


## Dataset interface and generated import preflight

Research NPZ views use canonical `{split}_images`, `{split}_labels`, and
`{split}_sample_ids` keys. The exporter and agent task description share the
same key builder. Before a paid run (including checkpoint resume), the task
description is compared to the actual mounted NPZ; a mismatch aborts setup.
Source aliases such as `val_images` must never be advertised as mounted keys.
Manifest inputs advertise their manifest interface rather than fictitious NPZ keys.

The Docker runner also rejects provably invalid unconditional NPZ subscripts
before launching a container. Each generated script then runs through a trusted
import preflight in the actual experiment image, under the same offline sandbox
limits. Missing direct top-level imports produce `IMPORT_PREFLIGHT_FAILED`,
installed version information where available, and similar exported names.
The interpreter returns this as a repairable `ImportPreflightError` node.
The original generated script remains unchanged for hashing and audit.

These are bounded checks, not proof of arbitrary generated Python correctness:
optional/conditional imports, function-local imports, dynamically computed API
names, and dynamic/conditional dataset access retain runtime validation. The
checks do not instantiate models, download pretrained weights, or authorize a
fallback to random initialization. Pretrained artifact provisioning remains a
separate requirement for transfer-learning tasks.

### Conditional keys and code-generation recovery

The task manager now explicitly carries the research-view interface and approved
contract into its curated worker description; merely adding fields to the saved
task JSON is insufficient because upstream only renders selected fields.

The dataset guard propagates simple constants, literal path variables, membership
tests on mounted `data.files`, and ternary key selections. A key that resolves to
`None` or a missing field is rejected before launching Docker. Unknown branches
invalidate inferred assignments instead of assuming a value; this remains bounded
static analysis, not execution of generated Python on the host.

Both runtime agent types require complete, syntactically valid fenced Python.
Prose-only repair responses get specific format feedback within the existing retry
limit. Exhaustion raises `CodeGenerationError`; prose is never returned as runnable
code. Code-only responses with valid fences no longer fail just because a prose
plan is missing. No automatic experiment restart or additional retry budget is added.

Code generation uses a dedicated system output contract and a nonempty user task
message. Retries include a bounded excerpt of the rejected response and specific
parsing feedback. This replaces the upstream all-system/empty-user request shape;
it improves the request contract but does not guarantee model compliance. Offline
parser tests alone do not establish successful live code generation.

## Model input and checkpoint runtime contract

New contracts record unambiguous explicit dimensions in
`execution_requirements.input_sizes` (height/width pairs). Existing approved
contracts retain their hashes: dimensions are derived from the original research
question when unambiguous. Multiple mentioned dimensions without an explicit
input-size list fail closed pending clarification; they are not guessed from the
dataset's native resolution. The contract review UI displays the effective sizes.

The Docker launcher instruments PyTorch model calls in both training and
inference, checking actual image tensor dimensions at model entry. Preprocessing
must resize inputs to the approved size; a manifest alone is not evidence.
Inference mode is determined by the mounted view, not a generated boolean.
It requires a nonempty `/workspace/model_checkpoint.pt`, successful `torch.load`
of that file, and complete `load_state_dict` restoration before image inference.
Merely opening the file, loading unrelated/random state, partially restoring a
model, or suppressing a failed check is rejected. Successful exit without an
observed image-model call is also rejected. Environment-only sandbox preflight
explicitly skips model execution checks; this is a host option, not a generated
code setting.

These hooks detect accidental research-contract violations in ordinary PyTorch
programs, not arbitrary malicious Python that tampers with instrumentation.
Models using nonstandard functional execution need separate supported
instrumentation rather than disabling the checks. The engineering smoke uses
16 real training images and 4 real validation images, one optimizer step, and an
independent process for checkpoint reload; it does not establish research quality
or consume sealed test data. See `scripts/model-runtime-smoke.py` (expects
read-only `/dataset`, writable `/workspace`, project at `/project`, and CUDA).

## Stage-2 tuning completion

Stage 2 uses host-side artifact checks for both main-stage and substage
completion, best-node selection, resume and final stage bookkeeping. Optional
VLM plot analysis and a second dataset are not prerequisites. A valid search
may finish without improving the baseline. Reaching the iteration limit without
valid evidence is an error, not success.

Generated tuning programs must save `working/tuning_evidence.json` with
`schema_version: 1`, `complete: true`, `seed`, `selection_metric`,
`selected_learning_rate`, and exactly two `candidates`. Each candidate contains
`learning_rate`, its best `validation_metric`, and a consecutive epoch `history`
containing `epoch`, `train_loss`, `validation_loss`, and `validation_metric`.
The selected score must agree with host-recomputed final validation predictions.
The two rates must include the baseline rate. Model Module definitions and
constructor expressions, plus manifest controls, must match the baseline.
Unsupported model definitions fail closed rather than assuming equivalence.
These checks catch accidental drift; generated training histories are not
independently measured by the host and are not proof against malicious code.

The evidence snapshot hashes tuning history and optional diagnostic progress
alongside the existing checkpoint and prediction artifacts. A new execution
clears stale worker tuning files. Legacy snapshots without tuning history are
preserved but cannot establish stage-2 completion; mutable progress is never
backfilled into historical evidence. An otherwise successful legacy candidate
without sufficient evidence stops with `TUNING_EVIDENCE_BLOCKED` before further
training on resume. Any replacement run requires a separately approved recovery
plan; the guard does not silently restart or manufacture a passing record.

`agent_progress.json` records stage entry and completed steps for the UI.
Execution progress is displayed separately from accepted scientific stages;
an interrupted task is not labelled as unconditionally ready to continue.
