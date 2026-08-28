#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_id="${TASK_ID:-p0-beta-e2e-$(date -u +%Y%m%dT%H%M%SZ)}"
direction="${RESEARCH_DIRECTION:-Study uncertainty-driven hard dynamic resolution on PathMNIST: route simple samples through low resolution and conditionally upgrade difficult samples, while preserving classification performance and measuring actual compute and latency trade-offs.}"

if [[ -z "${PARATERA_API_KEY:-}" ]]; then
  echo "PARATERA_API_KEY is not present. Run this from the interactive WSL shell where it is exported." >&2
  exit 4
fi
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run with: sudo -E bash ./scripts/p0-e2e.sh" >&2
  exit 3
fi

cd "$project_root"
mkdir -p runs/p0-e2e-logs state/workflow
log="runs/p0-e2e-logs/${task_id}.log"
exec > >(tee -a "$log") 2>&1

step() { printf '\n===== %s =====\n' "$1"; }
fail() { code=$?; echo "FAILED exit=$code task=$task_id log=$log"; exit "$code"; }
trap fail ERR

echo "TASK_ID=$task_id"
echo "LOG=$log"
echo "API key is present (value is never logged)."

if [[ "${SKIP_GATE_BUILD:-0}" == "1" ]]; then
  step "Use cached pinned Gate A images"
  docker image inspect \
    path-scientist-gate-a-runner:0.2 \
    path-scientist-gate-a-orchestrator:0.1 >/dev/null
else
  step "Build pinned Gate A images"
  bash ./scripts/gate-a.sh build
fi

step "Run repeatable offline Gate A acceptance"
bash ./scripts/gate-a.sh offline

step "Validate live gateway catalog and budget fit (no inference)"
bash ./scripts/gate-a.sh llm-preflight

if [[ "${SKIP_PATHMNIST_BUILD:-0}" == "1" ]]; then
  step "Use cached PathMNIST image"
  docker image inspect path-scientist-pathmnist-runner:0.1 >/dev/null
else
  step "Build and smoke-test PathMNIST image"
  bash ./scripts/pathmnist.sh build
fi

step "Create isolated research task"
bash ./scripts/pathmnist.sh autonomous-init \
  --task-id "$task_id" \
  --direction "$direction"

step "Collect verified research literature and freeze experiment specification"
bash ./scripts/pathmnist.sh autonomous-research --task-id "$task_id"

step "Run nested sandbox preflight"
bash ./scripts/pathmnist.sh autonomous-preflight --task-id "$task_id"

step "Run paid AI-Scientist-v2 experiment search"
bash ./scripts/pathmnist.sh autonomous-paid --task-id "$task_id" --confirm-paid

step "Run paid audited hard-routing repair stage"
bash ./scripts/pathmnist.sh autonomous-repair --task-id "$task_id" --confirm-paid

step "Export experiment evidence"
bash ./scripts/pathmnist.sh autonomous-export --task-id "$task_id"

step "Select and freeze validation-only candidate"
bash ./scripts/pathmnist.sh autonomous-freeze --task-id "$task_id"

step "Durably approve the one-time sealed test"
bash ./scripts/pathmnist.sh autonomous-test approve --task-id "$task_id"

step "Execute the one-time sealed test"
bash ./scripts/pathmnist.sh autonomous-test evaluate --task-id "$task_id"

step "Generate evidence-bound paper, independent review, revision, and translation"
bash ./scripts/pathmnist.sh autonomous-postprocess --task-id "$task_id"

step "Compile, quality-check, accept, and archive PDFs"
bash ./scripts/pathmnist.sh autonomous-pdf --task-id "$task_id"

step "Final independent acceptance report"
docker run --rm \
  --entrypoint python \
  --mount "type=bind,src=$project_root,dst=$project_root" \
  --workdir "$project_root" \
  --env PYTHONPATH="$project_root" \
  path-scientist-gate-a-orchestrator:0.1 \
  -m pathmnist.autonomous_acceptance \
  --task-root "$project_root/state/workflow/$task_id" \
  --target-stage archived \
  --require-pdf

trap - ERR
echo "PASSED task=$task_id log=$log"
