#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="path-scientist-pathmnist-runner:0.1"
docker_cmd=(docker)

credential_file="${XDG_CONFIG_HOME:-$HOME/.config}/path-scientist/env"
if [[ -f "$credential_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$credential_file"
  set +a
fi

require_sudo_ticket() {
  if docker info >/dev/null 2>&1; then
    docker_cmd=(docker)
  elif sudo -n docker info >/dev/null 2>&1; then
    docker_cmd=(sudo -n docker)
  else
    echo "Docker is unavailable for this user and no non-interactive sudo fallback exists." >&2
    exit 3
  fi
}

build() {
  require_sudo_ticket
  # Docker Desktop may fail while reading xattrs/ACLs on unrelated cache files
  # under /mnt/c even when .dockerignore excludes them. This image only COPYs
  # its locked requirements, so build from a minimal Linux staging directory.
  local build_context
  build_context="$(mktemp -d "${TMPDIR:-/tmp}/path-scientist-build.XXXXXX")"
  if [[ -z "$build_context" || "$build_context" != "${TMPDIR:-/tmp}/path-scientist-build."* ]]; then
    echo "Refusing unsafe temporary build context: $build_context" >&2
    exit 5
  fi
  trap "rm -rf -- '$build_context'" EXIT
  mkdir -p "$build_context/docker"
  cp "$project_root/docker/pathmnist.Dockerfile" "$build_context/docker/pathmnist.Dockerfile"
  cp "$project_root/docker/pathmnist-requirements.lock" "$build_context/docker/pathmnist-requirements.lock"
  "${docker_cmd[@]}" build --pull=false \
    -f "$build_context/docker/pathmnist.Dockerfile" \
    -t "$image" \
    "$build_context"
  rm -rf -- "$build_context"
  trap - EXIT
  "${docker_cmd[@]}" run --rm \
    --mount "type=bind,src=$project_root,dst=/workspace,readonly" \
    --workdir /workspace \
    "$image" python -m pathmnist framework-smoke --project-root /workspace
}

run_container() {
  require_sudo_ticket
  local extra_mounts=()
  if [[ -d "$project_root/runs" ]]; then
    extra_mounts+=(--mount "type=bind,src=$project_root/runs,dst=/workspace/runs")
  else
    extra_mounts+=(--mount "type=volume,src=pathmnist_runs,dst=/workspace/runs")
  fi
  "${docker_cmd[@]}" run --rm --gpus all \
    --shm-size 2g \
    --mount "type=bind,src=$project_root,dst=/workspace,readonly" \
    "${extra_mounts[@]}" \
    --workdir /workspace \
    "$image" "$@"
}

case "${1:-help}" in
  build)
    build
    ;;
  validate)
    run_container python -m pathmnist --config configs/pathmnist_m4.yaml validate --project-root /workspace
    ;;
  gpu-smoke)
    run_container python -m pathmnist --config configs/pathmnist_m4.yaml gpu-smoke
    ;;
  framework-smoke)
    run_container python -m pathmnist framework-smoke --project-root /workspace
    ;;
  train)
    shift
    run_container python -m pathmnist --config configs/pathmnist_m4.yaml train --project-root /workspace --output-root /workspace/runs/pathmnist-m4 "$@"
    ;;
  prepare-final)
    shift
    run_container python -m pathmnist --config configs/pathmnist_m4.yaml prepare-final --project-root /workspace --output-root /workspace/runs/pathmnist-m4 --candidate configs/pathmnist_final_candidate.json "$@"
    ;;
  evaluate-test)
    shift
    run_container python -m pathmnist --config configs/pathmnist_m4.yaml evaluate-test --project-root /workspace --output-root /workspace/runs/pathmnist-m4 --candidate configs/pathmnist_final_candidate.json "$@"
    ;;
  paper-smoke)
    shift
    require_sudo_ticket
    if [[ -z "${PARATERA_API_KEY:-}" ]]; then
      echo "PARATERA_API_KEY must be exported in the calling WSL shell." >&2
      exit 4
    fi
    mkdir -p "$project_root/state/workflow"
    "${docker_cmd[@]}" run --rm \
      --env PARATERA_API_KEY="$PARATERA_API_KEY" \
      --mount "type=bind,src=$project_root/state,dst=/workspace/state" \
      --mount "type=bind,src=$project_root/runs,dst=/workspace/runs,readonly" \
      --mount "type=bind,src=$project_root/pathmnist_64.npz,dst=/workspace/pathmnist_64.npz,readonly" \
      --mount "type=bind,src=$project_root/configs,dst=/workspace/configs,readonly" \
      --mount "type=bind,src=$project_root/docs,dst=/workspace/docs,readonly" \
      --mount "type=bind,src=$project_root/pathmnist,dst=/workspace/pathmnist,readonly" \
      --mount "type=bind,src=$project_root/gate_a,dst=/workspace/gate_a,readonly" \
      --workdir /workspace \
      "$image" python -m pathmnist --config /workspace/configs/pathmnist_m4.yaml paper-smoke \
      --project-root /workspace \
      --state-root /workspace/state/workflow \
      --confirm-paid "$@"
    ;;
  paper-export)
    shift
    require_sudo_ticket
    "${docker_cmd[@]}" run --rm \
      --env PYTHONPATH=/workspace \
      --mount "type=bind,src=$project_root,dst=/workspace,readonly" \
      --mount "type=bind,src=$project_root/docs,dst=/workspace/docs" \
      --workdir /workspace/docs \
      path-scientist-gate-a-runner:0.2 \
      sh -lc "python -m pathmnist.paper_export --input M5_FORMAL_PAPER.md --output latex/M5_FORMAL_PAPER.tex --language en \
        && python -m pathmnist.paper_export --input M5_FORMAL_PAPER_ZH.md --output latex/M5_FORMAL_PAPER_ZH.tex --language zh \
        && cd latex \
        && pdflatex -interaction=nonstopmode -halt-on-error M5_FORMAL_PAPER.tex \
        && pdflatex -interaction=nonstopmode -halt-on-error M5_FORMAL_PAPER.tex \
        && xelatex -interaction=nonstopmode -halt-on-error M5_FORMAL_PAPER_ZH.tex \
        && xelatex -interaction=nonstopmode -halt-on-error M5_FORMAL_PAPER_ZH.tex"
    ;;
  archive)
    shift
    require_sudo_ticket
    mkdir -p "$project_root/runs/archives"
    "${docker_cmd[@]}" run --rm \
      --env PYTHONPATH=/workspace \
      --mount "type=bind,src=$project_root,dst=/workspace,readonly" \
      --mount "type=bind,src=$project_root/runs,dst=/workspace/runs" \
      --workdir /workspace \
      path-scientist-gate-a-runner:0.2 \
      python -m pathmnist.archive --project-root /workspace --output /workspace/runs/archives/pathmnist-m4-m5-archive.zip "$@"
    ;;
  supervisor)
    shift
    require_sudo_ticket
    extra_mounts=()
    extra_env=()
    if [[ -n "${PARATERA_API_KEY:-}" ]]; then
      extra_env+=(--env "PARATERA_API_KEY=$PARATERA_API_KEY")
    fi
    if [[ -n "${S2_API_KEY:-}" ]]; then
      extra_env+=(--env "S2_API_KEY=$S2_API_KEY")
    fi
    mkdir -p "$project_root/state/workflow"
    extra_mounts+=(--mount "type=bind,src=$project_root/state,dst=/workspace/state")
    if [[ -d "$project_root/runs" ]]; then
      extra_mounts+=(--mount "type=bind,src=$project_root/runs,dst=/workspace/runs")
    else
      extra_mounts+=(--mount "type=volume,src=pathmnist_runs,dst=/workspace/runs")
    fi
    "${docker_cmd[@]}" run --rm --gpus all \
      --shm-size 2g \
      "${extra_mounts[@]}" \
      "${extra_env[@]}" \
      --mount "type=bind,src=$project_root,dst=/workspace,readonly" \
      --workdir /workspace \
      "$image" python -m pathmnist.worker --supervisor --project-root /workspace "$@"
    ;;
  web)
    shift
    require_sudo_ticket
    if ! "${docker_cmd[@]}" image inspect "$image" >/dev/null 2>&1; then
      echo "The web image is missing; building it once before startup..."
      build
    fi
    extra_mounts=()
    extra_env=()
    if [[ -n "${PARATERA_API_KEY:-}" ]]; then
      extra_env+=(--env "PARATERA_API_KEY=$PARATERA_API_KEY")
    fi
    if [[ -n "${S2_API_KEY:-}" ]]; then
      extra_env+=(--env "S2_API_KEY=$S2_API_KEY")
    fi
    mkdir -p "$project_root/state/workflow"
    extra_mounts+=(--mount "type=bind,src=$project_root/state,dst=$project_root/state")
    if [[ -d "$project_root/runs" ]]; then
      extra_mounts+=(--mount "type=bind,src=$project_root/runs,dst=$project_root/runs")
    else
      extra_mounts+=(--mount "type=volume,src=pathmnist_runs,dst=$project_root/runs")
    fi
    "${docker_cmd[@]}" run --rm --gpus all \
      --shm-size 2g \
      "${extra_env[@]}" \
      --publish 127.0.0.1:8501:8501 \
      --mount "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock" \
      --mount "type=bind,src=$project_root,dst=$project_root,readonly" \
      "${extra_mounts[@]}" \
      --workdir "$project_root" \
      "$image" python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats false "$@"
    ;;
  workflow-run)
    shift
    require_sudo_ticket
    extra_env=()
    if [[ -n "${PARATERA_API_KEY:-}" ]]; then
      extra_env+=(--env "PARATERA_API_KEY=$PARATERA_API_KEY")
    fi
    if [[ -n "${S2_API_KEY:-}" ]]; then
      extra_env+=(--env "S2_API_KEY=$S2_API_KEY")
    fi
    mkdir -p "$project_root/state/workflow" "$project_root/runs"
    "${docker_cmd[@]}" run --rm --gpus all \
      --shm-size 2g \
      "${extra_env[@]}" \
      --mount "type=bind,src=$project_root,dst=/workspace,readonly" \
      --mount "type=bind,src=$project_root/state,dst=/workspace/state" \
      --mount "type=bind,src=$project_root/runs,dst=/workspace/runs" \
      --workdir /workspace \
      "$image" python -m pathmnist --config /workspace/configs/pathmnist_m4.yaml workflow-run \
      --project-root /workspace --state-root /workspace/state/workflow "$@"
    ;;
  v2)
    shift
    require_sudo_ticket
    mkdir -p "$project_root/state/workflow"
    extra_env=()
    if [[ -n "${PARATERA_API_KEY:-}" ]]; then extra_env+=(--env PARATERA_API_KEY); fi
    "${docker_cmd[@]}" run --rm \
      "${extra_env[@]}" \
      --entrypoint python \
      --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_orchestrator \
      --project-root "$project_root" \
      --state-root "$project_root/state/workflow" "$@"
    ;;
  autonomous-init)
    shift
    require_sudo_ticket
    mkdir -p "$project_root/state/workflow"
    "${docker_cmd[@]}" run --rm \
      --shm-size 2g \
      --mount "type=bind,src=$project_root,dst=/workspace,readonly" \
      --mount "type=bind,src=$project_root/state,dst=/workspace/state" \
      --workdir /workspace \
      "$image" python -m pathmnist autonomous-init \
      --state-root /workspace/state/workflow \
      --dataset-path /workspace/pathmnist_64.npz "$@"
    ;;
  autonomous-preflight)
    shift
    require_sudo_ticket
    mkdir -p "$project_root/state/workflow"
    "${docker_cmd[@]}" run --rm \
      --entrypoint python \
      --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_preflight \
      --project-root "$project_root" \
      --state-root "$project_root/state/workflow" "$@"
    ;;
  autonomous-research)
    shift
    require_sudo_ticket
    mkdir -p "$project_root/state/workflow"
    extra_env=()
    if [[ -n "${S2_API_KEY:-}" ]]; then extra_env+=(--env S2_API_KEY); fi
    "${docker_cmd[@]}" run --rm \
      "${extra_env[@]}" \
      --entrypoint python \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_research \
      --state-root "$project_root/state/workflow" "$@"
    ;;
  autonomous-paid)
    shift
    require_sudo_ticket
    if [[ -z "${PARATERA_API_KEY:-}" ]]; then
      echo "PARATERA_API_KEY is absent from this WSL process." >&2
      exit 4
    fi
    "${docker_cmd[@]}" run --rm \
      --env PARATERA_API_KEY \
      --entrypoint python \
      --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_paid \
      --project-root "$project_root" \
      --state-root "$project_root/state/workflow" \
      --confirm-paid "$@"
    ;;
  autonomous-repair)
    shift
    require_sudo_ticket
    if [[ -z "${PARATERA_API_KEY:-}" ]]; then
      echo "PARATERA_API_KEY is absent from this WSL process." >&2
      exit 4
    fi
    "${docker_cmd[@]}" run --rm \
      --env PARATERA_API_KEY \
      --entrypoint python \
      --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_repair \
      --project-root "$project_root" \
      --state-root "$project_root/state/workflow" \
      --confirm-paid "$@"
    ;;
  autonomous-export)
    shift
    require_sudo_ticket
    "${docker_cmd[@]}" run --rm \
      --entrypoint python \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_export \
      --project-root "$project_root" \
      --state-root "$project_root/state/workflow" "$@"
    ;;
  autonomous-freeze)
    shift
    require_sudo_ticket
    "${docker_cmd[@]}" run --rm \
      --entrypoint python \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_freeze \
      --project-root "$project_root" \
      --state-root "$project_root/state/workflow" "$@"
    ;;
  autonomous-test)
    shift
    require_sudo_ticket
    if [[ "${1:-}" == "approve" ]]; then
      mode=approve
    elif [[ "${1:-}" == "evaluate" ]]; then
      mode=evaluate
      require_sudo_ticket
    else
      echo "Usage: autonomous-test approve|evaluate --task-id TASK" >&2
      exit 2
    fi
    shift
    extra_args=()
    if [[ "$mode" == "evaluate" ]]; then extra_args+=(--gpus all --shm-size 2g); fi
    "${docker_cmd[@]}" run --rm "${extra_args[@]}" \
      --entrypoint python \
      --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_test "$mode" \
      --project-root "$project_root" \
      --state-root "$project_root/state/workflow" "$@"
    ;;
  autonomous-postprocess)
    shift
    require_sudo_ticket
    if [[ -z "${PARATERA_API_KEY:-}" ]]; then
      echo "PARATERA_API_KEY is absent from this WSL process." >&2
      exit 4
    fi
    "${docker_cmd[@]}" run --rm \
      --env PARATERA_API_KEY \
      --entrypoint python \
      --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_postprocess \
      --project-root "$project_root" \
      --state-root "$project_root/state/workflow" "$@"
    ;;
  autonomous-pdf)
    shift
    require_sudo_ticket
    "${docker_cmd[@]}" run --rm \
      --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
      --mount "type=bind,src=$project_root,dst=$project_root" \
      --workdir "$project_root" \
      --env PYTHONPATH="$project_root" \
      --entrypoint python \
      path-scientist-gate-a-orchestrator:0.1 \
      -m pathmnist.autonomous_pdf \
      --project-root "$project_root" \
      --state-root "$project_root/state/workflow" "$@"
    ;;
  help|*)
    cat <<'EOF'
Usage: ./scripts/pathmnist.sh COMMAND

  build       Build the pinned PyTorch CUDA image
  validate    Verify dataset SHA-256, shapes, dtypes, and labels (read-only)
  gpu-smoke   Verify CUDA, tensor operation, and a small backward pass
  framework-smoke Verify AI-Scientist-v2 imports in the pinned runtime
  train       Run train/val experiments; add --smoke --limit-epochs N for engineering check
  prepare-final Retrain the frozen candidate and save three checkpoints
  evaluate-test Perform the one-time frozen test evaluation (requires explicit approval)
  web         Launch the Streamlit research-task UI and results dashboard
  v2          Deprecated alias for the research control plane
  workflow-run Legacy fixture/compatibility workflow; never grants formal acceptance
  autonomous-init Create a research task and train/validation-only dataset view
  autonomous-research Build research understanding, verified literature, idea, and experiment spec
  autonomous-preflight Verify AgentManager, checkpoint, and nested sandbox without LLM calls
  autonomous-paid Run the real four-stage AgentManager with GLM and a $2 hard limit
  autonomous-repair Continue a completed run with audited hard-routing repair stages
  autonomous-export Export successful journal nodes as standard experiment evidence
  autonomous-freeze Freeze the best audited validation-only dynamic candidate
  autonomous-test Approve or execute the one-time sealed test evaluation
  autonomous-postprocess Generate analysis, paper, review, revision, translation, and archive
  autonomous-pdf Build English and Chinese PDFs from the final revised papers
  supervisor  Restart PathMNIST workflow tasks with absent or stale worker locks
  paper-smoke Run a paid formal-paper smoke using frozen M4 artifacts
  paper-export Regenerate LaTeX from final paper Markdown and compile EN/ZH PDFs
  archive     Build the lightweight M4/M5 evidence archive under runs/archives
EOF
    ;;
esac
