#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner_image="path-scientist-gate-a-runner:0.2"
orchestrator_image="path-scientist-gate-a-orchestrator:0.1"

require_sudo_ticket() {
  if ! sudo -n true 2>/dev/null; then
    echo "Docker requires an interactive sudo ticket. Run 'sudo -v' in this terminal, then retry." >&2
    exit 3
  fi
}

build_images() {
  require_sudo_ticket
  # BuildKit inspects xattrs before applying .dockerignore. Windows-created
  # caches can therefore break the context walk even when ignored. Assemble a
  # minimal Linux-side context containing only files referenced by Dockerfiles.
  local build_context
  build_context="$(mktemp -d "${TMPDIR:-/tmp}/path-scientist-gate-build.XXXXXX")"
  if [[ -z "$build_context" || "$build_context" != "${TMPDIR:-/tmp}/path-scientist-gate-build."* ]]; then
    echo "Refusing unsafe temporary build context: $build_context" >&2
    exit 5
  fi
  trap 'rm -rf -- "$build_context"' RETURN
  mkdir -p "$build_context/docker" "$build_context/gate_a" "$build_context/configs" \
    "$build_context/vendor/AI-Scientist-v2"
  cp "$project_root/docker/runner.Dockerfile" "$build_context/docker/runner.Dockerfile"
  cp "$project_root/docker/orchestrator.Dockerfile" "$build_context/docker/orchestrator.Dockerfile"
  cp "$project_root/docker/runner-requirements.lock" "$build_context/docker/runner-requirements.lock"
  cp "$project_root/docker/orchestrator-requirements.lock" "$build_context/docker/orchestrator-requirements.lock"
  cp "$project_root/pyproject.toml" "$project_root/README.md" "$build_context/"
  cp -a "$project_root/gate_a/." "$build_context/gate_a/"
  cp -a "$project_root/configs/." "$build_context/configs/"
  cp -a "$project_root/vendor/AI-Scientist-v2/." "$build_context/vendor/AI-Scientist-v2/"
  find "$build_context" -type d -name '__pycache__' -prune -exec rm -rf -- {} +
  find "$build_context" -type f \( -name '*.pyc' -o -name '*.pdf' \) -delete
  sudo -n docker build --pull=false -f "$build_context/docker/runner.Dockerfile" -t "$runner_image" "$build_context"
  sudo -n docker build --pull=false -f "$build_context/docker/orchestrator.Dockerfile" -t "$orchestrator_image" "$build_context"
  rm -rf -- "$build_context"
  trap - RETURN
}

run_orchestrator() {
  require_sudo_ticket
  local include_key="$1"
  local config_path="$2"
  shift 2
  local env_args=()
  local sudo_args=(-n)
  if [[ "$include_key" == "yes" ]]; then
    local present=()
    local var
    for var in OPENROUTER_API_KEY ZHIPU_API_KEY PARATERA_API_KEY; do
      if [[ -n "${!var:-}" ]]; then
        present+=("$var")
      fi
    done
    if [[ ${#present[@]} -eq 0 ]]; then
      echo "OPENROUTER_API_KEY or ZHIPU_API_KEY must be present in this WSL process." >&2
      exit 4
    fi
    env_args=("${present[@]/#/--env=}")
    sudo_args+=("--preserve-env=$(IFS=,; echo "${present[*]}")")
  fi
  sudo "${sudo_args[@]}" docker run --rm \
    --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
    --mount "type=bind,src=$project_root,dst=$project_root" \
    --workdir "$project_root" \
    "${env_args[@]}" \
    "$orchestrator_image" \
    --config "$config_path" "$@"
}

openrouter_config="$project_root/configs/gate_a.yaml"
llm_config="$project_root/configs/gate_a_llm.yaml"

case "${1:-help}" in
  build)
    build_images
    ;;
  offline)
    run_orchestrator no "$openrouter_config" run --provider fixture --repeat 2 --output-root "$project_root/runs/gate-a-offline"
    ;;
  preflight)
    run_orchestrator yes "$openrouter_config" preflight --show-models
    ;;
  paid)
    run_orchestrator yes "$openrouter_config" run --provider openrouter --confirm-paid-smoke --output-root "$project_root/runs/gate-a-paid"
    ;;
  llm-preflight)
    if [[ -z "${PARATERA_API_KEY:-}" ]]; then
      echo "PARATERA_API_KEY is absent from this WSL process." >&2
      exit 4
    fi
    run_orchestrator yes "$llm_config" preflight --show-models
    ;;
  llm-paid)
    if [[ -z "${PARATERA_API_KEY:-}" ]]; then
      echo "PARATERA_API_KEY is absent from this WSL process." >&2
      exit 4
    fi
    run_orchestrator yes "$llm_config" run --provider openai_compatible --confirm-paid-smoke --output-root "$project_root/runs/gate-a-paid-llm"
    ;;
  help|*)
    cat <<'EOF'
Usage: ./scripts/gate-a.sh COMMAND

  build           Build the pinned runner and orchestrator images (no API charge)
  offline         Run the fixture chain twice and compare structural outputs (no API charge)
  preflight       Check current OpenRouter models and prices without inference
  paid            Execute one real OpenRouter smoke chain with a hard USD 2.00 ledger
  llm-preflight   Check the OpenAI-compatible gateway config, key, and budget fit
  llm-paid        Execute one real gateway smoke chain with a hard USD 2.00 ledger

Run `sudo -v` first. The `paid` command is intentionally separate and must not be run until
the offline report has passed and the user has explicitly approved paid inference.
Verify the base URL and model IDs in configs/gate_a_llm.yaml before paid use.
EOF
    ;;
esac
