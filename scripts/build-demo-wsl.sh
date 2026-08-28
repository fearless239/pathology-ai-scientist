#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$(pwd)}"
image="${2:-path-ai-scientist-demo:local}"
project_root="$(cd "$project_root" && pwd)"
build_context="$(mktemp -d /tmp/path-ai-scientist-build.XXXXXX)"

cleanup() {
  if [[ "$build_context" == /tmp/path-ai-scientist-build.* ]]; then
    rm -rf -- "$build_context"
  fi
}
trap cleanup EXIT

tar -C "$project_root" \
  --exclude=.git \
  --exclude=.venv \
  --exclude=.pytest_cache \
  --exclude=.ruff_cache \
  --exclude=state \
  --exclude=runs \
  --exclude=.demo \
  --exclude=.test-tmp \
  --exclude=tmp \
  -cf "$build_context/context.tar" .
tar -C "$build_context" -xf "$build_context/context.tar"
rm "$build_context/context.tar"

docker build --pull=false \
  -f "$build_context/docker/demo.Dockerfile" \
  -t "$image" \
  "$build_context"

echo "BUILT_IMAGE=$image"
