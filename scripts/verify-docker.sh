#!/usr/bin/env bash
set -euo pipefail

image="${1:?usage: verify-docker.sh IMAGE}"

uid="$(docker run --rm --entrypoint id "$image" -u)"
test "$uid" != "0"
echo "NON_ROOT_UID=$uid"

docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:uid=10001,gid=10001 \
  --entrypoint python \
  "$image" -m pathmnist.demo --output /tmp/demo

if docker run --rm --read-only --entrypoint sh "$image" -c "touch /app/should-fail"; then
  echo "READ_ONLY_CHECK=failed"
  exit 1
fi
echo "READ_ONLY_CHECK=passed"

if docker run --rm --entrypoint env "$image" \
  | grep -E "^(PARATERA_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)="; then
  echo "SECRET_CHECK=failed"
  exit 1
fi
echo "SECRET_CHECK=passed"

container_name="path-ai-scientist-verify-$$"
docker run -d \
  --name "$container_name" \
  --read-only \
  --tmpfs /tmp:uid=10001,gid=10001 \
  --tmpfs /app/.demo:uid=10001,gid=10001 \
  -p 8501:8501 \
  "$image" >/dev/null
cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

health_file="$(mktemp)"
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8501/_stcore/health >"$health_file"; then
    break
  fi
  sleep 1
done
grep -q "ok" "$health_file"
echo "STREAMLIT_HEALTH=$(cat "$health_file")"
docker ps --filter "name=$container_name"
docker logs --tail 30 "$container_name"
