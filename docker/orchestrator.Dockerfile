FROM docker:27.5.1-cli@sha256:851f91d241214e7c6db86513b270d58776379aacc5eb9c4a87e5b47115e3065c AS docker_cli

FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app
COPY docker/orchestrator-requirements.lock /tmp/orchestrator-requirements.lock
RUN python -m pip install --no-cache-dir --retries 5 --timeout 180 \
    -r /tmp/orchestrator-requirements.lock \
    && rm /tmp/orchestrator-requirements.lock

COPY pyproject.toml README.md /app/
COPY gate_a /app/gate_a
COPY configs /app/configs
COPY vendor/AI-Scientist-v2 /app/vendor/AI-Scientist-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "gate_a"]
