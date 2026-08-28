FROM docker:27.5.1-cli@sha256:851f91d241214e7c6db86513b270d58776379aacc5eb9c4a87e5b47115e3065c AS docker_cli

FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime@sha256:d0a2f5993da1a5646d77a95d4185d3d9fc79e95dcbc960daf245a18c7d6b6411

COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY docker/pathmnist-requirements.lock /tmp/pathmnist-requirements.lock
RUN python -m pip install --no-cache-dir --retries 5 --timeout 180 \
    -r /tmp/pathmnist-requirements.lock \
    --break-system-packages \
    && rm /tmp/pathmnist-requirements.lock

WORKDIR /workspace
