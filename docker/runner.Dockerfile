FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-lmodern=2.005-1 \
        texlive-latex-base=2022.20230122-3 \
        texlive-latex-recommended=2022.20230122-3 \
        texlive-xetex=2022.20230122-3 \
        fonts-noto-cjk=1:20220127+repack1-1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/runner-requirements.lock /tmp/runner-requirements.lock
RUN python -m pip install --no-cache-dir --retries 5 --timeout 180 \
    -r /tmp/runner-requirements.lock \
    && rm /tmp/runner-requirements.lock

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace
