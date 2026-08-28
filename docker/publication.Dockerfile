# Separate image: legacy tasks keep the original compiler environment.
FROM path-scientist-gate-a-runner:0.2
RUN apt-get update \
    && apt-get install -y --no-install-recommends texlive-fonts-recommended=2022.20230122-3 \
    && rm -rf /var/lib/apt/lists/*
