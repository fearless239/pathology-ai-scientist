FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

RUN useradd --create-home --uid 10001 pathscientist
WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY gate_a /app/gate_a
COPY pathmnist /app/pathmnist
COPY path_ai_scientist /app/path_ai_scientist
COPY app.py /app/app.py
RUN python -m pip install --no-cache-dir ".[ui]"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER pathscientist
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
