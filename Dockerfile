FROM python:3.13-slim

# LightGBM loads the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

# Same uid as the host user, so files written to mounted folders stay editable outside Docker.
RUN useradd --uid 1000 app
USER app

ENV PATH="/app/.venv/bin:$PATH"
# Start-up refreshes the match history and loads the model, hence the grace period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=8)"]
CMD ["uvicorn", "football_agent.app:app", "--host", "0.0.0.0", "--port", "8000"]
