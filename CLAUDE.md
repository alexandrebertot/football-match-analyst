# Football LLM agent

An LLM agent answers questions about the top 5 European leagues and the Champions League
by calling tools (football-data.org API, a LightGBM match predictor). A small open model is
later fine-tuned with QLoRA (distillation from a larger teacher model) to call tools reliably.

## Stack

- Python 3.13, `uv` for dependencies, `ruff` for lint and format, `pytest` for tests
- LLM: Qwen via Ollama locally, called through the OpenAI-compatible API (`openai` client, custom `base_url`)
- FastAPI, LightGBM, MLflow, Docker, GitHub Actions
- Fine-tuning: `transformers`, `peft`, `trl` on a local RTX 4060 Laptop GPU (8 GB VRAM)

## Layout

```
src/football_agent/
  data_api.py      # football-data.org client + SQLite cache
  tools.py         # tool functions + their JSON schemas
  agent.py         # tool-calling loop
  llm.py           # LLM client config
  app.py           # FastAPI app: chat page at /, agent at POST /ask
  static/          # chat page (HTML, CSS, JavaScript in one file)
  predictor/       # data download, features, training, registry, inference
    team_names.yaml  # football-data.org team ids -> CSV team names, updated each season
configs/
  datasets/        # dataset configs: seasons, split, form window
  training/        # training configs: features, LightGBM hyperparameters, early stopping
ollama/Modelfile   # football-qwen: base model and context size
Dockerfile, compose.yaml  # API image, run with docker compose
tests/             # mirrors src/ (tests/predictor/ for the predictor)
notebooks/         # exploration only, never imported by src/
```

## Commands

- Install: `uv sync && uv run pre-commit install`
- LLM (once, and after editing the Modelfile): `ollama create football-qwen -f ollama/Modelfile`
- Lint and format: `uv run ruff check --fix . && uv run ruff format .`
- Tests: `uv run pytest`
- API (needs Ollama running and a `champion` model in `mlflow.db`; refreshes the current season
  at start-up): `uv run --env-file .env uvicorn football_agent.app:app --reload`
- Predictor data: `uv run python -m football_agent.predictor.data` (downloads each finished season
  once and the season under way every time)
- Predictor dataset: `uv run python -m football_agent.predictor.prepare --config configs/datasets/<name>.yaml`
  (rerun it after any change to the feature code: training reads the saved Parquet files)
- Predictor training: `uv run python -m football_agent.predictor.train --config configs/training/<name>.yaml`
- Predictor champion: `uv run python -m football_agent.predictor.registry --run <run name>`
  (registers the latest run with that name as a new version of `match-outcome` and moves the
  `champion` alias to it; the agent always loads `models:/match-outcome@champion`)
- MLflow UI: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Code rules

- Simple and readable first. No abstraction, class, config option or parameter that the current task does not need.
- No dead code, no commented-out code, no placeholder functions, no `print` debugging left behind.
- Comments only to explain *why* something non-obvious is done. Never paraphrase the code. No banner or section comments. Short docstrings only on public functions whose behavior isn't obvious from the name and types.
- Type hints on all function signatures. Descriptive names in English.
- Small functions that do one thing. Pure functions where possible (easy to test).
- Handle errors where they can actually be handled; never swallow exceptions with a bare `except`.
- No new dependency without asking me first.
- Secrets only in `.env` (never committed). Never print or log API keys.
- Data leakage matters: predictor features must only use information available before kick-off.

## Tests

- Every new function in `src/` gets a pytest test in the same change, unless it's pure I/O glue.
- Mock external calls (football-data.org, LLM) in tests; tests must run offline.
- Run `ruff` and `pytest` before saying a task is done, and show me the result.

## Git

- Never commit on `main`. One branch per task, prefixed with its Conventional Commits type: `feat/…`, `fix/…`, `refactor/…`, `test/…`, `docs/…`, `ci/…`, `build/…`, `chore/…`.
- A branch only contains changes for its task. If you notice something unrelated, tell me instead of fixing it.
- Small commits with Conventional Commits messages: `feat: add standings tool`, `fix: …`, `test: …`.
- Never push, force-push, merge or rewrite history without my explicit request.
- pre-commit runs ruff (check + format) on commit and pytest on push. Never bypass it with `--no-verify`.
