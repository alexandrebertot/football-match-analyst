# Football LLM agent

An LLM agent answers questions about the top 5 European leagues and the Champions League
by calling tools (football-data.org API, a LightGBM match predictor). A small open model is
later fine-tuned with QLoRA (distillation from a 7-8B teacher) to call tools reliably.

## Stack

- Python 3.12, `uv` for dependencies, `ruff` for lint and format, `pytest` for tests
- LLM: Qwen via Ollama locally, called through the OpenAI-compatible API (`openai` client, custom `base_url`)
- FastAPI, LightGBM, MLflow, Docker, GitHub Actions
- Fine-tuning: `transformers`, `peft`, `trl` on a local RTX 5060 (8 GB VRAM, CUDA 12.8+)

## Layout

```
src/football_agent/
  data_api.py      # football-data.org client + SQLite cache
  tools.py         # tool functions + their JSON schemas
  agent.py         # tool-calling loop
  llm.py           # LLM client config
  app.py           # FastAPI app
  predictor/       # data loading, features, training, inference
tests/
notebooks/         # exploration only, never imported by src/
```

## Commands

- Install: `uv sync`
- Lint and format: `uv run ruff check --fix . && uv run ruff format .`
- Tests: `uv run pytest`
- API: `uv run uvicorn football_agent.app:app --reload`

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

- Never commit on `main`. One branch per task: `feat/…`, `fix/…`, `refactor/…`, `test/…`, `docs/…`.
- A branch only contains changes for its task. If you notice something unrelated, tell me instead of fixing it.
- Small commits with Conventional Commits messages: `feat: add standings tool`, `fix: …`, `test: …`.
- Never push, force-push, merge or rewrite history without my explicit request.
- pre-commit runs ruff (check + format) on commit and pytest on push. Never bypass it with `--no-verify`.
