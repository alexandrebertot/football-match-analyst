# football-match-analyst

[![CI](https://github.com/alexandrebertot/football-match-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandrebertot/football-match-analyst/actions/workflows/ci.yml)

## Run with Docker

Prerequisites:

- Ollama running on the host with the `football-qwen` model: `ollama create football-qwen -f ollama/Modelfile`
- A `.env` file containing `FOOTBALL_DATA_API_KEY=<your football-data.org key>`
- A trained model promoted to `champion` in `mlflow.db`:

  ```bash
  uv run python -m football_agent.predictor.data
  uv run python -m football_agent.predictor.prepare --config configs/datasets/last5.yaml
  uv run python -m football_agent.predictor.train --config configs/training/plus_shots_on_target_against.yaml
  uv run python -m football_agent.predictor.registry --run plus-shots-on-target-against
  ```

- A `.cache` folder (`mkdir -p .cache`), so that Docker does not create it as root

Then, from the repository root:

```bash
docker compose up --build
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is top of the Premier League?"}'
```

The container uses the host network to reach Ollama, which listens on the host's `localhost:11434`.
