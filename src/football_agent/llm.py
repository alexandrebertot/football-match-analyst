from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "football-qwen"
# One reply can take a while with thinking on; past this, Ollama is stuck rather than slow.
LLM_TIMEOUT_SECONDS = 120


def make_llm_client() -> OpenAI:
    # Ollama ignores the API key, but the openai client refuses to start without one. Retrying
    # a stuck Ollama would only multiply the wait.
    return OpenAI(
        base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=LLM_TIMEOUT_SECONDS, max_retries=0
    )
