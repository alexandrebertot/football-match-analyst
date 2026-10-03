from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "football-qwen"


def make_llm_client() -> OpenAI:
    # Ollama ignores the API key, but the openai client refuses to start without one.
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
