from football_agent.llm import LLM_TIMEOUT_SECONDS, OLLAMA_BASE_URL, make_llm_client


def test_make_llm_client_points_to_ollama() -> None:
    llm = make_llm_client()

    assert llm.base_url == OLLAMA_BASE_URL + "/"


def test_make_llm_client_gives_up_on_a_stuck_ollama_without_retrying() -> None:
    llm = make_llm_client()

    assert (llm.timeout, llm.max_retries) == (LLM_TIMEOUT_SECONDS, 0)
