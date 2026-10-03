from football_agent.llm import OLLAMA_BASE_URL, make_llm_client


def test_make_llm_client_points_to_ollama() -> None:
    llm = make_llm_client()

    assert llm.base_url == OLLAMA_BASE_URL + "/"
