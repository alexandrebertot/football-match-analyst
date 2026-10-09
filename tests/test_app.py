import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from football_agent import app as app_module


@pytest.fixture
def startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_predictor: Any) -> None:
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "CACHE_PATH", tmp_path / "cache.sqlite")
    # Start-up must neither download the match history nor read the real MLflow registry.
    monkeypatch.setattr(app_module, "download_history", lambda *args: None)
    monkeypatch.setattr(app_module, "load_predictor", lambda *args: fake_predictor)


@pytest.fixture
def client(startup: None) -> Iterator[TestClient]:
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_root_serves_the_chat_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<form id="question-form">' in response.text


def test_ask_returns_the_agent_answer_with_the_loaded_predictor_and_today(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_predictor: Any
) -> None:
    received = {}

    def fake_answer(question: str, llm: Any, context: Any) -> str:
        received["question"] = question
        received["predictor"] = context.predictor
        received["today"] = context.today
        return "Monaco est premier."

    monkeypatch.setattr(app_module, "answer", fake_answer)

    response = client.post("/ask", json={"question": "Qui est premier en Ligue 1 ?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Monaco est premier."}
    assert received["question"] == "Qui est premier en Ligue 1 ?"
    assert received["predictor"] is fake_predictor
    assert received["today"] == date.today()


@pytest.mark.parametrize(
    "body", [{}, {"question": ""}, {"question": 42}], ids=["missing", "empty", "not-a-string"]
)
def test_ask_rejects_invalid_question(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/ask", json=body).status_code == 422


def test_ask_turns_agent_failure_into_bad_gateway(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_answer(*args: Any) -> str:
        raise RuntimeError("No answer after 5 rounds of tool calls.")

    monkeypatch.setattr(app_module, "answer", failing_answer)

    response = client.post("/ask", json={"question": "Classement ?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "No answer after 5 rounds of tool calls."}


def test_app_refuses_to_start_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)

    with pytest.raises(KeyError, match="FOOTBALL_DATA_API_KEY"), TestClient(app_module.app):
        pass


def test_app_starts_with_the_downloaded_history_when_the_refresh_fails(
    startup: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_predictor: Any,
) -> None:
    def unreachable_site(*args: Any) -> None:
        raise httpx.ConnectError("football-data.co.uk is unreachable")

    monkeypatch.setattr(app_module, "download_history", unreachable_site)

    with caplog.at_level(logging.WARNING), TestClient(app_module.app):
        assert app_module.app.state.predictor is fake_predictor

    assert "Could not refresh the match history" in caplog.text
    assert "football-data.co.uk is unreachable" in caplog.text
