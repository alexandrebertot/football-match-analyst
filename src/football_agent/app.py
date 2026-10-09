import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import httpx
import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from football_agent.agent import answer
from football_agent.data_api import FootballDataClient, make_http_client, open_cache
from football_agent.llm import make_llm_client
from football_agent.predictor.data import RAW_DATA_DIR, download_history
from football_agent.predictor.predict import load_predictor
from football_agent.tools import ToolContext

CACHE_PATH = Path(".cache/football_data.sqlite")
CHAT_PAGE_PATH = Path(__file__).with_name("static") / "index.html"
TRACES_EXPERIMENT = "agent-traces"

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str


def start_tracing() -> None:
    """Record every question asked to the API as an MLflow trace, LLM calls included."""
    mlflow.set_experiment(TRACES_EXPERIMENT)
    mlflow.openai.autolog()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    start_tracing()
    app.state.llm = make_llm_client()
    app.state.http = make_http_client(os.environ["FOOTBALL_DATA_API_KEY"])
    # Predictions need this season's latest results, so the history is refreshed at start-up.
    # If football-data.co.uk cannot be reached, the history of a previous start is good enough.
    with httpx.Client(timeout=30.0) as history_http:
        try:
            download_history(history_http, RAW_DATA_DIR, date.today())
        except httpx.HTTPError as error:
            logger.warning(
                "Could not refresh the match history, using the files already downloaded: %s",
                error,
            )
    app.state.predictor = load_predictor(RAW_DATA_DIR, date.today())
    yield
    app.state.http.close()
    app.state.llm.close()
    # Traces are written in the background: finish writing them before the process exits.
    mlflow.flush_trace_async_logging()


app = FastAPI(title="Football agent", lifespan=lifespan)


@app.get("/")
def chat_page() -> FileResponse:
    return FileResponse(CHAT_PAGE_PATH)


@app.post("/ask")
def ask(body: AskRequest, request: Request) -> AskResponse:
    # One SQLite connection per request: FastAPI runs sync endpoints in a thread pool,
    # and a connection cannot be shared across threads.
    cache = open_cache(CACHE_PATH)
    try:
        context = ToolContext(
            football=FootballDataClient(request.app.state.http, cache),
            predictor=request.app.state.predictor,
            today=date.today(),
        )
        reply = answer(body.question, request.app.state.llm, context)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    finally:
        cache.close()
    return AskResponse(answer=reply)
