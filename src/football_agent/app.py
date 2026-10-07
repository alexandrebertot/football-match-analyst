import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from football_agent.agent import answer
from football_agent.data_api import FootballDataClient, make_http_client, open_cache
from football_agent.llm import make_llm_client
from football_agent.predictor.data import RAW_DATA_DIR, download_history
from football_agent.predictor.predict import load_predictor
from football_agent.tools import ToolContext

CACHE_PATH = Path(".cache/football_data.sqlite")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.llm = make_llm_client()
    app.state.http = make_http_client(os.environ["FOOTBALL_DATA_API_KEY"])
    # Predictions need this season's latest results, so the history is refreshed at start-up.
    with httpx.Client(timeout=30.0) as history_http:
        download_history(history_http, RAW_DATA_DIR, date.today())
    app.state.predictor = load_predictor(RAW_DATA_DIR, date.today())
    yield
    app.state.http.close()
    app.state.llm.close()


app = FastAPI(title="Football agent", lifespan=lifespan)


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
