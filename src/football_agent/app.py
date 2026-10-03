import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from football_agent.agent import answer
from football_agent.data_api import FootballDataClient, make_http_client, open_cache
from football_agent.llm import make_llm_client

CACHE_PATH = Path(".cache/football_data.sqlite")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.llm = make_llm_client()
    app.state.http = make_http_client(os.environ["FOOTBALL_DATA_API_KEY"])
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
        football = FootballDataClient(request.app.state.http, cache)
        reply = answer(body.question, request.app.state.llm, football, date.today())
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    finally:
        cache.close()
    return AskResponse(answer=reply)
