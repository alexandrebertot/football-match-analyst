import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from football_agent.data_api import open_cache


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = open_cache(tmp_path / "cache" / "responses.sqlite")
    yield db
    db.close()
