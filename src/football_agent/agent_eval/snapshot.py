import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from football_agent.data_api import make_http_client
from football_agent.tools import COMPETITIONS

SNAPSHOTS_DIR = Path("data/snapshots")
RESOURCES = ["standings", "teams", "matches"]
# The free football-data.org plan allows 10 requests per minute.
PAUSE_SECONDS = 7.0


def record_snapshot(
    http: httpx.Client, today: date, pause_seconds: float = PAUSE_SECONDS
) -> dict[str, Any]:
    """Download the standings, teams and season matches of every competition, as the API returns
    them, so that the agent can later be evaluated on data that no longer changes."""
    competitions: dict[str, dict[str, Any]] = {}
    for competition in COMPETITIONS:
        competitions[competition] = {}
        for resource in RESOURCES:
            response = http.get(f"/competitions/{competition}/{resource}")
            response.raise_for_status()
            competitions[competition][resource] = response.json()
            time.sleep(pause_seconds)
    return {"date": today.isoformat(), "competitions": competitions}


class SnapshotClient:
    """Answers the tools from a recorded snapshot, in the same JSON shape as football-data.org."""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.competitions = snapshot["competitions"]

    def get_standings(self, competition: str, season: int | None = None) -> dict[str, Any]:
        standings = self._recorded(competition)["standings"]
        recorded_season = int(standings["season"]["startDate"][:4])
        if season is not None and season != recorded_season:
            raise ValueError(f"The snapshot only holds the season starting in {recorded_season}.")
        return standings

    def get_teams(self, competition: str) -> dict[str, Any]:
        return self._recorded(competition)["teams"]

    def get_matches(self, competition: str, date_from: date, date_to: date) -> dict[str, Any]:
        return {
            "matches": [
                match
                for match in self._season_matches(competition)
                if date_from <= date.fromisoformat(match["utcDate"][:10]) <= date_to
            ]
        }

    def get_team_matches(self, team_id: int, competition: str, limit: int) -> dict[str, Any]:
        finished = sorted(
            (
                match
                for match in self._season_matches(competition)
                if match["status"] == "FINISHED"
                and team_id in (match["homeTeam"]["id"], match["awayTeam"]["id"])
            ),
            key=lambda match: match["utcDate"],
        )
        # Like the API: the last `limit` finished matches, oldest first.
        return {"matches": finished[max(len(finished) - limit, 0) :]}

    def _recorded(self, competition: str) -> dict[str, Any]:
        # The API answers an unknown code with an error the agent can act on; so does the snapshot.
        if competition not in self.competitions:
            raise ValueError(f"Unknown competition '{competition}'.")
        return self.competitions[competition]

    def _season_matches(self, competition: str) -> list[dict[str, Any]]:
        return self._recorded(competition)["matches"]["matches"]


if __name__ == "__main__":
    today = date.today()
    with make_http_client(os.environ["FOOTBALL_DATA_API_KEY"]) as http:
        snapshot = record_snapshot(http, today)
    path = SNAPSHOTS_DIR / f"{today.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    print(f"Snapshot written to {path}")
