import argparse
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from football_agent.agent_eval.snapshot import PAUSE_SECONDS
from football_agent.data_api import make_http_client
from football_agent.tools import COMPETITIONS

SEASONS_DIR = Path("data/seasons")
RESOURCES = ["matches", "teams"]
# The tables count the league matches and, in the Champions League, the league phase only.
TABLE_STAGES = ("REGULAR_SEASON", "LEAGUE_STAGE")
# A result awarded by the league (a forfeit, for example) counts in the table like a played one.
RESULT_STATUSES = ("FINISHED", "AWARDED")


def download_season(
    http: httpx.Client, season: int, pause_seconds: float = PAUSE_SECONDS
) -> dict[str, Any]:
    """Download every match and team of every competition over a whole season, as the API
    returns them."""
    competitions: dict[str, dict[str, Any]] = {}
    for competition in COMPETITIONS:
        competitions[competition] = {}
        for resource in RESOURCES:
            path = f"/competitions/{competition}/{resource}"
            response = http.get(path, params={"season": season})
            response.raise_for_status()
            competitions[competition][resource] = response.json()
            time.sleep(pause_seconds)
    return {"season": season, "competitions": competitions}


def kicks_off_before(match: dict[str, Any], day: date) -> bool:
    return date.fromisoformat(match["utcDate"][:10]) < day


def as_upcoming(match: dict[str, Any]) -> dict[str, Any]:
    """The match as it was before kick-off: scheduled, without a score."""
    no_goals = {"home": None, "away": None}
    score = {**match["score"], "winner": None, "fullTime": no_goals, "halfTime": no_goals}
    return {**match, "status": "TIMED", "score": score}


def standings_table(
    teams: list[dict[str, Any]], matches: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The table of `teams` after the finished `matches`, in the shape of the API standings.

    Ties are broken on goal difference, then goals scored, then name: some leagues use other
    rules, such as head-to-head results, so a few places may differ from the official table.
    """
    records = {
        team["id"]: {
            "team": {"id": team["id"], "name": team["name"], "shortName": team["shortName"]},
            "playedGames": 0,
            "won": 0,
            "draw": 0,
            "lost": 0,
            "goalsFor": 0,
            "goalsAgainst": 0,
        }
        for team in teams
    }
    for match in matches:
        goals = match["score"]["fullTime"]
        sides = [
            (match["homeTeam"]["id"], goals["home"], goals["away"]),
            (match["awayTeam"]["id"], goals["away"], goals["home"]),
        ]
        for team_id, scored, conceded in sides:
            record = records[team_id]
            record["playedGames"] += 1
            record["goalsFor"] += scored
            record["goalsAgainst"] += conceded
            if scored > conceded:
                record["won"] += 1
            elif scored == conceded:
                record["draw"] += 1
            else:
                record["lost"] += 1
    rows = [
        {
            **record,
            "goalDifference": record["goalsFor"] - record["goalsAgainst"],
            "points": 3 * record["won"] + record["draw"],
        }
        for record in records.values()
    ]
    rows.sort(
        key=lambda row: (
            -row["points"],
            -row["goalDifference"],
            -row["goalsFor"],
            row["team"]["shortName"],
        )
    )
    return [{"position": place, **row} for place, row in enumerate(rows, 1)]


def snapshot_as_of(season: dict[str, Any], day: date) -> dict[str, Any]:
    """The season as it stood on the morning of `day`, in the format of the evaluation
    snapshots: later matches are still to play, and the tables only count the results known
    by then."""
    competitions = {}
    for competition, recorded in season["competitions"].items():
        matches = recorded["matches"]["matches"]
        results = [
            match
            for match in matches
            if kicks_off_before(match, day)
            and match["status"] in RESULT_STATUSES
            and match["stage"] in TABLE_STAGES
        ]
        table = standings_table(recorded["teams"]["teams"], results)
        competitions[competition] = {
            "standings": {
                "competition": {"name": recorded["matches"]["competition"]["name"]},
                "season": {"startDate": matches[0]["season"]["startDate"]},
                "standings": [{"type": "TOTAL", "table": table}],
            },
            "teams": recorded["teams"],
            "matches": {
                "matches": [
                    match if kicks_off_before(match, day) else as_upcoming(match)
                    for match in matches
                ]
            },
        }
    return {"date": day.isoformat(), "competitions": competitions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a whole past season for distillation.")
    parser.add_argument("--season", type=int, required=True, help="Start year, e.g. 2025.")
    args = parser.parse_args()
    with make_http_client(os.environ["FOOTBALL_DATA_API_KEY"]) as http:
        season = download_season(http, args.season)
    path = SEASONS_DIR / f"{args.season}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(season), encoding="utf-8")
    print(f"Season written to {path}")
