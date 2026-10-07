from datetime import date
from pathlib import Path

import httpx
import pandas as pd

BASE_URL = "https://football-data.co.uk/mmz4281"
RAW_DATA_DIR = Path("data/raw")
LEAGUE_FILES = {"PL": "E0", "FL1": "F1", "BL1": "D1", "SA": "I1", "PD": "SP1"}
# Start year of the first season downloaded; dataset configs choose their seasons from there on.
FIRST_SEASON = 2016
# football-data.co.uk columns we keep, renamed. Odds are closing odds (just before kick-off).
COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "PSCH": "pinnacle_odds_home",
    "PSCD": "pinnacle_odds_draw",
    "PSCA": "pinnacle_odds_away",
    "AvgCH": "average_odds_home",
    "AvgCD": "average_odds_draw",
    "AvgCA": "average_odds_away",
}


def season_code(season_start: int) -> str:
    """Return the football-data.co.uk season code, e.g. '2526' for the season starting in 2025."""
    return f"{season_start % 100:02d}{(season_start + 1) % 100:02d}"


def csv_url(competition: str, season_start: int) -> str:
    return f"{BASE_URL}/{season_code(season_start)}/{LEAGUE_FILES[competition]}.csv"


def raw_csv_path(data_dir: Path, competition: str, season_start: int) -> Path:
    return data_dir / f"{competition}_{season_code(season_start)}.csv"


def current_season(today: date) -> int:
    """Start year of the season under way on `today`: seasons start in August."""
    return today.year if today.month >= 8 else today.year - 1


def download_season(
    http: httpx.Client,
    data_dir: Path,
    competition: str,
    season_start: int,
    overwrite: bool = False,
) -> Path:
    """Download one season's CSV into `data_dir`, unless it is already there and not overwritten."""
    path = raw_csv_path(data_dir, competition, season_start)
    if path.exists() and not overwrite:
        return path
    response = http.get(csv_url(competition, season_start))
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def load_season(path: Path, competition: str, season_start: int) -> pd.DataFrame:
    """Load one season's CSV with our column names; columns missing from old files become NaN."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    matches = raw.reindex(columns=list(COLUMNS)).rename(columns=COLUMNS)
    # Older files write two-digit years (13/08/16), newer ones four-digit years (15/08/2025).
    date_format = "%d/%m/%y" if len(matches["date"].iloc[0]) == 8 else "%d/%m/%Y"
    matches["date"] = pd.to_datetime(matches["date"], format=date_format)
    matches["competition"] = competition
    matches["season"] = season_start
    return matches


def load_matches(data_dir: Path, seasons: range) -> pd.DataFrame:
    """Load every competition over `seasons` into one table sorted by date."""
    seasons_tables = [
        load_season(raw_csv_path(data_dir, competition, season_start), competition, season_start)
        for competition in LEAGUE_FILES
        for season_start in seasons
    ]
    matches = pd.concat(seasons_tables, ignore_index=True)
    return matches.sort_values("date", kind="stable", ignore_index=True)


def history_seasons(today: date) -> range:
    """Seasons from the first one we use up to the one under way on `today`."""
    return range(FIRST_SEASON, current_season(today) + 1)


def download_history(http: httpx.Client, data_dir: Path, today: date) -> None:
    """Download each finished season once, and the season under way every time (it grows weekly).

    The season under way is skipped as long as football-data.co.uk has not published it.
    """
    season_under_way = current_season(today)
    for competition in LEAGUE_FILES:
        for season_start in history_seasons(today):
            overwrite = season_start == season_under_way
            try:
                download_season(http, data_dir, competition, season_start, overwrite)
            except httpx.HTTPStatusError as error:
                # A season's file only appears with its first results, around mid-August.
                if not (overwrite and error.response.status_code == 404):
                    raise


def load_history(data_dir: Path, today: date) -> pd.DataFrame:
    """Every match played so far: the finished seasons, plus the season under way once published."""
    season_under_way = current_season(today)
    finished = load_matches(data_dir, range(FIRST_SEASON, season_under_way))
    under_way = []
    for competition in LEAGUE_FILES:
        path = raw_csv_path(data_dir, competition, season_under_way)
        if path.exists():
            under_way.append(load_season(path, competition, season_under_way))
    matches = pd.concat([finished, *under_way], ignore_index=True)
    return matches.sort_values("date", kind="stable", ignore_index=True)


if __name__ == "__main__":
    with httpx.Client(timeout=30.0) as http:
        download_history(http, RAW_DATA_DIR, date.today())
