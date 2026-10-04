from pathlib import Path

import httpx

BASE_URL = "https://football-data.co.uk/mmz4281"
RAW_DATA_DIR = Path("data/raw")
# Our competition codes (the ones the agent uses) mapped to football-data.co.uk file names.
LEAGUE_FILES = {"PL": "E0", "FL1": "F1", "BL1": "D1", "SA": "I1", "PD": "SP1"}
# Start years of the 10 complete seasons, 2016-17 to 2025-26.
SEASONS = range(2016, 2026)


def season_code(season_start: int) -> str:
    """Return the football-data.co.uk season code, e.g. '2526' for the season starting in 2025."""
    return f"{season_start % 100:02d}{(season_start + 1) % 100:02d}"


def csv_url(competition: str, season_start: int) -> str:
    return f"{BASE_URL}/{season_code(season_start)}/{LEAGUE_FILES[competition]}.csv"


def raw_csv_path(data_dir: Path, competition: str, season_start: int) -> Path:
    return data_dir / f"{competition}_{season_code(season_start)}.csv"


def download_season(
    http: httpx.Client, data_dir: Path, competition: str, season_start: int
) -> Path:
    """Download one season's CSV into `data_dir`, unless it is already there."""
    path = raw_csv_path(data_dir, competition, season_start)
    if path.exists():
        return path
    response = http.get(csv_url(competition, season_start))
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


if __name__ == "__main__":
    with httpx.Client(timeout=30.0) as http:
        for competition in LEAGUE_FILES:
            for season_start in SEASONS:
                download_season(http, RAW_DATA_DIR, competition, season_start)
