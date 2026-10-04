from pathlib import Path

import httpx
import pandas as pd
import pytest

from football_agent.predictor.data import (
    COLUMNS,
    LEAGUE_FILES,
    csv_url,
    download_season,
    load_matches,
    load_season,
    raw_csv_path,
    season_code,
)

# Shaped like the 2016-17 files: two-digit years and no average closing odds.
OLD_FORMAT_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,PSCH,PSCD,PSCA\n"
    "E0,13/08/16,Burnley,Swansea,0,1,A,10,17,3,9,2.54,3.27,3.04\n"
)
# Shaped like the 2025-26 files: UTF-8 BOM, a Time column, four-digit years and extra columns.
NEW_FORMAT_CSV = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Referee,HS,AS,HST,AST,"
    "PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA\n"
    "E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,A Taylor,19,10,10,3,"
    "1.29,6.55,9.75,1.29,6.02,8.68\n"
)


@pytest.mark.parametrize(
    ("season_start", "expected"), [(2025, "2526"), (2016, "1617"), (1999, "9900")]
)
def test_season_code_joins_two_digit_years(season_start: int, expected: str) -> None:
    assert season_code(season_start) == expected


def test_csv_url_uses_football_data_file_name_of_the_competition() -> None:
    assert csv_url("FL1", 2025) == "https://football-data.co.uk/mmz4281/2526/F1.csv"


def fake_http(requests: list[httpx.Request], status_code: int = 200) -> httpx.Client:
    """Return an HTTP client whose fake server records requests and answers with a tiny CSV."""

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, content=b"Div,Date\nF1,15/08/2025\n")

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_download_season_saves_the_csv(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    path = download_season(fake_http(requests), tmp_path / "raw", "FL1", 2025)

    assert path == raw_csv_path(tmp_path / "raw", "FL1", 2025)
    assert path.read_bytes() == b"Div,Date\nF1,15/08/2025\n"
    assert str(requests[0].url) == "https://football-data.co.uk/mmz4281/2526/F1.csv"


def test_download_season_skips_files_already_downloaded(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    http = fake_http(requests)

    download_season(http, tmp_path, "FL1", 2025)
    download_season(http, tmp_path, "FL1", 2025)

    assert len(requests) == 1


def test_download_season_raises_on_http_error_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        download_season(fake_http([], status_code=404), tmp_path, "FL1", 2025)

    assert not raw_csv_path(tmp_path, "FL1", 2025).exists()


def test_load_season_renames_columns_and_drops_the_others(tmp_path: Path) -> None:
    path = tmp_path / "PL_2526.csv"
    path.write_text(NEW_FORMAT_CSV, encoding="utf-8")

    matches = load_season(path, "PL", 2025)

    assert list(matches.columns) == [*COLUMNS.values(), "competition", "season"]
    match = matches.iloc[0]
    assert match["date"] == pd.Timestamp("2025-08-15")
    assert (match["home_team"], match["away_team"], match["result"]) == (
        "Liverpool",
        "Bournemouth",
        "H",
    )
    assert (match["home_goals"], match["away_goals"]) == (4, 2)
    assert match["pinnacle_odds_home"] == 1.29
    assert (match["competition"], match["season"]) == ("PL", 2025)


def test_load_season_reads_old_format_files(tmp_path: Path) -> None:
    path = tmp_path / "PL_1617.csv"
    path.write_text(OLD_FORMAT_CSV, encoding="utf-8")

    match = load_season(path, "PL", 2016).iloc[0]

    assert match["date"] == pd.Timestamp("2016-08-13")
    assert match["pinnacle_odds_home"] == 2.54
    assert pd.isna(match["average_odds_home"])


def test_load_matches_stacks_every_competition_sorted_by_date(tmp_path: Path) -> None:
    for i, competition in enumerate(LEAGUE_FILES):
        # One match per competition, dated in reverse order of the loading loop.
        csv = NEW_FORMAT_CSV.replace("15/08/2025", f"{20 - i}/08/2025")
        raw_csv_path(tmp_path, competition, 2025).write_text(csv, encoding="utf-8")

    matches = load_matches(tmp_path, range(2025, 2026))

    assert len(matches) == len(LEAGUE_FILES)
    assert matches["date"].is_monotonic_increasing
    assert list(matches["competition"]) == list(reversed(LEAGUE_FILES))
