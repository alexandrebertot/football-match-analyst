from pathlib import Path

import httpx
import pytest

from football_agent.predictor.data import (
    csv_url,
    download_season,
    raw_csv_path,
    season_code,
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
