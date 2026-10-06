from collections import Counter

from football_agent.predictor.teams import load_team_names, read_team_names

TEAMS_PER_COMPETITION = {"PL": 20, "FL1": 18, "BL1": 18, "SA": 20, "PD": 20}


def test_team_names_cover_every_team_of_the_current_season() -> None:
    by_competition = read_team_names()

    assert {competition: len(teams) for competition, teams in by_competition.items()} == (
        TEAMS_PER_COMPETITION
    )


def test_each_team_id_appears_once() -> None:
    ids = [team_id for teams in read_team_names().values() for team_id in teams]

    assert [team_id for team_id, count in Counter(ids).items() if count > 1] == []


def test_two_teams_of_a_competition_never_share_a_csv_name() -> None:
    for competition, teams in read_team_names().items():
        duplicates = [name for name, count in Counter(teams.values()).items() if count > 1]
        assert duplicates == [], competition


def test_load_team_names_maps_api_ids_to_csv_names() -> None:
    names = load_team_names()

    assert len(names) == sum(TEAMS_PER_COMPETITION.values())
    assert (names[524], names[523], names[80], names[81]) == (
        "Paris SG",
        "Lyon",
        "Espanol",
        "Barcelona",
    )
