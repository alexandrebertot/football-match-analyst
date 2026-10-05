from pathlib import Path

import yaml

TEAM_NAMES_PATH = Path(__file__).with_name("team_names.yaml")


def read_team_names() -> dict[str, dict[int, str]]:
    """Return, per competition, each football-data.org team id and its name in the CSV files."""
    return yaml.safe_load(TEAM_NAMES_PATH.read_text(encoding="utf-8"))


def load_team_names() -> dict[int, str]:
    """Map each football-data.org team id to its name in the historical CSV files."""
    return {
        team_id: name for teams in read_team_names().values() for team_id, name in teams.items()
    }
