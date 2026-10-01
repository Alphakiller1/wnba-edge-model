import pandas as pd

from wnba_edges import export


def test_kickoff_is_the_game_start_not_the_build_time():
    row = pd.Series({"date": "2026-10-01", "time": "2026-10-02T01:00Z",
                     "generated_at": "2026-10-01T11:28:55Z"})
    human, utc = export._kickoff(row)
    assert utc == "2026-10-02T01:00:00Z"
    assert human == "2026-10-01 9:00 PM ET"


def test_kickoff_without_a_start_time_publishes_no_utc():
    human, utc = export._kickoff(pd.Series({"date": "2026-10-01", "generated_at": "2026-10-01T11:28:55Z"}))
    assert utc is None and human == "2026-10-01"
