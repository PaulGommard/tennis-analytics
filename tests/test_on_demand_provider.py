from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx

from tennis_analytics.product.providers.espn import EspnTennisProvider


def test_espn_provider_filters_to_resolved_mens_singles_and_parses_results(tmp_path):
    starts = (datetime.now(UTC) + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    scoreboard = {
        "events": [{
            "name": "Rolex Shanghai Masters",
            "groupings": [{
                "grouping": {"slug": "mens-singles"},
                "competitions": [{
                    "id": "42", "date": starts,
                    "status": {"type": {"state": "pre"}},
                    "type": {"slug": "mens-singles"},
                    "round": {"displayName": "Round 1"},
                    "venue": {"fullName": "Shanghai"},
                    "competitors": [
                        {"id": "1", "athlete": {"fullName": "Player One"}},
                        {"id": "2", "athlete": {"fullName": "Player Two"}},
                    ],
                }],
            }],
        }],
    }
    results = """
    <div class="mod-container mod-page-header"><h1>Player One</h1></div>
    <span>ATP Rank</span> #12 <span>Birth Date</span>October 28, 1997 (Age: 28)
    <div class="gradient-container"><div class="game-details"><p><strong><a>Japan Open</a></strong>
    - Tokyo<br>September 27, 2026 to October 4, 2026</p></div></div>
    <table><tr class="total"><td>Men's Singles</td></tr><tr class="oddrow">
    <td>1st</td><td><a href="/tennis/player/_/id/9/opponent">Opponent</a></td>
    <td>W</td><td>6-4, 6-4</td></tr></table>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(scoreboard) if "scoreboard" in str(request.url) else results
        return httpx.Response(200, text=body)

    provider = EspnTennisProvider(
        cache_dir=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    matches = provider.get_upcoming_matches(days=1)
    player = provider.get_player_recent_matches("1", year=2026)

    assert [match["external_match_id"] for match in matches] == ["espn:42"]
    assert matches[0]["surface"] == "Hard"
    assert player["name"] == "Player One"
    assert player["rank"] == 12
    assert player["matches"][0]["opponent"]["external_id"] == "9"
    assert provider.request_count == 2


def test_espn_provider_disk_cache_avoids_second_request(tmp_path):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"events": []})

    provider = EspnTennisProvider(
        cache_dir=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider.get_upcoming_matches(days=1)
    provider.get_upcoming_matches(days=1)

    assert calls == 1
    assert provider.request_count == 1
