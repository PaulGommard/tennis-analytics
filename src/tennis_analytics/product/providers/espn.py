from __future__ import annotations

import hashlib
import html
import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from tennis_analytics.paths import DATA_DIR
from tennis_analytics.product.providers.historical import _normalize
from tennis_analytics.product.providers.on_demand import OnDemandTennisDataProvider

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
RESULTS_URL = "https://www.espn.com/tennis/player/results/_/id/{player_id}/year/{year}"
RANKINGS_URL = "https://sports.core.api.espn.com/v2/sports/tennis/leagues/atp/rankings?limit=10&lang=en&region=us"


class ProviderError(RuntimeError):
    pass


def _plain(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


class EspnTennisProvider(OnDemandTennisDataProvider):
    """Read-only ESPN adapter with a bounded request budget and disk cache.

    ESPN does not publish this endpoint as a supported commercial API. Every
    returned record therefore keeps its provenance and parsing failures are
    surfaced instead of being silently guessed.
    """

    provider_name = "espn_public"

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        ttl_seconds: int = 1_800,
        max_requests: int = 6,
        client: httpx.Client | None = None,
    ) -> None:
        self.cache_dir = cache_dir or DATA_DIR / "runtime" / "http_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.max_requests = max_requests
        self._request_count = 0
        self._client = client or httpx.Client(
            timeout=10,
            follow_redirects=True,
            headers={"User-Agent": "tennis-analytics-local/1.0 (personal research)"},
        )

    @property
    def request_count(self) -> int:
        return self._request_count

    def reset_budget(self) -> None:
        self._request_count = 0

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.json"

    def _get(self, url: str) -> tuple[str, dict[str, Any]]:
        path = self._cache_path(url)
        now = time.time()
        cached: dict[str, Any] | None = None
        if path.exists():
            try:
                cached = json.loads(path.read_text())
                if now - float(cached["fetched_epoch"]) <= self.ttl_seconds:
                    return str(cached["body"]), {"cache": "fresh", "fetched_at": cached["fetched_at"]}
            except (ValueError, KeyError, OSError):
                cached = None
        if self._request_count >= self.max_requests:
            if cached:
                return str(cached["body"]), {"cache": "stale", "fetched_at": cached["fetched_at"]}
            raise ProviderError(f"External request budget exhausted ({self.max_requests}).")
        last_error: Exception | None = None
        for attempt in range(2):
            if self._request_count >= self.max_requests:
                break
            self._request_count += 1
            try:
                response = self._client.get(url)
                response.raise_for_status()
                stamp = datetime.now(UTC).isoformat()
                path.write_text(json.dumps({"fetched_epoch": now, "fetched_at": stamp, "body": response.text}))
                return response.text, {"cache": "miss", "fetched_at": stamp}
            except (httpx.HTTPError, OSError) as error:
                last_error = error
                if attempt == 0 and self._request_count < self.max_requests:
                    time.sleep(0.2)
        if cached:
            return str(cached["body"]), {"cache": "stale", "fetched_at": cached["fetched_at"]}
        raise ProviderError(f"ESPN request failed: {last_error}") from last_error

    @staticmethod
    def _surface(tournament: str) -> str:
        normalized = _normalize(tournament)
        aliases = {
            "rolex shanghai masters": "Hard",
            "shanghai masters": "Hard",
            "kinoshita group japan open tennis championships": "Hard",
            "japan open": "Hard",
            "china open": "Hard",
            "aito hangzhou open": "Hard",
            "cincinnati open": "Hard",
            "national bank open presented by rogers": "Hard",
            "mifel tennis open by telcel oppo": "Hard",
            "open occitanie": "Hard",
            "millennium estoril open": "Clay",
            "efg swiss open gstaad": "Clay",
            "bitpanda hamburg open": "Clay",
            "internazionali bnl d italia": "Clay",
            "mutua madrid open": "Clay",
            "rio open presented by claro": "Clay",
            "barcelona open banc sabadell": "Clay",
            "roland garros": "Clay",
            "wimbledon": "Grass",
            "us open": "Hard",
            "australian open": "Hard",
        }
        return aliases.get(normalized, "unknown")

    def get_upcoming_matches(self, *, days: int = 3) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        end = now + timedelta(days=days)
        # ESPN's tennis feed expands the active tournament from one calendar day;
        # its apparent range syntax can silently return only the first tournament.
        body, provenance = self._get(f"{SCOREBOARD_URL}?dates={now:%Y%m%d}")
        payload = json.loads(body)
        matches: list[dict[str, Any]] = []
        for event in payload.get("events", []):
            tournament = event.get("name") or event.get("shortName") or "Unknown tournament"
            for grouping in event.get("groupings", []):
                group = grouping.get("grouping", {})
                if group.get("slug") != "mens-singles":
                    continue
                for competition in grouping.get("competitions", []):
                    if competition.get("type", {}).get("slug") != "mens-singles":
                        continue
                    if competition.get("status", {}).get("type", {}).get("state") != "pre":
                        continue
                    try:
                        starts = datetime.fromisoformat(competition["date"])
                    except (KeyError, ValueError):
                        continue
                    if starts < now - timedelta(hours=2) or starts > end:
                        continue
                    competitors = competition.get("competitors", [])
                    if len(competitors) != 2:
                        continue
                    players = []
                    for competitor in competitors:
                        athlete = competitor.get("athlete", {})
                        external_id = str(competitor.get("id", ""))
                        name = athlete.get("fullName") or athlete.get("displayName")
                        if not name or name == "TBD" or external_id.startswith("-"):
                            players = []
                            break
                        players.append({"external_id": external_id, "name": name})
                    if len(players) != 2:
                        continue
                    matches.append(
                        {
                            "external_match_id": f"espn:{competition['id']}",
                            "starts_at": starts.isoformat(),
                            "tournament": tournament,
                            "round": competition.get("round", {}).get("displayName", "unknown"),
                            "surface": self._surface(tournament),
                            "venue": competition.get("venue", {}).get("fullName"),
                            "player_a": players[0],
                            "player_b": players[1],
                            "provider": self.provider_name,
                            "source_fetched_at": provenance["fetched_at"],
                            "source_cache": provenance["cache"],
                        }
                    )
        matches.sort(key=lambda row: row["starts_at"])
        return matches

    def get_player_recent_matches(
        self, external_player_id: str, *, year: int
    ) -> dict[str, Any]:
        body, provenance = self._get(
            RESULTS_URL.format(player_id=external_player_id, year=year)
        )
        heading = re.search(
            r'<div class="[^"]*mod-page-header[^"]*"[^>]*>.*?<h1[^>]*>(.*?)</h1>',
            body,
            re.IGNORECASE | re.DOTALL,
        )
        player_name = _plain(heading.group(1)) if heading else external_player_id
        rank_match = re.search(r"ATP Rank</span>\s*#(\d+)", body, re.IGNORECASE)
        birth_match = re.search(r"Birth Date</span>\s*([^<(]+)", body, re.IGNORECASE)
        hand_match = re.search(r"Plays:\s*</?[^>]*>?\s*(Right|Left)|Plays:\s*(Right|Left)", body, re.IGNORECASE)
        blocks = re.split(r'<div class="gradient-container"[^>]*>', body, flags=re.IGNORECASE)[1:]
        rows: list[dict[str, Any]] = []
        for block in blocks:
            header, _, tail = block.partition("</div></div>")
            tournament_match = re.search(r"<strong>\s*<a[^>]*>(.*?)</a>", header, re.IGNORECASE | re.DOTALL)
            date_match = re.search(
                r"<br>\s*([A-Z][a-z]+ \d{1,2}, \d{4})\s+to", header, re.IGNORECASE
            )
            table_match = re.search(r"<table[^>]*>(.*?)</table>", tail, re.IGNORECASE | re.DOTALL)
            if not tournament_match or not date_match or not table_match:
                continue
            if "Men's Singles" not in _plain(table_match.group(1)):
                continue
            tournament = _plain(tournament_match.group(1))
            try:
                match_date = (
                    datetime.strptime(date_match.group(1), "%B %d, %Y")
                    .replace(tzinfo=UTC)
                    .date()
                    .isoformat()
                )
            except ValueError:
                continue
            for row_html in re.findall(
                r'<tr class="(?:oddrow|evenrow)"[^>]*>(.*?)</tr>', table_match.group(1), re.IGNORECASE | re.DOTALL
            ):
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.IGNORECASE | re.DOTALL)
                if len(cells) != 4:
                    continue
                result = _plain(cells[2]).upper()
                opponent_link = re.search(r'/id/(\d+)/[^"?]*[^>]*>(.*?)</a>', cells[1], re.IGNORECASE | re.DOTALL)
                if result not in {"W", "L"} or not opponent_link:
                    continue
                opponent_name = _plain(opponent_link.group(2))
                if "/" in opponent_name or "&" in opponent_name:
                    continue
                rows.append(
                    {
                        "date": match_date,
                        "tournament": tournament,
                        "surface": self._surface(tournament),
                        "round": _plain(cells[0]),
                        "opponent": {
                            "external_id": opponent_link.group(1),
                            "name": opponent_name,
                        },
                        "result": result,
                        "score": _plain(cells[3]),
                    }
                )
        return {
            "external_player_id": external_player_id,
            "name": player_name,
            "rank": int(rank_match.group(1)) if rank_match else None,
            "rank_points": None,
            "birth_date_text": birth_match.group(1).strip() if birth_match else None,
            "hand": next((value.title() for value in (hand_match.groups() if hand_match else ()) if value), None),
            "matches": rows,
            "provider": self.provider_name,
            "source_fetched_at": provenance["fetched_at"],
            "source_cache": provenance["cache"],
        }

    def get_rankings(self) -> dict[str, Any]:
        """Return the current weekly top-150 index keyed by ESPN athlete id."""
        index_body, _ = self._get(RANKINGS_URL)
        index = json.loads(index_body)
        try:
            ranking_url = str(index["items"][0]["$ref"]).replace("http://", "https://")
        except (KeyError, IndexError) as error:
            raise ProviderError("ESPN ranking index has no current ranking resource.") from error
        ranking_body, provenance = self._get(ranking_url)
        payload = json.loads(ranking_body)
        rankings: dict[str, dict[str, float | int]] = {}
        for item in payload.get("ranks", []):
            athlete_ref = item.get("athlete", {}).get("$ref", "")
            athlete_id = re.search(r"/athletes/(\d+)", athlete_ref)
            if athlete_id:
                rankings[athlete_id.group(1)] = {
                    "rank": int(item["current"]),
                    "rank_points": float(item["points"]),
                }
        return {
            "as_of": payload.get("lastUpdated"),
            "players": rankings,
            "source_cache": provenance["cache"],
        }
