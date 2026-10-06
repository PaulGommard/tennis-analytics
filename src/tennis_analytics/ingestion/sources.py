from __future__ import annotations

from dataclasses import dataclass

LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


@dataclass(frozen=True)
class Source:
    source_id: str
    name: str
    original_url: str
    distribution_url: str
    raw_base_url: str
    license_id: str = "CC BY-NC-SA 4.0"
    license_url: str = LICENSE_URL
    commercial_use: bool = False
    snapshot: str | None = None


ATP_ARCHIVE = Source(
    source_id="atp_archive",
    name="Jeff Sackmann ATP data — archival mirror",
    original_url="https://github.com/JeffSackmann/tennis_atp",
    distribution_url="https://github.com/Aneeshers/tennis-sackmann-archive/tree/main/atp",
    raw_base_url=("https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp"),
    snapshot="Mirror reports an upstream snapshot from June 2026.",
)

SLAM_POINT_BY_POINT = Source(
    source_id="slam_pointbypoint",
    name="Jeff Sackmann Grand Slam point-by-point — archival mirror",
    original_url="https://github.com/JeffSackmann/tennis_slam_pointbypoint",
    distribution_url=(
        "https://github.com/Aneeshers/tennis-sackmann-archive/tree/main/slam_pointbypoint"
    ),
    raw_base_url=(
        "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/slam_pointbypoint"
    ),
    snapshot="Upstream commit 6febb77, October 2024; coverage ends in 2024.",
)

MATCH_CHARTING_PROJECT = Source(
    source_id="match_charting_project",
    name="Tennis Abstract Match Charting Project",
    original_url="https://github.com/JeffSackmann/tennis_MatchChartingProject",
    distribution_url="https://github.com/JeffSackmann/tennis_MatchChartingProject",
    raw_base_url=(
        "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
    ),
    snapshot="Live master branch; every downloaded file is pinned locally by SHA-256.",
)
