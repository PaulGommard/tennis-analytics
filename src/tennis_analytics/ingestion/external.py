from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tennis_analytics.ingestion.downloader import Artifact, download_artifacts
from tennis_analytics.ingestion.sources import MATCH_CHARTING_PROJECT, SLAM_POINT_BY_POINT
from tennis_analytics.paths import EXTERNAL_DIR

LOGGER = logging.getLogger(__name__)

SLAMS = ("ausopen", "frenchopen", "wimbledon", "usopen")
# The October 2024 archive is not a complete Cartesian product. Wimbledon was
# cancelled in 2020, and these other tournament/year pairs are absent upstream.
SLAM_EVENTS_ABSENT = {
    (2020, "wimbledon"),
    (2022, "ausopen"),
    (2022, "frenchopen"),
    (2023, "ausopen"),
    (2023, "frenchopen"),
    (2024, "ausopen"),
    (2024, "frenchopen"),
}
MCP_CORE_FILES = (
    "charting-m-matches.csv",
    "charting-m-points-to-2009.csv",
    "charting-m-points-2010s.csv",
    "charting-m-points-2020s.csv",
    "data_dictionary.txt",
    "README.md",
)
MCP_STATS_FILES = (
    "charting-m-stats-KeyPointsReturn.csv",
    "charting-m-stats-KeyPointsServe.csv",
    "charting-m-stats-NetPoints.csv",
    "charting-m-stats-Overview.csv",
    "charting-m-stats-Rally.csv",
    "charting-m-stats-ReturnDepth.csv",
    "charting-m-stats-ReturnOutcomes.csv",
    "charting-m-stats-ServeBasics.csv",
    "charting-m-stats-ServeDirection.csv",
    "charting-m-stats-ServeInfluence.csv",
    "charting-m-stats-ShotDirOutcomes.csv",
    "charting-m-stats-ShotDirection.csv",
    "charting-m-stats-ShotTypes.csv",
    "charting-m-stats-SnV.csv",
    "charting-m-stats-SvBreakSplit.csv",
    "charting-m-stats-SvBreakTotal.csv",
)


def slam_artifacts(
    start_year: int = 2011,
    end_year: int = 2024,
    destination_root: Path = EXTERNAL_DIR / "slam_pointbypoint",
) -> list[Artifact]:
    if start_year < 2011 or end_year > 2024 or end_year < start_year:
        raise ValueError("The archived Grand Slam point-by-point coverage is 2011–2024")
    artifacts: list[Artifact] = []
    for year in range(start_year, end_year + 1):
        for slam in SLAMS:
            if (year, slam) in SLAM_EVENTS_ABSENT:
                continue
            for kind in ("matches", "points"):
                filename = f"{year}-{slam}-{kind}.csv"
                artifacts.append(
                    Artifact(
                        source=SLAM_POINT_BY_POINT,
                        remote_name=filename,
                        destination=destination_root / str(year) / filename,
                        dataset="slam_pointbypoint",
                        year=year,
                    )
                )
    for filename in ("data_dictionary.txt", "UPSTREAM_README.md"):
        artifacts.append(
            Artifact(
                source=SLAM_POINT_BY_POINT,
                remote_name=filename,
                destination=destination_root / "metadata" / filename,
                dataset="slam_pointbypoint_reference",
            )
        )
    return artifacts


def mcp_artifacts(
    *,
    include_stats: bool = False,
    destination_root: Path = EXTERNAL_DIR / "match_charting_project",
) -> list[Artifact]:
    files = MCP_CORE_FILES + (MCP_STATS_FILES if include_stats else ())
    return [
        Artifact(
            source=MATCH_CHARTING_PROJECT,
            remote_name=filename,
            destination=destination_root / filename,
            dataset="match_charting_project",
        )
        for filename in files
    ]


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Download point- and shot-level datasets")
    command.add_argument(
        "--dataset",
        choices=("slam-pointbypoint", "mcp", "all"),
        default="all",
    )
    command.add_argument("--start-year", type=int, default=2011)
    command.add_argument("--end-year", type=int, default=2024)
    command.add_argument("--include-mcp-stats", action="store_true")
    command.add_argument("--verbose", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    artifacts: list[Artifact] = []
    if args.dataset in ("slam-pointbypoint", "all"):
        artifacts.extend(slam_artifacts(args.start_year, args.end_year))
    if args.dataset in ("mcp", "all"):
        artifacts.extend(mcp_artifacts(include_stats=args.include_mcp_stats))
    downloaded, reused = download_artifacts(artifacts)
    LOGGER.info("Complete: %d downloaded, %d already present", downloaded, reused)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
