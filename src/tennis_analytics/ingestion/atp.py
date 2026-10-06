from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from tennis_analytics.ingestion.downloader import Artifact, download_artifacts
from tennis_analytics.ingestion.sources import ATP_ARCHIVE
from tennis_analytics.paths import RAW_DIR

MATCH_COLUMNS = (
    "tourney_id",
    "tourney_name",
    "surface",
    "tourney_date",
    "winner_id",
    "winner_name",
    "loser_id",
    "loser_name",
    "score",
)
LOGGER = logging.getLogger(__name__)


def build_artifacts(
    start_year: int,
    end_year: int,
    destination_root: Path = RAW_DIR / "atp_archive",
    include_reference: bool = True,
) -> list[Artifact]:
    if start_year < 1968:
        raise ValueError("ATP season files begin in 1968")
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")
    artifacts = [
        Artifact(
            source=ATP_ARCHIVE,
            remote_name=f"atp_matches_{year}.csv",
            destination=destination_root / "matches" / f"atp_matches_{year}.csv",
            dataset="atp_matches",
            year=year,
            required_columns=MATCH_COLUMNS,
        )
        for year in range(start_year, end_year + 1)
    ]
    if include_reference:
        for filename in (
            "atp_players.csv",
            "matches_data_dictionary.txt",
            "UPSTREAM_README.md",
        ):
            artifacts.append(
                Artifact(
                    source=ATP_ARCHIVE,
                    remote_name=filename,
                    destination=destination_root / "metadata" / filename,
                    dataset="atp_reference",
                )
            )
    return artifacts


def parser() -> argparse.ArgumentParser:
    current_year = datetime.now(UTC).year
    command = argparse.ArgumentParser(description="Download immutable ATP season CSV files")
    command.add_argument("--start-year", type=int, default=2000)
    command.add_argument("--end-year", type=int, default=current_year)
    command.add_argument(
        "--no-reference-files",
        action="store_true",
        help="Do not fetch players and upstream documentation",
    )
    command.add_argument("--verbose", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    artifacts = build_artifacts(
        args.start_year,
        args.end_year,
        include_reference=not args.no_reference_files,
    )
    downloaded, reused = download_artifacts(artifacts)
    LOGGER.info("Complete: %d downloaded, %d already present", downloaded, reused)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
