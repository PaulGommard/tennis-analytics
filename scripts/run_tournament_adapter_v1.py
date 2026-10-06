from __future__ import annotations

import argparse
import json

from tennis_analytics.tournament.modeling import train_tournament_adapter
from tennis_analytics.tournament.replay import build_tournament_replay
from tennis_analytics.tournament.reporting import write_tournament_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, train, and report Tournament Adapter V1")
    parser.add_argument(
        "--reuse-replay",
        action="store_true",
        help="Reuse the existing tournament_context_v1.parquet",
    )
    args = parser.parse_args()
    result = {}
    if not args.reuse_replay:
        result["replay"] = build_tournament_replay()
    result["model"] = train_tournament_adapter()
    result["report"] = str(write_tournament_report())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
