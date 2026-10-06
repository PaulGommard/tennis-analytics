from __future__ import annotations

import argparse
import sys

import uvicorn

from tennis_analytics.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tennis Prediction Platform V1 locally")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    # Python 3.14 no longer guarantees that the current working directory is on
    # sys.path when this entry point is reached through a script in scripts/.
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
