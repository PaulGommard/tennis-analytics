from __future__ import annotations

import argparse
import json
from pathlib import Path

from tennis_analytics.ingestion.manifest import load_manifest
from tennis_analytics.paths import MANIFEST_PATH, PROJECT_ROOT
from tennis_analytics.utils.files import sha256_file


def verify_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    failures: list[dict[str, str]] = []
    for record in manifest["files"]:
        path = Path(record["relative_path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        if path.stat().st_size != record.get("bytes"):
            failures.append({"path": str(path), "reason": "size mismatch"})
            continue
        if sha256_file(path) != record.get("sha256"):
            failures.append({"path": str(path), "reason": "SHA-256 mismatch"})
    return {
        "manifest": str(manifest_path),
        "checked": len(manifest["files"]),
        "failures": failures,
        "valid": not failures,
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Verify every manifest checksum")
    command.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = verify_manifest(args.manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
