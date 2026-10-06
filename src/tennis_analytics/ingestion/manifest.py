from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tennis_analytics.paths import MANIFEST_PATH


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "updated_at": None, "files": []}
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError(f"Unsupported or invalid manifest: {path}")
    return manifest


def save_manifest(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = utc_now()
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def upsert_file(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    key = (record["source_id"], record["relative_path"])
    files = manifest["files"]
    for index, existing in enumerate(files):
        if (existing.get("source_id"), existing.get("relative_path")) == key:
            files[index] = record
            break
    else:
        files.append(record)
    files.sort(key=lambda item: (item["source_id"], item["relative_path"]))
