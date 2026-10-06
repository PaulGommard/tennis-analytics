from __future__ import annotations

import csv
import logging
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tennis_analytics.ingestion.manifest import (
    load_manifest,
    save_manifest,
    upsert_file,
    utc_now,
)
from tennis_analytics.ingestion.sources import Source
from tennis_analytics.paths import MANIFEST_PATH, PROJECT_ROOT, ensure_data_directories
from tennis_analytics.utils.files import sha256_file

LOGGER = logging.getLogger(__name__)
USER_AGENT = "tennis-analytics-data-pipeline/0.1"


class DownloadError(RuntimeError):
    """A remote artifact could not be downloaded or validated."""


@dataclass(frozen=True)
class Artifact:
    source: Source
    remote_name: str
    destination: Path
    dataset: str
    year: int | None = None
    required_columns: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return f"{self.source.raw_base_url}/{self.remote_name}"


def _validate_artifact(path: Path, artifact: Artifact) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise DownloadError(f"Empty download: {artifact.url}")
    with path.open("rb") as stream:
        prefix = stream.read(512).lower()
    if b"<html" in prefix or b"<!doctype" in prefix:
        raise DownloadError(f"Expected data, received HTML: {artifact.url}")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
            header = next(csv.reader(stream), [])
        missing = set(artifact.required_columns).difference(header)
        if missing:
            raise DownloadError(
                f"CSV {artifact.remote_name} lacks required columns: {sorted(missing)}"
            )


def _record_for(
    artifact: Artifact,
    status: str,
    previous: dict[str, object] | None = None,
) -> dict[str, object]:
    path = artifact.destination
    try:
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative_path = path.as_posix()
    return {
        "source_id": artifact.source.source_id,
        "dataset": artifact.dataset,
        "relative_path": relative_path,
        "original_filename": artifact.remote_name,
        "url": artifact.url,
        "original_source_url": artifact.source.original_url,
        "distribution_url": artifact.source.distribution_url,
        "downloaded_at": (previous or {}).get("downloaded_at", utc_now()),
        "verified_at": utc_now(),
        "year": artifact.year,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "license": {
            "id": artifact.source.license_id,
            "url": artifact.source.license_url,
            "commercial_use": artifact.source.commercial_use,
        },
        "source_snapshot": artifact.source.snapshot,
        "status": status,
    }


def download_artifacts(
    artifacts: list[Artifact],
    *,
    timeout: int = 60,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[int, int]:
    """Download atomically; return ``(downloaded, reused)``.

    Existing raw files are never overwritten. A newer upstream snapshot must use
    a different destination directory so the original remains recoverable.
    """
    ensure_data_directories()
    manifest = load_manifest(manifest_path)
    downloaded = 0
    reused = 0
    failures: list[str] = []

    for artifact in artifacts:
        artifact.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            relative_path = artifact.destination.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            relative_path = artifact.destination.as_posix()
        previous = next(
            (
                item
                for item in manifest["files"]
                if item.get("source_id") == artifact.source.source_id
                and item.get("relative_path") == relative_path
            ),
            None,
        )
        if artifact.destination.exists():
            try:
                _validate_artifact(artifact.destination, artifact)
                current_sha256 = sha256_file(artifact.destination)
                if previous and previous.get("sha256") != current_sha256:
                    raise DownloadError(
                        f"Checksum changed for immutable raw file: {artifact.destination}"
                    )
                upsert_file(manifest, _record_for(artifact, "reused", previous))
                reused += 1
                LOGGER.info("Already present, verified: %s", artifact.destination)
            except (OSError, DownloadError) as exc:
                failures.append(str(exc))
                LOGGER.error("Existing file is invalid (not overwritten): %s", exc)
            continue

        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{artifact.destination.name}.", dir=artifact.destination.parent
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            request = urllib.request.Request(artifact.url, headers={"User-Agent": USER_AGENT})
            LOGGER.info("Downloading %s", artifact.url)
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            _validate_artifact(temporary, artifact)
            if artifact.destination.exists():
                raise DownloadError(
                    f"Destination appeared during download; refusing overwrite: "
                    f"{artifact.destination}"
                )
            temporary.replace(artifact.destination)
            upsert_file(manifest, _record_for(artifact, "downloaded"))
            save_manifest(manifest, manifest_path)
            downloaded += 1
        except (OSError, urllib.error.URLError, DownloadError) as exc:
            failures.append(f"{artifact.url}: {exc}")
            LOGGER.error("Download failed: %s", failures[-1])
        finally:
            temporary.unlink(missing_ok=True)

    save_manifest(manifest, manifest_path)
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise DownloadError(f"{len(failures)} artifact(s) failed:\n{details}")
    return downloaded, reused
