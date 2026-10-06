import json

from tennis_analytics.ingestion.external import slam_artifacts
from tennis_analytics.ingestion.manifest import load_manifest, save_manifest, upsert_file


def test_manifest_upsert_is_deterministic(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = load_manifest(path)
    record = {"source_id": "source", "relative_path": "data/raw/file.csv", "sha256": "abc"}
    upsert_file(manifest, record)
    upsert_file(manifest, {**record, "sha256": "def"})
    save_manifest(manifest, path)
    loaded = json.loads(path.read_text())
    assert len(loaded["files"]) == 1
    assert loaded["files"][0]["sha256"] == "def"


def test_slam_inventory_skips_events_absent_from_archive():
    names = {artifact.remote_name for artifact in slam_artifacts(2020, 2024)}
    assert "2020-wimbledon-points.csv" not in names
    assert "2024-frenchopen-matches.csv" not in names
    assert "2024-usopen-points.csv" in names
