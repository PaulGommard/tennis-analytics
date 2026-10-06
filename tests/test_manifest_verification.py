import json

import tennis_analytics.validation.manifest as verifier


def test_manifest_verifier_detects_mutation(tmp_path, monkeypatch):
    artifact = tmp_path / "raw.csv"
    artifact.write_text("a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at": None,
                "files": [
                    {
                        "relative_path": "raw.csv",
                        "bytes": artifact.stat().st_size,
                        "sha256": "incorrect",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(verifier, "PROJECT_ROOT", tmp_path)
    result = verifier.verify_manifest(manifest)
    assert result["valid"] is False
    assert result["failures"][0]["reason"] == "SHA-256 mismatch"
