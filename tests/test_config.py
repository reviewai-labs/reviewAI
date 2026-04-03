from __future__ import annotations

from reviewai.config import DEFAULT_MODEL_ROOT, Settings


def test_blank_model_root_uses_default_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("REVIEWAI_MODEL_ROOT", raising=False)
    monkeypatch.delenv("REVIEWAI_MANIFEST_PATH", raising=False)

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("REVIEWAI_MODEL_ROOT=\n", encoding="utf-8")

    settings = Settings.from_env(dotenv_path)

    assert settings.model_root == DEFAULT_MODEL_ROOT
    assert settings.manifest_path == DEFAULT_MODEL_ROOT / "artifacts" / "final_model_manifest.json"
