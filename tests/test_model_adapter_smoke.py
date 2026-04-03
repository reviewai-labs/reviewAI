from __future__ import annotations

import os
from pathlib import Path

import pytest

from reviewai.config import DEFAULT_MODEL_ROOT
from reviewai.config import Settings
from reviewai.services.analyzer import extract_aspect_outputs
from reviewai.services.model_adapter import WorkspaceModelAdapter


def _resolve_smoke_test_paths() -> tuple[Path, Path]:
    raw_model_root = os.getenv("REVIEWAI_MODEL_ROOT", "").strip()
    raw_manifest_path = os.getenv("REVIEWAI_MANIFEST_PATH", "").strip()

    model_root = Path(raw_model_root).expanduser().resolve() if raw_model_root else DEFAULT_MODEL_ROOT
    manifest_path = (
        Path(raw_manifest_path).expanduser().resolve()
        if raw_manifest_path
        else model_root / "artifacts" / "final_model_manifest.json"
    )
    return model_root, manifest_path


def test_workspace_model_adapter_smoke_path():
    model_root, manifest_path = _resolve_smoke_test_paths()
    if not manifest_path.is_file() or not (model_root / "src").is_dir():
        pytest.skip(
            "Set REVIEWAI_MODEL_ROOT or REVIEWAI_MANIFEST_PATH to a local reviewAI-model workspace to run this smoke test."
        )

    settings = Settings(
        app_name="reviewAI",
        host="127.0.0.1",
        port=8008,
        log_level="INFO",
        device="cpu",
        batch_size=4,
        model_root=model_root,
        manifest_path=manifest_path,
        allowed_origins=("*",),
        eager_load_model=False,
    )
    adapter = WorkspaceModelAdapter(settings)

    result = adapter.analyze_review("The sushi was excellent but the service was slow.")

    assert result["model_version"] == "two_stage_roberta_base_seed42"
    assert set(result["semeval_raw_outputs"]) == {
        "food",
        "service",
        "price",
        "ambience",
        "anecdotes/miscellaneous",
    }
    aspect_outputs = extract_aspect_outputs(result)
    assert set(aspect_outputs) == {"food", "ambiance", "service", "value"}
    for output in aspect_outputs.values():
        if output["present"]:
            assert output["rating_suggested"] is not None
            assert 1.0 <= output["rating_suggested"] <= 5.0
        else:
            assert output["rating_suggested"] is None
