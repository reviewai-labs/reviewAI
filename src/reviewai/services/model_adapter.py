"""Thin adapter around the configured reviewAI-model inference workspace."""

from __future__ import annotations

from functools import cached_property
import json
from pathlib import Path
import sys
from threading import Lock
from typing import Protocol, Sequence

from reviewai.config import Settings
from reviewai.errors import ConfigurationError, ModelUnavailableError


class InferenceAdapter(Protocol):
    def analyze_review(self, review_text: str) -> dict:
        ...

    def analyze_reviews(self, review_texts: Sequence[str]) -> list[dict]:
        ...

    def get_model_version(self) -> str:
        ...

    def is_loaded(self) -> bool:
        ...


class WorkspaceModelAdapter:
    """Load and cache the predictor from the configured model workspace."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._predictor = None
        self._lock = Lock()

    @cached_property
    def manifest(self) -> dict:
        manifest_path = self.settings.manifest_path
        if not manifest_path.is_file():
            raise ConfigurationError(
                f"Model manifest not found: {manifest_path}",
                details=[{"field": "manifest_path", "message": str(manifest_path)}],
            )
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def is_loaded(self) -> bool:
        return self._predictor is not None

    def get_model_version(self) -> str:
        return str(self.manifest["model_version"])

    def analyze_review(self, review_text: str) -> dict:
        try:
            return self._get_predictor().predict(review_text)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to run model inference.",
                details=[{"message": str(exc)}],
            ) from exc

    def analyze_reviews(self, review_texts: Sequence[str]) -> list[dict]:
        try:
            return self._get_predictor().predict_many(
                review_texts,
                batch_size=self.settings.batch_size,
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to run batch model inference.",
                details=[{"message": str(exc)}],
            ) from exc

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        with self._lock:
            if self._predictor is not None:
                return self._predictor

            predictor_class = self._import_predictor_class()
            self._predictor = predictor_class.from_manifest(
                self.settings.manifest_path,
                device=self.settings.device,
                local_files_only=True,
            )
            return self._predictor

    def _import_predictor_class(self):
        src_root = self.settings.model_root / "src"
        if not src_root.is_dir():
            raise ConfigurationError(
                f"Model source root not found: {src_root}",
                details=[{"field": "model_root", "message": str(self.settings.model_root)}],
            )

        src_root_str = str(src_root)
        if src_root_str not in sys.path:
            sys.path.insert(0, src_root_str)

        try:
            from review_aspect_model.inference import ReviewAIModelWorkspacePredictor
        except Exception as exc:
            raise ModelUnavailableError(
                "Unable to import the configured model predictor.",
                details=[{"message": str(exc)}],
            ) from exc

        return ReviewAIModelWorkspacePredictor
