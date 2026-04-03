from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reviewai.app import create_app
from reviewai.config import Settings
from reviewai.services.analyzer import ReviewAnalysisService

TEST_MODEL_ROOT = Path("/tmp/reviewAI-model")


class FakeAdapter:
    def __init__(self):
        self.loaded = False

    def analyze_review(self, review_text: str) -> dict:
        self.loaded = True
        return self._prediction(review_text)

    def analyze_reviews(self, review_texts: list[str]) -> list[dict]:
        self.loaded = True
        return [self._prediction(text) for text in review_texts]

    def get_model_version(self) -> str:
        return "fake-two-stage"

    def is_loaded(self) -> bool:
        return self.loaded

    def _prediction(self, review_text: str) -> dict:
        return {
            "model_version": "fake-two-stage",
            "text": review_text,
            "semeval_raw_outputs": {
                "food": {
                    "present": True,
                    "polarity": "positive",
                    "confidence": 0.91,
                    "final_probabilities": {
                        "none": 0.04,
                        "positive": 0.91,
                        "negative": 0.03,
                        "neutral": 0.01,
                        "conflict": 0.01,
                    },
                    "stage1": {
                        "label": "present",
                        "probabilities": {"none": 0.05, "present": 0.95},
                    },
                    "stage2": {
                        "label": "positive",
                        "probabilities": {
                            "positive": 0.96,
                            "negative": 0.02,
                            "neutral": 0.01,
                            "conflict": 0.01,
                        },
                    },
                },
                "service": {
                    "present": True,
                    "polarity": "negative",
                    "confidence": 0.78,
                    "final_probabilities": {
                        "none": 0.12,
                        "positive": 0.05,
                        "negative": 0.78,
                        "neutral": 0.03,
                        "conflict": 0.02,
                    },
                    "stage1": {
                        "label": "present",
                        "probabilities": {"none": 0.12, "present": 0.88},
                    },
                    "stage2": {
                        "label": "negative",
                        "probabilities": {
                            "positive": 0.06,
                            "negative": 0.89,
                            "neutral": 0.03,
                            "conflict": 0.02,
                        },
                    },
                },
                "price": {
                    "present": False,
                    "polarity": "none",
                    "confidence": 0.93,
                    "final_probabilities": {
                        "none": 0.93,
                        "positive": 0.03,
                        "negative": 0.02,
                        "neutral": 0.01,
                        "conflict": 0.01,
                    },
                    "stage1": {
                        "label": "none",
                        "probabilities": {"none": 0.93, "present": 0.07},
                    },
                    "stage2": {"label": None, "probabilities": None},
                },
                "ambience": {
                    "present": False,
                    "polarity": "none",
                    "confidence": 0.87,
                    "final_probabilities": {
                        "none": 0.87,
                        "positive": 0.05,
                        "negative": 0.03,
                        "neutral": 0.03,
                        "conflict": 0.02,
                    },
                    "stage1": {
                        "label": "none",
                        "probabilities": {"none": 0.87, "present": 0.13},
                    },
                    "stage2": {"label": None, "probabilities": None},
                },
                "anecdotes/miscellaneous": {
                    "present": False,
                    "polarity": "none",
                    "confidence": 0.95,
                    "final_probabilities": {
                        "none": 0.95,
                        "positive": 0.02,
                        "negative": 0.01,
                        "neutral": 0.01,
                        "conflict": 0.01,
                    },
                    "stage1": {
                        "label": "none",
                        "probabilities": {"none": 0.95, "present": 0.05},
                    },
                    "stage2": {"label": None, "probabilities": None},
                },
            },
            "aspect_outputs": {
                "food": {
                    "present": True,
                    "polarity": "positive",
                    "confidence": 0.91,
                    "rating_suggested": 4.61,
                    "source_category": "food",
                },
                "ambiance": {
                    "present": False,
                    "polarity": "none",
                    "confidence": 0.87,
                    "rating_suggested": None,
                    "source_category": "ambience",
                },
                "service": {
                    "present": True,
                    "polarity": "negative",
                    "confidence": 0.78,
                    "rating_suggested": 1.77,
                    "source_category": "service",
                },
                "value": {
                    "present": False,
                    "polarity": "none",
                    "confidence": 0.93,
                    "rating_suggested": None,
                    "source_category": "price",
                },
            },
        }


@pytest.fixture
def fake_settings() -> Settings:
    return Settings(
        app_name="reviewAI",
        host="127.0.0.1",
        port=8008,
        log_level="INFO",
        device="cpu",
        batch_size=8,
        model_root=TEST_MODEL_ROOT,
        manifest_path=TEST_MODEL_ROOT / "artifacts" / "final_model_manifest.json",
        allowed_origins=("*",),
        eager_load_model=False,
    )


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def client(fake_settings: Settings, fake_adapter: FakeAdapter) -> TestClient:
    service = ReviewAnalysisService(settings=fake_settings, adapter=fake_adapter)
    app = create_app(settings=fake_settings, analysis_service=service)
    return TestClient(app)
