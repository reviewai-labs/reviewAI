"""Service orchestration for review inference responses."""

from __future__ import annotations

from reviewai.config import Settings
from reviewai.schemas.api import (
    AnalyzeReviewRequest,
    AnalyzeReviewResponse,
    AnalyzeReviewsBatchRequest,
    AnalyzeReviewsBatchResponse,
    HealthResponse,
)
from reviewai.services.model_adapter import InferenceAdapter

ASPECT_OUTPUT_ORDER = ("food", "ambiance", "service", "value")
ASPECT_OUTPUT_KEYS = frozenset(ASPECT_OUTPUT_ORDER)
ASPECT_OUTPUT_FIELDS = frozenset(("present", "polarity", "confidence", "rating_suggested"))


def _is_aspect_outputs_candidate(candidate: object) -> bool:
    if not isinstance(candidate, dict) or set(candidate) != ASPECT_OUTPUT_KEYS:
        return False

    return all(
        isinstance(output, dict) and ASPECT_OUTPUT_FIELDS.issubset(output)
        for output in candidate.values()
    )


def extract_aspect_outputs(raw_prediction: dict) -> dict:
    direct_outputs = raw_prediction.get("aspect_outputs")
    if _is_aspect_outputs_candidate(direct_outputs):
        return direct_outputs

    for value in raw_prediction.values():
        if _is_aspect_outputs_candidate(value):
            return value

    raise ValueError("Model prediction did not include recognizable aspect outputs.")


class ReviewAnalysisService:
    def __init__(self, settings: Settings, adapter: InferenceAdapter):
        self.settings = settings
        self.adapter = adapter

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="reviewAI",
            model_version=self.adapter.get_model_version(),
            model_root=str(self.settings.model_root),
            predictor_loaded=self.adapter.is_loaded(),
        )

    def analyze_review(self, payload: AnalyzeReviewRequest) -> AnalyzeReviewResponse:
        raw_prediction = self.adapter.analyze_review(payload.review_text)
        return self._build_review_response(
            raw_prediction=raw_prediction,
            include_raw_outputs=payload.include_raw_outputs,
        )

    def analyze_reviews(self, payload: AnalyzeReviewsBatchRequest) -> AnalyzeReviewsBatchResponse:
        raw_predictions = self.adapter.analyze_reviews(
            [review.review_text for review in payload.reviews]
        )
        results = [
            self._build_review_response(
                raw_prediction=raw_prediction,
                include_raw_outputs=payload.include_raw_outputs,
            )
            for raw_prediction in raw_predictions
        ]
        return AnalyzeReviewsBatchResponse(results=results)

    def _build_review_response(
        self,
        *,
        raw_prediction: dict,
        include_raw_outputs: bool,
    ) -> AnalyzeReviewResponse:
        source_outputs = extract_aspect_outputs(raw_prediction)
        aspect_outputs = {}
        for aspect in ASPECT_OUTPUT_ORDER:
            source_output = source_outputs[aspect]
            aspect_outputs[aspect] = {
                "present": source_output["present"],
                "polarity": source_output["polarity"],
                "confidence": source_output["confidence"],
                "rating_suggested": source_output["rating_suggested"],
            }

        return AnalyzeReviewResponse(
            model_version=raw_prediction["model_version"],
            analysis_language=None,
            semeval_raw_outputs=(
                raw_prediction["semeval_raw_outputs"] if include_raw_outputs else None
            ),
            aspect_outputs=aspect_outputs,
        )
