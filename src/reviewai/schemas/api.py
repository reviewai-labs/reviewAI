"""API request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Polarity = Literal["none", "positive", "negative", "neutral", "conflict"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AnalyzeReviewRequest(StrictModel):
    review_text: str = Field(min_length=1, max_length=5000)
    language_hint: str | None = Field(default=None, min_length=2, max_length=32)
    include_raw_outputs: bool = True

    @field_validator("review_text", mode="before")
    @classmethod
    def strip_review_text(cls, value: str) -> str:
        stripped = str(value).strip()
        if not stripped:
            raise ValueError("review_text must not be blank")
        return stripped

    @field_validator("language_hint", mode="before")
    @classmethod
    def normalize_language_hint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None


class BatchReviewItem(StrictModel):
    review_text: str = Field(min_length=1, max_length=5000)
    language_hint: str | None = Field(default=None, min_length=2, max_length=32)

    @field_validator("review_text", mode="before")
    @classmethod
    def strip_review_text(cls, value: str) -> str:
        stripped = str(value).strip()
        if not stripped:
            raise ValueError("review_text must not be blank")
        return stripped

    @field_validator("language_hint", mode="before")
    @classmethod
    def normalize_language_hint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None


class AnalyzeReviewsBatchRequest(StrictModel):
    reviews: list[BatchReviewItem] = Field(min_length=1, max_length=32)
    include_raw_outputs: bool = True


class BinaryProbabilities(StrictModel):
    none: float = Field(ge=0.0, le=1.0)
    present: float = Field(ge=0.0, le=1.0)


class PolarityProbabilities(StrictModel):
    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    conflict: float = Field(ge=0.0, le=1.0)


class FinalProbabilities(StrictModel):
    none: float = Field(ge=0.0, le=1.0)
    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    conflict: float = Field(ge=0.0, le=1.0)


class RawStage1Output(StrictModel):
    label: Literal["none", "present"]
    probabilities: BinaryProbabilities


class RawStage2Output(StrictModel):
    label: Literal["positive", "negative", "neutral", "conflict"] | None
    probabilities: PolarityProbabilities | None


class SemEvalAspectOutput(StrictModel):
    present: bool
    polarity: Polarity
    confidence: float = Field(ge=0.0, le=1.0)
    final_probabilities: FinalProbabilities
    stage1: RawStage1Output
    stage2: RawStage2Output


class SemEvalRawOutputs(StrictModel):
    food: SemEvalAspectOutput
    service: SemEvalAspectOutput
    price: SemEvalAspectOutput
    ambience: SemEvalAspectOutput
    anecdotes_miscellaneous: SemEvalAspectOutput = Field(alias="anecdotes/miscellaneous")


class AspectOutput(StrictModel):
    present: bool
    polarity: Polarity
    confidence: float = Field(ge=0.0, le=1.0)
    rating_suggested: float | None = Field(default=None, ge=1.0, le=5.0)

    @model_validator(mode="after")
    def validate_rating_suggested(self) -> "AspectOutput":
        if self.present and self.rating_suggested is None:
            raise ValueError("rating_suggested must be numeric when the aspect is present")
        if not self.present and self.rating_suggested is not None:
            raise ValueError("rating_suggested must be null when the aspect is absent")
        return self


class AspectOutputs(StrictModel):
    food: AspectOutput
    ambiance: AspectOutput
    service: AspectOutput
    value: AspectOutput


class AnalyzeReviewResponse(StrictModel):
    model_version: str
    analysis_language: str | None = None
    semeval_raw_outputs: SemEvalRawOutputs | None = None
    aspect_outputs: AspectOutputs = Field(
        description="Primary public aspect summary for client integrations.",
    )


class AnalyzeReviewsBatchResponse(StrictModel):
    results: list[AnalyzeReviewResponse]


class HealthResponse(StrictModel):
    status: Literal["ok"]
    service: Literal["reviewAI"]
    model_version: str
    model_root: str
    predictor_loaded: bool
