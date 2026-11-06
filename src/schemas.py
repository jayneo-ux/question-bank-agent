"""Pydantic schemas for `EmergingTechQuizAgent` (`src.schemas`)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AnswerOption(BaseModel):
    """Single multiple-choice answer option (A, B, C, or D)."""

    label: Literal["A", "B", "C", "D"]
    text: str = Field(min_length=1)


class QuizQuestion(BaseModel):
    """Validated 4-option multiple-choice question for `EmergingTechQuizAgent`."""

    question: str = Field(min_length=10)
    options: list[AnswerOption] = Field(min_length=4, max_length=4)
    correct_label: Literal["A", "B", "C", "D"]
    correct_explanation: str = Field(min_length=1)
    source_headline: str = Field(min_length=1)
    source_url: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_options(self) -> QuizQuestion:
        labels = [opt.label for opt in self.options]
        if sorted(labels) != ["A", "B", "C", "D"]:
            raise ValueError("options must include exactly labels A, B, C, and D.")
        if self.correct_label not in labels:
            raise ValueError("correct_label must match one of the option labels.")
        return self
