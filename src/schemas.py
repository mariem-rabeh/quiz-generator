"""Schémas Pydantic du quiz : QCM (4 options), flashcard, sortie globale."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class MCQ(BaseModel):
    """Question à choix multiples : exactement 4 options, une seule correcte."""

    question: str = Field(min_length=1)
    options: list[str]
    correct_index: int
    explanation: str = Field(min_length=1)
    source_page: int = Field(ge=1)
    source: str = Field(min_length=1)

    @field_validator("question", "explanation", "source", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("options")
    @classmethod
    def _exactly_four_nonempty_options(cls, options: list[str]) -> list[str]:
        cleaned = [opt.strip() for opt in options]
        if len(cleaned) != 4:
            raise ValueError("Un QCM doit avoir exactement 4 options.")
        if any(not opt for opt in cleaned):
            raise ValueError("Chaque option de QCM doit être non vide.")
        return cleaned

    @model_validator(mode="after")
    def _correct_index_in_range(self) -> MCQ:
        if not 0 <= self.correct_index <= 3:
            raise ValueError("correct_index doit être compris entre 0 et 3.")
        return self


class Flashcard(BaseModel):
    """Carte recto/verso ancrée à une page source."""

    front: str = Field(min_length=1)
    back: str = Field(min_length=1)
    source_page: int = Field(ge=1)
    source: str = Field(min_length=1)

    @field_validator("front", "back", "source", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class QuizOutput(BaseModel):
    """Ensemble de QCM et de flashcards produits à partir du retrieval."""

    mcqs: list[MCQ] = Field(default_factory=list)
    flashcards: list[Flashcard] = Field(default_factory=list)
