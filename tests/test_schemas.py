"""Validation Pydantic des QCM (4 options) et des flashcards."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import Flashcard, MCQ, QuizOutput


def _mcq(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "question": "Quelle organelle réalise la photosynthèse ?",
        "options": ["Mitochondrie", "Chloroplaste", "Noyau", "Ribosome"],
        "correct_index": 1,
        "explanation": "Les chloroplastes absorbent la lumière.",
        "source_page": 3,
        "source": "cours.pdf",
    }
    data.update(overrides)
    return data


def test_mcq_accepts_exactly_four_options() -> None:
    item = MCQ.model_validate(_mcq())
    assert len(item.options) == 4
    assert item.correct_index == 1


def test_mcq_rejects_three_options() -> None:
    with pytest.raises(ValidationError, match="exactement 4"):
        MCQ.model_validate(_mcq(options=["A", "B", "C"]))


def test_mcq_rejects_five_options() -> None:
    with pytest.raises(ValidationError, match="exactement 4"):
        MCQ.model_validate(_mcq(options=["A", "B", "C", "D", "E"]))


def test_mcq_rejects_empty_option() -> None:
    with pytest.raises(ValidationError):
        MCQ.model_validate(_mcq(options=["A", "B", "C", "  "]))


def test_mcq_rejects_invalid_correct_index() -> None:
    with pytest.raises(ValidationError, match="correct_index"):
        MCQ.model_validate(_mcq(correct_index=4))


def test_flashcard_strips_whitespace() -> None:
    card = Flashcard.model_validate(
        {
            "front": "  ATP  ",
            "back": "  énergie  ",
            "source_page": 1,
            "source": " bio.pdf ",
        }
    )
    assert card.front == "ATP"
    assert card.back == "énergie"
    assert card.source == "bio.pdf"


def test_quiz_output_allows_fewer_items_than_requested() -> None:
    quiz = QuizOutput.model_validate({"mcqs": [_mcq()], "flashcards": []})
    assert len(quiz.mcqs) == 1
    assert quiz.flashcards == []


def test_quiz_output_defaults_to_empty_lists() -> None:
    quiz = QuizOutput.model_validate({})
    assert quiz.mcqs == []
    assert quiz.flashcards == []
