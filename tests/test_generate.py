"""Tests de parsing JSON et du prompt fixe (sans appel OpenAI)."""

from __future__ import annotations

import pytest

from src.chunk import Chunk
from src.generate import PROMPT_TEMPLATE, QuizGenerationError, generate_quiz, parse_quiz_json


def _valid_payload() -> str:
    return """
    {
      "mcqs": [
        {
          "question": "Où se déroule la photosynthèse ?",
          "options": ["Mitochondrie", "Chloroplaste", "Noyau", "Vacuole"],
          "correct_index": 1,
          "explanation": "Les chloroplastes absorbent la lumière.",
          "source_page": 2,
          "source": "cours.pdf"
        }
      ],
      "flashcards": [
        {
          "front": "Chloroplaste",
          "back": "Organelle de la photosynthèse",
          "source_page": 2,
          "source": "cours.pdf"
        }
      ]
    }
    """


def test_parse_quiz_json_accepts_fenced_block() -> None:
    quiz = parse_quiz_json("```json\n" + _valid_payload() + "\n```")
    assert len(quiz.mcqs) == 1
    assert quiz.mcqs[0].correct_index == 1


def test_parse_quiz_json_rejects_malformed() -> None:
    with pytest.raises(QuizGenerationError, match="JSON invalide"):
        parse_quiz_json("{not json")


def test_generate_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def chat(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            return "{bad"
        return _valid_payload()

    chunks = [
        Chunk(id="1", text="Les chloroplastes absorbent la lumière.", page=2, source="cours.pdf")
    ]
    quiz = generate_quiz("photosynthèse", retrieved=chunks, chat_fn=chat)
    assert calls["n"] == 3
    assert len(quiz.mcqs) == 1


def test_generate_trims_to_requested_counts() -> None:
    def chat(_prompt: str) -> str:
        return _valid_payload()

    chunks = [Chunk(id="1", text="extrait", page=1, source="cours.pdf")]
    quiz = generate_quiz("x", n_mcq=0, n_flashcards=0, retrieved=chunks, chat_fn=chat)
    assert quiz.mcqs == []
    assert quiz.flashcards == []


def test_prompt_template_is_fixed() -> None:
    assert "UNIQUEMENT les extraits" in PROMPT_TEMPLATE
    assert "N'invente aucun fait" in PROMPT_TEMPLATE
    formatted = PROMPT_TEMPLATE.format(topic="ADN", n_mcq=3, n_flash=4, context="...")
    assert "ADN" in formatted
