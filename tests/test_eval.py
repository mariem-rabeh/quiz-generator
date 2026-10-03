"""Tests des métriques d'évaluation (aucun chiffre inventé hors fixture locale)."""

from __future__ import annotations

from src.chunk import Chunk
from src.schemas import Flashcard, MCQ, QuizOutput
from eval.eval_questions import evaluate_quiz


def test_evaluate_quiz_grounding_and_citations() -> None:
    chunks = [
        Chunk(
            id="1",
            text="Les chloroplastes absorbent principalement le bleu et le rouge.",
            page=3,
            source="cours.pdf",
        )
    ]
    quiz = QuizOutput(
        mcqs=[
            MCQ(
                question="Quelle organelle absorbe la lumière ?",
                options=["Noyau", "Chloroplaste", "Lysosome", "Ribosome"],
                correct_index=1,
                explanation="Les chloroplastes absorbent le bleu et le rouge.",
                source_page=3,
                source="cours.pdf",
            )
        ],
        flashcards=[
            Flashcard(
                front="Chloroplaste",
                back="Absorbe le bleu et le rouge",
                source_page=3,
                source="cours.pdf",
            )
        ],
    )
    metrics = evaluate_quiz(quiz, chunks)
    assert metrics["n_mcq"] == 1
    assert metrics["mcq_four_options"] is True
    assert metrics["mcq_citation_rate"] == 1.0
    assert metrics["mcq_grounding_rate"] == 1.0
    assert metrics["flashcard_citation_rate"] == 1.0
