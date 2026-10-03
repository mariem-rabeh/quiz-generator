"""Tests du benchmark de génération (retrieval et LLM factices, aucun appel API)."""

from __future__ import annotations

import pytest

from eval.eval_questions import run_benchmark
from src.chunk import Chunk
from src.generate import FatalApiError, TransientApiError

CHUNKS = [
    Chunk(
        id="1",
        text="Les chloroplastes absorbent principalement le bleu et le rouge.",
        page=3,
        source="cours.pdf",
    )
]

VALID_JSON = """
{
  "mcqs": [
    {
      "question": "Quelle organelle absorbe la lumière ?",
      "options": ["Noyau", "Chloroplaste", "Lysosome", "Ribosome"],
      "correct_index": 1,
      "explanation": "Les chloroplastes absorbent le bleu et le rouge.",
      "source_page": 3,
      "source": "cours.pdf"
    }
  ],
  "flashcards": []
}
"""


def _retrieve(topic: str, **_kwargs: object) -> list[Chunk]:
    return CHUNKS


def test_benchmark_counts_valid_json_first_try() -> None:
    answers = iter([VALID_JSON, "{bad", VALID_JSON, VALID_JSON])
    report = run_benchmark(
        ["photosynthèse"], 4, retrieve_fn=_retrieve, chat_fn=lambda _p: next(answers)
    )
    assert report["runs"] == 4
    assert report["answered"] == 4
    assert report["valid_json_first_try"] == 3
    assert report["valid_json_first_try_rate"] == 0.75
    assert report["mean_mcq_citation_rate"] == 1.0
    assert report["mean_mcq_grounding_rate"] == 1.0
    assert report["mean_flashcard_grounding_rate"] is None  # aucune flashcard générée
    assert len(report["invalid_examples"]) == 1


def test_benchmark_counts_api_errors_separately() -> None:
    answers = iter([TransientApiError("quota"), VALID_JSON])

    def chat(_prompt: str) -> str:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    report = run_benchmark(["x"], 2, retrieve_fn=_retrieve, chat_fn=chat)
    assert report["api_errors"] == 1
    assert report["answered"] == 1
    assert report["valid_json_first_try_rate"] == 1.0


def test_benchmark_rate_is_none_when_nothing_answered() -> None:
    def chat(_prompt: str) -> str:
        raise TransientApiError("quota")

    report = run_benchmark(["x"], 3, retrieve_fn=_retrieve, chat_fn=chat)
    assert report["answered"] == 0
    assert report["valid_json_first_try_rate"] is None


def test_benchmark_stops_on_fatal_error() -> None:
    def chat(_prompt: str) -> str:
        raise FatalApiError("clé refusée")

    with pytest.raises(FatalApiError):
        run_benchmark(["x"], 5, retrieve_fn=_retrieve, chat_fn=chat)


def test_benchmark_rotates_topics() -> None:
    seen: list[str] = []

    def retrieve(topic: str, **_kwargs: object) -> list[Chunk]:
        seen.append(topic)
        return CHUNKS

    run_benchmark(["a", " b "], 5, retrieve_fn=retrieve, chat_fn=lambda _p: VALID_JSON)
    assert seen == ["a", "b", "a", "b", "a"]


def test_benchmark_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        run_benchmark(["x"], 0, retrieve_fn=_retrieve, chat_fn=lambda _p: VALID_JSON)
    with pytest.raises(ValueError):
        run_benchmark(["  "], 3, retrieve_fn=_retrieve, chat_fn=lambda _p: VALID_JSON)
