"""Tests du filtre par source et de la gestion des erreurs OpenAI (sans appel réseau)."""

from __future__ import annotations

import httpx
import openai
import pytest

from src import generate
from src.chunk import Chunk
from src.generate import (
    FatalApiError,
    QuizGenerationError,
    TransientApiError,
    chat_completion,
    generate_quiz,
)

VALID_JSON = """
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
  "flashcards": []
}
"""

CHUNKS = [
    Chunk(id="1", text="Les chloroplastes absorbent la lumière.", page=2, source="cours.pdf")
]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Neutralise les attentes entre retries et les enregistre."""
    waits: list[float] = []
    monkeypatch.setattr(generate, "_sleep", waits.append)
    return waits


def test_generate_passes_source_to_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_retrieve(query: str, **kwargs: object) -> list[Chunk]:
        seen.update(kwargs)
        return CHUNKS

    monkeypatch.setattr(generate, "retrieve", fake_retrieve)
    generate_quiz("photosynthèse", source="cours.pdf", chat_fn=lambda _p: VALID_JSON)
    assert seen["source"] == "cours.pdf"


def test_transient_error_is_retried_with_backoff(no_sleep: list[float]) -> None:
    calls = {"n": 0}

    def chat(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientApiError("quota")
        return VALID_JSON

    quiz = generate_quiz("x", retrieved=CHUNKS, chat_fn=chat)
    assert calls["n"] == 3
    assert len(quiz.mcqs) == 1
    assert no_sleep == [2.0, 4.0]


def test_transient_error_exhausted_gives_clear_message() -> None:
    def chat(_prompt: str) -> str:
        raise TransientApiError("quota")

    with pytest.raises(TransientApiError, match="indisponible"):
        generate_quiz("x", retrieved=CHUNKS, chat_fn=chat)


def test_fatal_error_is_not_retried(no_sleep: list[float]) -> None:
    calls = {"n": 0}

    def chat(_prompt: str) -> str:
        calls["n"] += 1
        raise FatalApiError("clé refusée")

    with pytest.raises(FatalApiError):
        generate_quiz("x", retrieved=CHUNKS, chat_fn=chat)
    assert calls["n"] == 1
    assert no_sleep == []


def test_json_hint_only_added_after_invalid_json() -> None:
    prompts: list[str] = []
    answers = iter([TransientApiError("quota"), "{bad", VALID_JSON])

    def chat(prompt: str) -> str:
        prompts.append(prompt)
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    generate_quiz("x", retrieved=CHUNKS, chat_fn=chat)
    hint = "Le JSON précédent était invalide"
    assert hint not in prompts[0]
    assert hint not in prompts[1]  # l'essai précédent était une erreur réseau
    assert hint in prompts[2]  # l'essai précédent était un JSON invalide


def test_json_failure_after_retries_is_reported() -> None:
    with pytest.raises(QuizGenerationError, match="JSON invalide après 2 retries"):
        generate_quiz("x", retrieved=CHUNKS, chat_fn=lambda _p: "{bad")


# --- Traduction des erreurs du SDK OpenAI -------------------------------------------


class _FakeCompletions:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def create(self, **_kwargs: object) -> None:
        raise self._error


class _FakeChat:
    def __init__(self, error: Exception) -> None:
        self.completions = _FakeCompletions(error)


class _FakeClient:
    def __init__(self, error: Exception) -> None:
        self.chat = _FakeChat(error)


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1"))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            openai.AuthenticationError("bad key", response=_response(401), body=None),
            FatalApiError,
        ),
        (
            openai.RateLimitError("quota", response=_response(429), body=None),
            TransientApiError,
        ),
        (
            openai.InternalServerError("oops", response=_response(500), body=None),
            TransientApiError,
        ),
        (
            openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")),
            TransientApiError,
        ),
        (
            openai.BadRequestError("bad", response=_response(400), body=None),
            FatalApiError,
        ),
    ],
)
def test_chat_completion_translates_openai_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: type[Exception]
) -> None:
    monkeypatch.setattr(generate, "openai_client", lambda: _FakeClient(error))
    with pytest.raises(expected):
        chat_completion("prompt")
