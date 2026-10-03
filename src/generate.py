"""Génération de QCM et flashcards : prompt fixe, JSON validé, 2 retries."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Sequence

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import ValidationError

from src.chunk import Chunk
from src.schemas import QuizOutput
from src.store import DEFAULT_TOP_K, openai_client, retrieve

ChatFn = Callable[[str], str]

PROMPT_TEMPLATE = """Tu es un générateur de quiz pédagogique.

Règles STRICTES :
- Utilise UNIQUEMENT les extraits de cours fournis ci-dessous.
- N'invente aucun fait, chiffre, définition ou exemple absent des extraits.
- Si l'information est insuffisante, génère MOINS d'éléments (listes éventuellement vides) plutôt que d'inventer.
- Chaque QCM a exactement 4 options, une seule bonne réponse (correct_index entre 0 et 3).
- Chaque item cite source_page et source d'un extrait qui le justifie.
- Réponds UNIQUEMENT en JSON valide, sans markdown, sans commentaire.

Format JSON :
{{
  "mcqs": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "...",
      "source_page": 1,
      "source": "cours.pdf"
    }}
  ],
  "flashcards": [
    {{
      "front": "...",
      "back": "...",
      "source_page": 1,
      "source": "cours.pdf"
    }}
  ]
}}

Sujet demandé : {topic}
Nombre maximum de QCM : {n_mcq}
Nombre maximum de flashcards : {n_flash}

Extraits :
{context}
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2.0  # multiplié par le numéro d'essai pour les erreurs réseau


class QuizGenerationError(RuntimeError):
    """Échec de génération ou JSON invalide après retries."""


class TransientApiError(QuizGenerationError):
    """Erreur OpenAI temporaire (quota, timeout, réseau, serveur) : on peut réessayer."""


class FatalApiError(QuizGenerationError):
    """Erreur OpenAI non récupérable (clé invalide, requête refusée) : inutile de réessayer."""


def _sleep(seconds: float) -> None:
    """Indirection pour pouvoir neutraliser l'attente dans les tests."""
    time.sleep(seconds)


def chat_model() -> str:
    """Modèle de chat : LLM_MODEL (tout fournisseur), sinon OPENAI_CHAT_MODEL."""
    load_dotenv()
    return os.getenv("LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def json_mode_enabled() -> bool:
    """Mettre LLM_JSON_MODE=0 si le fournisseur refuse ``response_format=json_object``."""
    load_dotenv()
    return os.getenv("LLM_JSON_MODE", "1").strip().lower() not in {"0", "false", "no"}


def build_context(chunks: Sequence[Chunk]) -> str:
    """Assemble le contexte retrieval, un bloc par chunk."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Extrait {index} | source={chunk.source} | page={chunk.page}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def parse_quiz_json(raw: str) -> QuizOutput:
    """Parse et valide le JSON modèle ; retire un éventuel fence markdown."""
    payload = raw.strip()
    fenced = _JSON_FENCE_RE.search(payload)
    if fenced:
        payload = fenced.group(1).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise QuizGenerationError(f"JSON invalide : {exc}") from exc
    try:
        return QuizOutput.model_validate(data)
    except ValidationError as exc:
        raise QuizGenerationError(f"JSON non conforme au schéma : {exc}") from exc


def build_prompt(
    topic: str,
    chunks: Sequence[Chunk],
    *,
    n_mcq: int = 5,
    n_flashcards: int = 5,
) -> str:
    """Remplit le template fixe : seul le contexte (issu du retrieval) change."""
    return PROMPT_TEMPLATE.format(
        topic=topic.strip() or "(contenu du document)",
        n_mcq=n_mcq,
        n_flash=n_flashcards,
        context=build_context(chunks),
    )


def generate_quiz(
    topic: str,
    *,
    n_mcq: int = 5,
    n_flashcards: int = 5,
    k: int = DEFAULT_TOP_K,
    persist_path: str | None = None,
    collection_name: str | None = None,
    source: str | None = None,
    chat_fn: ChatFn | None = None,
    retrieved: Sequence[Chunk] | None = None,
) -> QuizOutput:
    """Retrieval → prompt fixe → JSON validé (1 essai + 2 retries).

    Args:
        source: nom du PDF à interroger ; évite de mélanger plusieurs cours indexés.

    Raises:
        FatalApiError: clé invalide ou requête refusée (aucun retry).
        QuizGenerationError: JSON invalide ou service indisponible après les retries.
    """
    if n_mcq < 0 or n_flashcards < 0:
        raise ValueError("n_mcq et n_flashcards doivent être >= 0")

    chunks = list(retrieved) if retrieved is not None else retrieve(
        topic,
        k=k,
        persist_path=persist_path,
        collection_name=collection_name or "quiz_chunks",
        source=source,
    )
    if not chunks:
        raise QuizGenerationError(
            "Aucun extrait récupéré : indexez un PDF texte avant de générer."
        )

    prompt = build_prompt(topic, chunks, n_mcq=n_mcq, n_flashcards=n_flashcards)
    complete = chat_fn or chat_completion
    last_error: QuizGenerationError | None = None
    raw = ""
    for attempt in range(MAX_RETRIES + 1):
        if attempt == 0 or isinstance(last_error, TransientApiError):
            user_prompt = prompt
        else:
            user_prompt = (
                prompt
                + "\n\nLe JSON précédent était invalide. "
                "Renvoie UNIQUEMENT un objet JSON conforme."
            )
        try:
            raw = complete(user_prompt)
            quiz = parse_quiz_json(raw)
            return _trim_quiz(quiz, n_mcq=n_mcq, n_flashcards=n_flashcards)
        except FatalApiError:
            raise
        except TransientApiError as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                _sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        except QuizGenerationError as exc:
            last_error = exc

    if isinstance(last_error, TransientApiError):
        raise TransientApiError(
            f"Service OpenAI indisponible après {MAX_RETRIES} retries : {last_error}"
        ) from last_error
    raise QuizGenerationError(
        f"JSON invalide après {MAX_RETRIES} retries. Dernier extrait : {raw[:400]!r}"
    ) from last_error


def _trim_quiz(quiz: QuizOutput, *, n_mcq: int, n_flashcards: int) -> QuizOutput:
    return QuizOutput(
        mcqs=list(quiz.mcqs[:n_mcq]),
        flashcards=list(quiz.flashcards[:n_flashcards]),
    )


def chat_completion(prompt: str) -> str:
    """Appelle le LLM et traduit les erreurs OpenAI en erreurs du projet.

    Raises:
        TransientApiError: quota, timeout, réseau ou erreur serveur (réessayable).
        FatalApiError: clé invalide ou requête refusée (non réessayable).
    """
    client: OpenAI = openai_client()
    options: dict[str, object] = {}
    if json_mode_enabled():
        options["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(
            model=chat_model(),
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )
    except AuthenticationError as exc:
        raise FatalApiError(
            "Clé OpenAI refusée : vérifiez OPENAI_API_KEY dans .env."
        ) from exc
    except RateLimitError as exc:
        raise TransientApiError(
            "Quota ou limite de débit OpenAI atteint : réessayez dans un instant."
        ) from exc
    except (APIConnectionError, InternalServerError) as exc:
        raise TransientApiError(f"OpenAI injoignable ou en erreur : {exc}") from exc
    except APIStatusError as exc:
        raise FatalApiError(f"Requête refusée par OpenAI ({exc.status_code}).") from exc
    except OpenAIError as exc:
        raise FatalApiError(f"Erreur OpenAI : {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise QuizGenerationError("Réponse modèle vide.")
    return content


_chat_completion = chat_completion  # alias de compatibilité
