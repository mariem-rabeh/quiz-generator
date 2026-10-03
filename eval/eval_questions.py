"""Évaluation d'un QuizOutput par rapport aux extraits retrieval.

Deux modes :

* ``--quiz`` + ``--context`` : évaluation hors ligne de fichiers JSON (aucun appel API).
* ``--benchmark N`` : lance N générations réelles (appels OpenAI, donc payants) et
  mesure le taux de JSON valide au premier essai. Les chiffres affichés sont
  uniquement ceux calculés pendant l'exécution : rien n'est inventé ni mis en cache.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.chunk import Chunk
from src.generate import (
    FatalApiError,
    QuizGenerationError,
    TransientApiError,
    build_prompt,
    chat_completion,
    parse_quiz_json,
)
from src.schemas import QuizOutput
from src.store import DEFAULT_TOP_K, retrieve

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if len(token) > 2}


def _context_blob(chunks: list[Chunk]) -> str:
    return "\n".join(chunk.text for chunk in chunks).lower()


def _page_index(chunks: list[Chunk]) -> set[tuple[str, int]]:
    return {(chunk.source, chunk.page) for chunk in chunks}


def _overlap_ratio(text: str, context: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    present = {token for token in tokens if token in context}
    return len(present) / len(tokens)


def evaluate_quiz(quiz: QuizOutput, chunks: list[Chunk]) -> dict[str, Any]:
    """Métriques déterministes : schéma, citations, recouvrement lexical."""
    context = _context_blob(chunks)
    cited_pages = _page_index(chunks)

    mcq_cited = 0
    mcq_grounded = 0
    for item in quiz.mcqs:
        if (item.source, item.source_page) in cited_pages:
            mcq_cited += 1
        answer = item.options[item.correct_index]
        ratio = _overlap_ratio(f"{answer} {item.explanation}", context)
        if ratio >= 0.3:
            mcq_grounded += 1

    flash_cited = 0
    flash_grounded = 0
    for card in quiz.flashcards:
        if (card.source, card.source_page) in cited_pages:
            flash_cited += 1
        if _overlap_ratio(f"{card.front} {card.back}", context) >= 0.3:
            flash_grounded += 1

    n_mcq = len(quiz.mcqs)
    n_flash = len(quiz.flashcards)
    return {
        "n_mcq": n_mcq,
        "n_flashcards": n_flash,
        "mcq_four_options": all(len(item.options) == 4 for item in quiz.mcqs),
        "mcq_citation_rate": (mcq_cited / n_mcq) if n_mcq else None,
        "mcq_grounding_rate": (mcq_grounded / n_mcq) if n_mcq else None,
        "flashcard_citation_rate": (flash_cited / n_flash) if n_flash else None,
        "flashcard_grounding_rate": (flash_grounded / n_flash) if n_flash else None,
        "n_context_chunks": len(chunks),
    }


def _chunks_from_json(payload: list[dict[str, Any]]) -> list[Chunk]:
    return [
        Chunk(
            id=str(item.get("id") or f"eval-{index}"),
            text=str(item["text"]),
            page=int(item["page"]),
            source=str(item["source"]),
        )
        for index, item in enumerate(payload)
    ]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_benchmark(
    topics: Sequence[str],
    runs: int = 30,
    *,
    n_mcq: int = 5,
    n_flashcards: int = 5,
    k: int = DEFAULT_TOP_K,
    source: str | None = None,
    persist_path: str | None = None,
    retrieve_fn: Callable[..., list[Chunk]] | None = None,
    chat_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Lance ``runs`` générations (un seul essai chacune) et mesure la qualité.

    Le taux de JSON valide au premier essai ne compte que les réponses reçues : une
    erreur réseau ou de quota est comptée à part (``api_errors``), pas comme un JSON
    invalide. Les sujets sont utilisés à tour de rôle.

    Raises:
        ValueError: ``runs`` < 1 ou aucun sujet fourni.
        FatalApiError: clé invalide ou requête refusée (le benchmark s'arrête).
    """
    if runs < 1:
        raise ValueError("runs doit être >= 1")
    cleaned_topics = [topic.strip() for topic in topics if topic.strip()]
    if not cleaned_topics:
        raise ValueError("Fournissez au moins un sujet.")

    fetch = retrieve_fn or retrieve
    complete = chat_fn or chat_completion

    answered = 0
    valid_first_try = 0
    api_errors = 0
    schema_errors: list[str] = []
    grounding: dict[str, list[float]] = {"mcq": [], "flashcard": []}
    citation: dict[str, list[float]] = {"mcq": [], "flashcard": []}

    for index in range(runs):
        topic = cleaned_topics[index % len(cleaned_topics)]
        chunks = fetch(topic, k=k, persist_path=persist_path, source=source)
        prompt = build_prompt(topic, chunks, n_mcq=n_mcq, n_flashcards=n_flashcards)
        try:
            raw = complete(prompt)
        except FatalApiError:
            raise
        except TransientApiError:
            api_errors += 1
            continue
        answered += 1
        try:
            quiz = parse_quiz_json(raw)
        except QuizGenerationError as exc:
            schema_errors.append(str(exc)[:200])
            continue
        valid_first_try += 1
        metrics = evaluate_quiz(quiz, chunks)
        for kind in ("mcq", "flashcard"):
            for bucket, suffix in ((grounding, "grounding_rate"), (citation, "citation_rate")):
                value = metrics[f"{kind}_{suffix}"]
                if value is not None:
                    bucket[kind].append(value)

    return {
        "runs": runs,
        "answered": answered,
        "api_errors": api_errors,
        "valid_json_first_try": valid_first_try,
        "valid_json_first_try_rate": (valid_first_try / answered) if answered else None,
        "mean_mcq_grounding_rate": _mean(grounding["mcq"]),
        "mean_mcq_citation_rate": _mean(citation["mcq"]),
        "mean_flashcard_grounding_rate": _mean(grounding["flashcard"]),
        "mean_flashcard_citation_rate": _mean(citation["flashcard"]),
        "invalid_examples": schema_errors[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Évalue la qualité des quiz : fichiers JSON (hors ligne) ou benchmark de "
            "N générations réelles (appels OpenAI payants)."
        )
    )
    parser.add_argument("--quiz", help="Fichier JSON QuizOutput (mode hors ligne)")
    parser.add_argument("--context", help="Fichier JSON liste de chunks (mode hors ligne)")
    parser.add_argument(
        "--benchmark", type=int, metavar="N", help="Lance N générations réelles (ex. 30)"
    )
    parser.add_argument(
        "--topics",
        help="Sujets séparés par des virgules (obligatoire avec --benchmark)",
    )
    parser.add_argument("--source", help="Nom du PDF indexé à interroger (benchmark)")
    args = parser.parse_args()

    if args.benchmark is not None:
        if not args.topics:
            parser.error("--benchmark nécessite --topics")
        try:
            report = run_benchmark(
                args.topics.split(","), args.benchmark, source=args.source
            )
        except (QuizGenerationError, RuntimeError) as exc:
            parser.exit(1, f"Benchmark interrompu : {exc}\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if not (args.quiz and args.context):
        parser.error("Utilisez --quiz et --context, ou --benchmark N --topics ...")
    quiz = QuizOutput.model_validate(json.loads(Path(args.quiz).read_text(encoding="utf-8")))
    chunks = _chunks_from_json(json.loads(Path(args.context).read_text(encoding="utf-8")))
    metrics = evaluate_quiz(quiz, chunks)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
