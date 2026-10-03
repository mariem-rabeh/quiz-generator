"""Révision espacée Leitner : intervalles 0 / 1 / 3 / 7 / 14 jours."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from src.schemas import Flashcard

DEFAULT_PROGRESS_PATH = Path("data") / "progress.json"
INTERVALS_DAYS = (0, 1, 3, 7, 14)
MAX_BOX = len(INTERVALS_DAYS) - 1


@dataclass
class ProgressEntry:
    """État Leitner d'une flashcard."""

    card_id: str
    front: str
    back: str
    source: str
    source_page: int
    box: int
    due: str
    correct: int = 0
    incorrect: int = 0


def card_id_for(card: Flashcard) -> str:
    return f"{card.source}::p{card.source_page}::{card.front.strip().lower()}"


def load_progress(path: str | Path = DEFAULT_PROGRESS_PATH) -> dict[str, ProgressEntry]:
    """Charge data/progress.json (dict vide si absent)."""
    progress_path = Path(path)
    if not progress_path.is_file():
        return {}
    raw = json.loads(progress_path.read_text(encoding="utf-8"))
    cards = raw.get("cards", raw)
    entries: dict[str, ProgressEntry] = {}
    for key, value in cards.items():
        entries[key] = ProgressEntry(**value)
    return entries


def save_progress(
    entries: dict[str, ProgressEntry],
    path: str | Path = DEFAULT_PROGRESS_PATH,
) -> None:
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "cards": {key: asdict(entry) for key, entry in entries.items()}
    }
    progress_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_flashcards(
    cards: list[Flashcard],
    *,
    path: str | Path = DEFAULT_PROGRESS_PATH,
    today: date | None = None,
) -> int:
    """Ajoute les nouvelles cartes en boîte 0, dues aujourd'hui. Retourne le nombre créé."""
    today = today or date.today()
    entries = load_progress(path)
    created = 0
    for card in cards:
        identifier = card_id_for(card)
        if identifier in entries:
            continue
        entries[identifier] = ProgressEntry(
            card_id=identifier,
            front=card.front,
            back=card.back,
            source=card.source,
            source_page=card.source_page,
            box=0,
            due=today.isoformat(),
        )
        created += 1
    save_progress(entries, path)
    return created


def due_cards(
    *,
    path: str | Path = DEFAULT_PROGRESS_PATH,
    today: date | None = None,
) -> list[ProgressEntry]:
    today = today or date.today()
    entries = load_progress(path)
    due: list[ProgressEntry] = []
    for entry in entries.values():
        if date.fromisoformat(entry.due) <= today:
            due.append(entry)
    return sorted(due, key=lambda item: (item.box, item.due, item.card_id))


def record_review(
    card_id: str,
    remembered: bool,
    *,
    path: str | Path = DEFAULT_PROGRESS_PATH,
    today: date | None = None,
) -> ProgressEntry:
    """Succès : boîte +1 ; échec : retour boîte 0. Met à jour la date due."""
    today = today or date.today()
    entries = load_progress(path)
    if card_id not in entries:
        raise KeyError(f"Carte inconnue : {card_id}")
    entry = entries[card_id]
    if remembered:
        entry.box = min(entry.box + 1, MAX_BOX)
        entry.correct += 1
    else:
        entry.box = 0
        entry.incorrect += 1
    entry.due = (today + timedelta(days=INTERVALS_DAYS[entry.box])).isoformat()
    entries[card_id] = entry
    save_progress(entries, path)
    return entry
