"""Tests Leitner : intervalles 0/1/3/7/14 et persistance JSON."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.review import INTERVALS_DAYS, add_flashcards, due_cards, record_review
from src.schemas import Flashcard


def _card() -> Flashcard:
    return Flashcard(
        front="ATP",
        back="Adénosine triphosphate",
        source_page=1,
        source="cours.pdf",
    )


def test_intervals_match_leitner() -> None:
    assert INTERVALS_DAYS == (0, 1, 3, 7, 14)


def test_new_card_is_due_today(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    today = date(2026, 10, 3)
    created = add_flashcards([_card()], path=path, today=today)
    assert created == 1
    due = due_cards(path=path, today=today)
    assert len(due) == 1
    assert due[0].box == 0
    assert due[0].due == "2026-10-03"


def test_success_advances_box_and_interval(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    today = date(2026, 10, 3)
    add_flashcards([_card()], path=path, today=today)
    entry = due_cards(path=path, today=today)[0]
    updated = record_review(entry.card_id, True, path=path, today=today)
    assert updated.box == 1
    assert updated.due == "2026-10-04"
    assert due_cards(path=path, today=today) == []
    assert len(due_cards(path=path, today=date(2026, 10, 4))) == 1


def test_failure_resets_to_box_zero(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    today = date(2026, 10, 3)
    add_flashcards([_card()], path=path, today=today)
    card_id = due_cards(path=path, today=today)[0].card_id
    record_review(card_id, True, path=path, today=today)
    updated = record_review(card_id, False, path=path, today=date(2026, 10, 4))
    assert updated.box == 0
    assert updated.due == "2026-10-04"


def test_add_flashcards_skips_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    add_flashcards([_card()], path=path, today=date(2026, 10, 3))
    again = add_flashcards([_card()], path=path, today=date(2026, 10, 4))
    assert again == 0


def test_unknown_card_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        record_review("missing", True, path=tmp_path / "progress.json")
