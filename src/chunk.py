"""Nettoyage et découpe récursive des pages (taille ~900, overlap ~10 %)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from src.parse import PageText

TARGET_CHUNK_SIZE = 900
CHUNK_OVERLAP = 90  # ~10 % de 900

# Paragraphe, puis ligne, puis phrase, puis mot.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """Unité indexable : identifiant stable, texte, page, fichier source."""

    id: str
    text: str
    page: int
    source: str


def chunk_pages(
    pages: list[PageText],
    *,
    chunk_size: int = TARGET_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Découpe chaque page indépendamment pour conserver le numéro de page."""
    _validate_sizes(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        parts = split_text(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    id=_chunk_id(page.source, page.page, index, part),
                    text=part,
                    page=page.page,
                    source=page.source,
                )
            )
    return chunks


def clean_text(text: str) -> str:
    """Normalise espaces et césures simples sans changer le fond du cours."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00ad", "")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_text(
    text: str,
    *,
    chunk_size: int = TARGET_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Découpe récursive : paragraphe, ligne, phrase, mot."""
    _validate_sizes(chunk_size, chunk_overlap)
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    pieces = _recursive_split(cleaned, stage=0, chunk_size=chunk_size)
    return _merge_with_overlap(pieces, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _validate_sizes(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être >= 0 et < chunk_size")


def _recursive_split(text: str, stage: int, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    if stage == 0:
        parts = text.split("\n\n")
    elif stage == 1:
        parts = text.split("\n")
    elif stage == 2:
        parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    elif stage == 3:
        parts = text.split(" ")
    else:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    result: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            result.append(part)
        else:
            result.extend(_recursive_split(part, stage + 1, chunk_size))
    return result or [text[:chunk_size]]


def _merge_with_overlap(
    pieces: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current_parts: list[str] = []

    def current_text() -> str:
        return " ".join(current_parts).strip()

    for piece in pieces:
        if not current_parts:
            current_parts = [piece]
            continue
        candidate = f"{current_text()} {piece}"
        if len(candidate) <= chunk_size:
            current_parts.append(piece)
            continue
        chunks.append(current_text())
        overlap = _tail_words(chunks[-1], chunk_overlap)
        current_parts = overlap + [piece]
        if len(" ".join(current_parts)) > chunk_size:
            current_parts = [piece]

    leftover = current_text()
    if leftover:
        chunks.append(leftover)
    return chunks


def _tail_words(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or not text:
        return []
    tail = text[-max_chars:].strip()
    return tail.split() if tail else []


def _chunk_id(source: str, page: int, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    stem = Path(source).stem or "document"
    return f"{stem}::p{page}::c{index:04d}::{digest}"
