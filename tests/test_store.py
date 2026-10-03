"""Tests d'indexation Chroma sans doublons (embeddings factices, pas d'API)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunk import Chunk
from src.store import EmptyIndexError, index_chunks, retrieve


def _embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(text)), 0.25, 0.5] for text in texts]


def _chunk(text: str, *, chunk_id: str, page: int = 1) -> Chunk:
    return Chunk(id=chunk_id, text=text, page=page, source="cours.pdf")


def test_index_skips_existing_ids(tmp_path: Path) -> None:
    chunk = _chunk("Les chloroplastes absorbent le bleu et le rouge.", chunk_id="c1")
    first = index_chunks([chunk], persist_path=tmp_path, embed_fn=_embed)
    second = index_chunks([chunk], persist_path=tmp_path, embed_fn=_embed)
    assert first == 1
    assert second == 0


def test_retrieve_returns_page_and_source(tmp_path: Path) -> None:
    chunks = [
        _chunk("La mitochondrie produit l'ATP dans la cellule.", chunk_id="m1", page=2),
        _chunk("Le noyau contient l'ADN génomique des eucaryotes.", chunk_id="n1", page=5),
    ]
    index_chunks(chunks, persist_path=tmp_path, embed_fn=_embed)
    found = retrieve("ATP mitochondrie", k=1, persist_path=tmp_path, embed_fn=_embed)
    assert found
    assert found[0].source == "cours.pdf"
    assert found[0].page in {2, 5}
    assert found[0].text


def test_retrieve_empty_index_raises(tmp_path: Path) -> None:
    with pytest.raises(EmptyIndexError):
        retrieve("quoi", persist_path=tmp_path, embed_fn=_embed)
