"""Tests de l'indexation par lots et de la progression (embeddings factices)."""

from __future__ import annotations

from pathlib import Path

from src.chunk import Chunk
from src.store import index_chunks


def _chunks(count: int) -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", text=f"extrait numéro {i}", page=1 + i // 10, source="cours.pdf")
        for i in range(count)
    ]


def test_progress_is_reported_after_each_batch(tmp_path: Path) -> None:
    batch_sizes: list[int] = []
    progress: list[tuple[int, int]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        batch_sizes.append(len(texts))
        return [[float(len(text)), 0.1, 0.2] for text in texts]

    added = index_chunks(
        _chunks(130),
        persist_path=tmp_path,
        embed_fn=embed,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert added == 130
    assert batch_sizes == [64, 64, 2]
    assert progress == [(0, 130), (64, 130), (128, 130), (130, 130)]


def test_duplicate_ids_in_input_are_indexed_once(tmp_path: Path) -> None:
    chunk = Chunk(id="same", text="doublon", page=1, source="cours.pdf")
    added = index_chunks(
        [chunk, chunk],
        persist_path=tmp_path,
        embed_fn=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    assert added == 1


def test_already_indexed_chunks_report_zero_progress(tmp_path: Path) -> None:
    embed = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]  # noqa: E731
    index_chunks(_chunks(3), persist_path=tmp_path, embed_fn=embed)

    progress: list[tuple[int, int]] = []
    added = index_chunks(
        _chunks(3),
        persist_path=tmp_path,
        embed_fn=embed,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert added == 0
    assert progress == [(0, 0)]
