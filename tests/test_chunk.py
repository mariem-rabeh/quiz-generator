"""Tests du chunking récursif et de l'extraction PDF."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.chunk import TARGET_CHUNK_SIZE, chunk_pages, clean_text, split_text
from src.parse import PageText, PdfParseError, extract_pages


def test_short_text_stays_a_single_chunk() -> None:
    text = "Le théorème de Pythagore relie les côtés d'un triangle rectangle."
    chunks = split_text(text)
    assert chunks == [text]
    assert len(chunks[0]) < TARGET_CHUNK_SIZE


def test_chunk_size_stays_near_target() -> None:
    paragraph = (
        "La photosynthèse convertit la lumière en énergie chimique. "
        "Les chloroplastes absorbent principalement le bleu et le rouge. "
    )
    text = paragraph * 80  # largement au-dessus de 900 caractères
    chunks = split_text(text)
    assert len(chunks) > 1
    # Le dernier chunk peut être plus court ; les autres restent autour de la cible.
    for chunk in chunks[:-1]:
        assert 400 <= len(chunk) <= TARGET_CHUNK_SIZE + 80
    assert all(len(chunk) <= TARGET_CHUNK_SIZE + 80 for chunk in chunks)


def test_overlap_between_consecutive_chunks() -> None:
    text = " ".join(f"mot{i:03d}" for i in range(400))
    chunks = split_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    first_tail = chunks[0][-20:].split()[-1]
    assert first_tail in chunks[1]


def test_recursive_split_long_line_without_paragraphs() -> None:
    text = " ".join(["phrase longue sans saut de ligne."] * 60)
    assert "\n" not in text
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= TARGET_CHUNK_SIZE + 80 for c in chunks)


def test_chunk_pages_keeps_page_and_source() -> None:
    pages = [
        PageText(page=2, text="Alpha " * 20, source="cours.pdf"),
        PageText(page=3, text="Beta " * 300, source="cours.pdf"),
    ]
    chunks = chunk_pages(pages)
    assert chunks
    assert {c.source for c in chunks} == {"cours.pdf"}
    assert {c.page for c in chunks} <= {2, 3}
    assert all(c.id.startswith("cours::p") for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_clean_text_collapses_whitespace() -> None:
    raw = "  Hello   world  \n\n\n  next  "
    assert clean_text(raw) == "Hello world\n\nnext"


def test_empty_text_yields_no_chunks() -> None:
    assert split_text("   \n  ") == []
    assert chunk_pages([PageText(page=1, text="  ", source="x.pdf")]) == []


def test_extract_pages_skips_empty_and_keeps_page_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "lesson.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Page one content about enzymes.")
    doc.new_page()  # vide → ignorée
    doc.new_page().insert_text((72, 72), "Page three content about substrates.")
    doc.save(pdf_path)
    doc.close()

    pages = extract_pages(pdf_path)
    assert [p.page for p in pages] == [1, 3]
    assert pages[0].source == "lesson.pdf"
    assert "enzymes" in pages[0].text
    assert "substrates" in pages[1].text


def test_extract_pages_detects_scanned_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    with pytest.raises(PdfParseError, match="scanné"):
        extract_pages(pdf_path)


def test_extract_pages_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PdfParseError, match="introuvable"):
        extract_pages(tmp_path / "absent.pdf")
