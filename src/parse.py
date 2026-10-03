"""Extraction du texte PDF, page par page, avec numéro de page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


class PdfParseError(Exception):
    """PDF illisible, vide, ou sans texte extractible."""


@dataclass(frozen=True)
class PageText:
    """Texte d'une page non vide."""

    page: int
    text: str
    source: str


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Lit un PDF et retourne le texte des pages non vides (pages 1-indexées).

    Raises:
        PdfParseError: fichier introuvable, illisible, ou probablement scanné.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise PdfParseError(f"Fichier introuvable : {path}")
    if path.suffix.lower() != ".pdf":
        raise PdfParseError(f"Le fichier n'est pas un PDF : {path.name}")

    try:
        document = fitz.open(path)
    except Exception as exc:  # noqa: BLE001 — PyMuPDF lève des types variés
        raise PdfParseError(f"PDF illisible : {path.name}") from exc

    source = path.name
    pages: list[PageText] = []
    try:
        if document.page_count == 0:
            raise PdfParseError(f"PDF sans pages : {source}")

        for index in range(document.page_count):
            raw = document.load_page(index).get_text("text")
            text = _normalize_extracted_text(raw)
            if not text:
                continue
            pages.append(PageText(page=index + 1, text=text, source=source))
    finally:
        document.close()

    if not pages:
        raise PdfParseError(
            f"PDF probablement scanné (aucun texte extractible) : {source}. "
            "Utilisez un PDF avec une couche texte, ou de l'OCR."
        )
    return pages


def _normalize_extracted_text(raw: str | None) -> str:
    """Supprime les espaces superflus tout en gardant les sauts de paragraphe."""
    if not raw:
        return ""
    lines = [line.strip() for line in raw.replace("\r\n", "\n").split("\n")]
    collapsed = "\n".join(line for line in lines)
    while "\n\n\n" in collapsed:
        collapsed = collapsed.replace("\n\n\n", "\n\n")
    return collapsed.strip()
