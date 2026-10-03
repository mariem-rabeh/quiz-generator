"""Interface Streamlit : indexer un cours PDF, générer un quiz, réviser ses flashcards."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.chunk import chunk_pages
from src.generate import FatalApiError, QuizGenerationError, generate_quiz
from src.parse import PdfParseError, extract_pages
from src.review import add_flashcards, due_cards, record_review
from src.store import (
    EmptyIndexError,
    MissingApiKeyError,
    get_collection,
    index_chunks,
    require_api_key,
)

st.set_page_config(page_title="Quiz Generator", page_icon="📚", layout="centered")
st.title("📚 Générateur de quiz depuis tes cours")

# --- Clé API : message clair avant toute chose -------------------------------------
try:
    require_api_key()
except MissingApiKeyError as exc:
    st.error(str(exc))
    st.stop()


def indexed_sources() -> list[str]:
    """Noms des PDF déjà présents dans l'index."""
    metadatas = get_collection().get(include=["metadatas"]).get("metadatas") or []
    return sorted({str(meta.get("source")) for meta in metadatas if meta and meta.get("source")})


def index_uploaded_pdf(uploaded) -> int:
    """Enregistre le PDF sous son nom d'origine, puis parse, découpe et indexe."""
    folder = Path(tempfile.mkdtemp())
    path = folder / Path(uploaded.name).name
    path.write_bytes(uploaded.getbuffer())

    pages = extract_pages(path)
    chunks = chunk_pages(pages)
    bar = st.progress(0.0, text="Création des embeddings...")

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / total if total else 1.0, text=f"Indexation : {done}/{total}")

    added = index_chunks(chunks, on_progress=on_progress)
    bar.empty()
    return added


tab_quiz, tab_review = st.tabs(["🧠 Générer un quiz", "🔁 Réviser"])

# =====================================================================================
# Onglet 1 : indexation + génération
# =====================================================================================
with tab_quiz:
    st.subheader("1. Dépose ton cours")
    uploaded = st.file_uploader("Cours au format PDF", type="pdf")
    if uploaded and st.button("Indexer ce PDF"):
        try:
            with st.spinner("Lecture et indexation..."):
                added = index_uploaded_pdf(uploaded)
            if added:
                st.success(f"{added} nouveaux morceaux indexés.")
            else:
                st.info("Ce PDF était déjà indexé : rien à ajouter.")
        except PdfParseError as exc:
            st.error(str(exc))
        except (MissingApiKeyError, FatalApiError) as exc:
            st.error(str(exc))

    st.subheader("2. Choisis un sujet")
    sources = indexed_sources()
    if not sources:
        st.info("Aucun cours indexé pour le moment.")
    else:
        source = st.selectbox("Cours à interroger", sources)
        topic = st.text_input("Sujet (ex. : les embeddings)")
        col_mcq, col_cards = st.columns(2)
        n_mcq = col_mcq.slider("Nombre de QCM", 3, 10, 5)
        n_cards = col_cards.slider("Nombre de flashcards", 3, 10, 5)

        if st.button("Générer le quiz", type="primary", disabled=not topic.strip()):
            try:
                with st.spinner("Génération en cours..."):
                    quiz = generate_quiz(
                        topic, n_mcq=n_mcq, n_flashcards=n_cards, source=source
                    )
                st.session_state["quiz"] = quiz
                st.session_state["quiz_id"] = st.session_state.get("quiz_id", 0) + 1
                created = add_flashcards(quiz.flashcards)
                if created:
                    st.toast(f"{created} flashcards ajoutées à la révision.")
            except EmptyIndexError as exc:
                st.error(str(exc))
            except QuizGenerationError as exc:  # inclut FatalApiError / TransientApiError
                st.error(str(exc))

    quiz = st.session_state.get("quiz")
    if quiz:
        quiz_id = st.session_state["quiz_id"]
        answers = {
            index: st.session_state.get(f"q{quiz_id}_{index}")
            for index in range(len(quiz.mcqs))
        }
        answered = [i for i, choice in answers.items() if choice is not None]
        score = sum(
            1 for i in answered
            if quiz.mcqs[i].options.index(answers[i]) == quiz.mcqs[i].correct_index
        )

        st.divider()
        st.header("QCM")
        if not quiz.mcqs:
            st.warning("Le cours ne contient pas assez d'information pour ce sujet.")
        else:
            st.metric("Score de la session", f"{score} / {len(answered)} répondues")
        for index, mcq in enumerate(quiz.mcqs):
            choice = st.radio(
                f"{index + 1}. {mcq.question}",
                mcq.options,
                index=None,
                key=f"q{quiz_id}_{index}",
            )
            if choice is not None:
                if mcq.options.index(choice) == mcq.correct_index:
                    st.success(f"✅ Correct. {mcq.explanation} (page {mcq.source_page})")
                else:
                    good = mcq.options[mcq.correct_index]
                    st.error(
                        f"❌ Bonne réponse : {good}. {mcq.explanation} "
                        f"(page {mcq.source_page})"
                    )

        st.header("Flashcards")
        for card in quiz.flashcards:
            with st.expander(card.front):
                st.write(card.back)
                st.caption(f"{card.source}, page {card.source_page}")

# =====================================================================================
# Onglet 2 : révision espacée (Leitner)
# =====================================================================================
with tab_review:
    due = due_cards()
    if not due:
        st.success("Aucune carte à réviser aujourd'hui. 🎉")
    else:
        card = due[0]
        st.caption(f"{len(due)} carte(s) à réviser · boîte {card.box} · {card.source}, page {card.source_page}")
        st.subheader(card.front)
        shown_key = f"show_{card.card_id}"
        if not st.session_state.get(shown_key):
            if st.button("Voir la réponse"):
                st.session_state[shown_key] = True
                st.rerun()
        else:
            st.info(card.back)
            col_ok, col_ko = st.columns(2)
            if col_ok.button("✅ Je savais"):
                record_review(card.card_id, True)
                st.session_state.pop(shown_key, None)
                st.rerun()
            if col_ko.button("❌ Je ne savais pas"):
                record_review(card.card_id, False)
                st.session_state.pop(shown_key, None)
                st.rerun()
