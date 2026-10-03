"""Embeddings (locaux ou API compatible OpenAI) et index Chroma : retrieval top-k.

Le LLM de chat passe par n'importe quelle API compatible OpenAI (OpenAI, Groq, Gemini,
Mistral, Ollama...) en changeant seulement ``LLM_BASE_URL``, ``LLM_API_KEY`` et
``LLM_MODEL`` dans ``.env``. Les embeddings peuvent être calculés localement et
gratuitement (``EMBEDDING_PROVIDER=local``), ce qui évite tout quota d'API.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from src.chunk import Chunk

DEFAULT_COLLECTION = "quiz_chunks"
DEFAULT_PERSIST_DIR = Path("data") / "chroma"
DEFAULT_TOP_K = 5
_EMBED_BATCH_SIZE = 64
_PLACEHOLDER_KEYS = {"", "sk-replace-me", "replace-me"}

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


class MissingApiKeyError(RuntimeError):
    """Clé d'API du LLM absente ou encore égale au placeholder."""


class EmptyIndexError(RuntimeError):
    """Aucun chunk dans la collection Chroma."""


def embedding_provider() -> str:
    """``local`` (gratuit, sur ta machine) ou ``openai`` (API compatible OpenAI)."""
    load_dotenv()
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider not in {"local", "openai"}:
        raise ValueError("EMBEDDING_PROVIDER doit valoir 'local' ou 'openai'.")
    return provider


def embedding_model() -> str:
    """Modèle d'embedding de l'API (provider ``openai``) : le même pour index et requête."""
    load_dotenv()
    return os.getenv("EMBEDDING_MODEL") or os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )


def llm_base_url() -> str | None:
    """URL de l'API compatible OpenAI ; ``None`` = OpenAI officiel."""
    load_dotenv()
    return os.getenv("LLM_BASE_URL", "").strip() or None


def require_api_key() -> str:
    """Charge .env et refuse une clé manquante ou placeholder."""
    load_dotenv()
    key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if key in _PLACEHOLDER_KEYS:
        raise MissingApiKeyError(
            "Clé d'API absente. Copiez .env.example vers .env et renseignez LLM_API_KEY."
        )
    return key


def openai_client() -> OpenAI:
    """Client du SDK OpenAI, pointé sur LLM_BASE_URL si elle est définie."""
    return OpenAI(api_key=require_api_key(), base_url=llm_base_url())


@lru_cache(maxsize=1)
def _local_encoder() -> Callable[[list[str]], list[list[float]]]:
    """Charge une seule fois le modèle d'embedding local.

    Par défaut : le modèle intégré à Chroma (all-MiniLM-L6-v2, téléchargé au premier
    usage). Pour des cours en français ou en arabe, définis ``EMBEDDING_LOCAL_MODEL``
    (ex. ``paraphrase-multilingual-MiniLM-L12-v2``) après ``pip install sentence-transformers``.
    """
    load_dotenv()
    name = os.getenv("EMBEDDING_LOCAL_MODEL", "").strip()
    if name:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name)
        return lambda texts: [[float(x) for x in v] for v in model.encode(texts)]

    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    default = DefaultEmbeddingFunction()
    return lambda texts: [[float(x) for x in v] for v in default(texts)]


def embed_texts(texts: Sequence[str], *, client: OpenAI | None = None) -> list[list[float]]:
    """Embedde une liste de textes avec le même modèle pour l'index et la requête."""
    if not texts:
        return []
    if embedding_provider() == "local":
        encoder = _local_encoder()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            vectors.extend(encoder(list(texts[start : start + _EMBED_BATCH_SIZE])))
        return vectors

    api = client or openai_client()
    model = embedding_model()
    vectors = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = list(texts[start : start + _EMBED_BATCH_SIZE])
        response = api.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
    return vectors


def persist_dir(path: str | Path | None = None) -> Path:
    directory = Path(path) if path else DEFAULT_PERSIST_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_collection(
    *,
    persist_path: str | Path | None = None,
    name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
):
    chroma = client or chromadb.PersistentClient(path=str(persist_dir(persist_path)))
    return chroma.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    persist_path: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
    embed_fn: EmbedFn | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Indexe les chunks dont l'id n'est pas déjà présent. Retourne le nombre ajouté.

    L'indexation se fait par lots de 64 : ``on_progress(fait, total)`` est appelé au
    départ puis après chaque lot, et un lot déjà écrit n'est pas perdu si une erreur
    survient plus tard (les ids existants sont ignorés à la relance).
    """
    if not chunks:
        return 0

    collection = get_collection(
        persist_path=persist_path, name=collection_name, client=client
    )
    # Dédoublonne aussi à l'intérieur de la liste fournie (ids identiques).
    unique: dict[str, Chunk] = {chunk.id: chunk for chunk in chunks}
    existing = set(collection.get(ids=list(unique)).get("ids") or [])
    new_chunks = [chunk for chunk_id, chunk in unique.items() if chunk_id not in existing]
    total = len(new_chunks)
    if on_progress:
        on_progress(0, total)
    if not new_chunks:
        return 0

    if embed_fn is not None:
        encoder = embed_fn
    elif embedding_provider() == "openai":
        api = openai_client()  # un seul client pour tous les lots
        encoder = lambda texts: embed_texts(texts, client=api)  # noqa: E731
    else:
        encoder = lambda texts: embed_texts(texts)  # noqa: E731  (modèle local)

    done = 0
    for start in range(0, total, _EMBED_BATCH_SIZE):
        batch = new_chunks[start : start + _EMBED_BATCH_SIZE]
        embeddings = encoder([chunk.text for chunk in batch])
        collection.add(
            ids=[chunk.id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=embeddings,
            metadatas=[{"page": chunk.page, "source": chunk.source} for chunk in batch],
        )
        done += len(batch)
        if on_progress:
            on_progress(done, total)
    return total


def retrieve(
    query: str,
    *,
    k: int = DEFAULT_TOP_K,
    persist_path: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
    embed_fn: EmbedFn | None = None,
    source: str | None = None,
) -> list[Chunk]:
    """Retourne les k chunks les plus proches, embeddings identiques à l'index."""
    collection = get_collection(
        persist_path=persist_path, name=collection_name, client=client
    )
    total = collection.count()
    if total == 0:
        raise EmptyIndexError("Index vide : indexez d'abord un PDF.")

    encoder = embed_fn or (lambda texts: embed_texts(texts))
    query_embedding = encoder([query])[0]
    kwargs: dict[str, object] = {
        "query_embeddings": [query_embedding],
        "n_results": min(max(k, 1), total),
        "include": ["documents", "metadatas"],
    }
    if source:
        kwargs["where"] = {"source": source}

    result = collection.query(**kwargs)
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    chunks: list[Chunk] = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=False):
        meta = metadata or {}
        text = document or str(meta.get("text") or "")
        chunks.append(
            Chunk(
                id=str(chunk_id),
                text=text,
                page=int(meta.get("page") or 1),
                source=str(meta.get("source") or ""),
            )
        )
    return chunks
