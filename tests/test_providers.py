"""Tests de la configuration multi-fournisseurs (aucun appel réseau, aucun téléchargement)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import generate, store
from src.chunk import Chunk

ENV_VARS = [
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "OPENAI_CHAT_MODEL",
    "LLM_JSON_MODE",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODEL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Variables d'environnement vierges ; le fichier .env local est ignoré."""
    monkeypatch.setattr(store, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(generate, "load_dotenv", lambda *a, **k: False)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_api_key_prefers_llm_api_key_and_falls_back_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "old-key")
    assert store.require_api_key() == "old-key"
    monkeypatch.setenv("LLM_API_KEY", "new-key")
    assert store.require_api_key() == "new-key"


@pytest.mark.parametrize("placeholder", ["", "replace-me", "sk-replace-me"])
def test_placeholder_key_is_rejected(monkeypatch: pytest.MonkeyPatch, placeholder: str) -> None:
    monkeypatch.setenv("LLM_API_KEY", placeholder)
    with pytest.raises(store.MissingApiKeyError):
        store.require_api_key()


def test_client_uses_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    assert str(store.openai_client().base_url).startswith("https://api.example.com/v1")


def test_client_defaults_to_official_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "key")
    assert "api.openai.com" in str(store.openai_client().base_url)


def test_chat_model_and_json_mode_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert generate.chat_model() == "gpt-4o-mini"
    monkeypatch.setenv("LLM_MODEL", "my-model")
    assert generate.chat_model() == "my-model"

    assert generate.json_mode_enabled() is True
    monkeypatch.setenv("LLM_JSON_MODE", "0")
    assert generate.json_mode_enabled() is False


class _Recorder:
    """Faux client OpenAI qui mémorise les arguments de chat.completions.create."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.chat = self
        self.completions = self

    def create(self, **kwargs: object):
        self.kwargs = kwargs

        class _Msg:
            content = "{}"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def test_json_mode_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(generate, "openai_client", lambda: recorder)

    generate.chat_completion("prompt")
    assert recorder.kwargs["response_format"] == {"type": "json_object"}

    monkeypatch.setenv("LLM_JSON_MODE", "0")
    generate.chat_completion("prompt")
    assert "response_format" not in recorder.kwargs


def test_embedding_provider_defaults_to_local_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert store.embedding_provider() == "local"
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    assert store.embedding_provider() == "openai"
    monkeypatch.setenv("EMBEDDING_PROVIDER", "nimporte")
    with pytest.raises(ValueError):
        store.embedding_provider()


def test_local_embeddings_need_no_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Avec le provider local, indexer fonctionne sans aucune clé d'API."""
    monkeypatch.setattr(
        store, "_local_encoder", lambda: (lambda texts: [[float(len(t)), 0.1, 0.2] for t in texts])
    )
    chunk = Chunk(id="c1", text="Les chloroplastes absorbent la lumière.", page=1, source="a.pdf")
    assert store.index_chunks([chunk], persist_path=tmp_path) == 1
    found = store.retrieve("chloroplastes", persist_path=tmp_path)
    assert found[0].id == "c1"


def test_embed_texts_batches_local_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[int] = []

    def encoder(texts: list[str]) -> list[list[float]]:
        sizes.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(store, "_local_encoder", lambda: encoder)
    vectors = store.embed_texts([f"t{i}" for i in range(130)])
    assert len(vectors) == 130
    assert sizes == [64, 64, 2]
