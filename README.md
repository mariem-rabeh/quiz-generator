# 📚 Quiz Generator : QCM et flashcards depuis tes cours (RAG)

Dépose un cours en PDF, choisis un sujet : l'application génère des **QCM corrigés** et des **flashcards**,
chacun relié à la **page source** du cours. Les questions sont produites **uniquement à partir des extraits
retrouvés** dans ton document (pipeline RAG), ce qui limite les hallucinations.

> **Démo :** _à ajouter (GIF ou lien Streamlit Cloud)_

## Fonctionnement

```text
PDF de cours
   ↓ parsing (PyMuPDF, page par page)
   ↓ nettoyage + recursive chunking (~900 caractères, overlap ~10 %)
   ↓ embeddings (locaux ou API) → ChromaDB (sans doublons)
                              ↓
Sujet choisi → retrieval (similarité cosinus, top-k)
                              ↓
        template de prompt FIXE + extraits → LLM (JSON)
                              ↓
        validation Pydantic (4 options, page source...)
                              ↓
   QCM + flashcards → interface Streamlit + révision espacée (Leitner)
```

Le prompt est un **template fixe** : seul le contexte change, et il est rempli automatiquement par le retrieval.
Ajouter un cours = l'indexer, sans toucher au prompt.

## Installation

```bash
git clone https://github.com/mariem-rabeh/quiz-generator.git && cd quiz-generator
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Puis copie `.env.example` vers `.env` et renseigne ton fournisseur de LLM.
Toute API **compatible OpenAI** fonctionne (OpenAI, Groq, Gemini, xAI, Mistral, Ollama...) :
il suffit de changer ces trois variables.

```text
LLM_API_KEY=ta-clé
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=nom-du-modèle
```

Les **embeddings sont calculés localement** par défaut (`EMBEDDING_PROVIDER=local`) : aucun quota ni coût d'API
pour l'indexation. Si tu changes de modèle d'embedding, supprime le dossier `data/chroma` et réindexe tes PDF
(les vecteurs de deux modèles différents ne sont pas comparables).

## Utilisation

```bash
streamlit run app.py
```

1. Dépose un PDF et clique sur **Indexer ce PDF**.
2. Choisis le cours, saisis un sujet, règle le nombre de QCM et de flashcards, puis **Générer le quiz**.
3. Réponds aux QCM (correction immédiate, explication, page source).
4. Onglet **Réviser** : tes flashcards reviennent selon le système de boîtes de Leitner (0, 1, 3, 7, 14 jours).

## Structure du projet

```text
quiz-generator/
├── app.py                  Interface Streamlit
├── src/
│   ├── parse.py            Extraction du texte PDF page par page
│   ├── chunk.py            Nettoyage + recursive chunking avec overlap
│   ├── store.py            Embeddings, index ChromaDB, retrieval
│   ├── schemas.py          Modèles Pydantic (MCQ, Flashcard, QuizOutput)
│   ├── generate.py         Template de prompt fixe, génération, retries
│   └── review.py           Révision espacée (Leitner)
├── tests/                  Tests pytest (aucun appel réseau)
└── eval/eval_questions.py  Évaluation hors ligne + benchmark de générations
```

## Tests et évaluation

```bash
pytest
```

Évaluation hors ligne d'un quiz déjà généré (aucun appel API) :

```bash
python -m eval.eval_questions --quiz quiz.json --context chunks.json
```

Benchmark de **30 générations réelles** (appels OpenAI payants) :

```bash
python -m eval.eval_questions --benchmark 30 --topics "sujet 1,sujet 2" --source cours.pdf
```

### Résultats

_À compléter avec tes propres mesures. Aucun chiffre n'est donné ici tant qu'il n'a pas été mesuré._

| Mesure | Comment la mesurer | Résultat |
|---|---|---|
| JSON valide au premier essai | `--benchmark 30` | _à compléter_ |
| QCM fidèles au cours | Vérification manuelle de 30 QCM | _à compléter_ |
| Page citée correcte | Vérification manuelle | _à compléter_ |
| Qualité des distracteurs | Note manuelle sur 5 | _à compléter_ |

## Limites

- Les **PDF scannés** (sans couche texte) ne sont pas lisibles : un OCR serait nécessaire.
- Chaque génération consomme des appels à l'API du LLM : les offres gratuites ont des **limites de débit** et de quota.
- Le modèle d'embedding local par défaut est surtout entraîné sur l'anglais : pour des cours en français ou en
  arabe, utilise un modèle multilingue (voir `.env.example`).
- La qualité des **mauvaises réponses** (distracteurs) dépend du modèle et reste à vérifier.
- Le LLM peut citer une page inexacte : l'évaluation mesure ce taux, mais l'application ne le corrige pas.

## Améliorations possibles

- Retrieval hybride (BM25 + dense + Reciprocal Rank Fusion)
- OCR pour les PDF scannés
- Export des flashcards vers Anki
- Cours en arabe et multilingue

## Licence

MIT, voir [LICENSE](LICENSE).
