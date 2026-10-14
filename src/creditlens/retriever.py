"""Clause-level hybrid retrieval over the policy knowledge base.

Documents are split on their ``## <CLAUSE-ID> Title`` headings so every retrieved chunk maps
to exactly one citable clause. Two retrievers run side by side:

* **lexical** - TF-IDF with bigrams: exact on thresholds, IDs and policy jargon;
* **dense**   - Gemini embeddings: matches paraphrases ("can my brother co-sign?" -> COL-4).

Their rankings are merged with Reciprocal Rank Fusion, which needs no score calibration
between the two systems. Without an embedding key the retriever degrades to lexical only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .embeddings import GeminiEmbedder, get_embedder
from .schemas import RetrievedClause

HEADING = re.compile(r"^##\s+([A-Z]{2,5}-\d+)\s+(.*)$")
RRF_K = 60  # standard damping constant for reciprocal rank fusion
MIN_LEXICAL = 0.05
MIN_DENSE = 0.60  # Gemini embeddings: off-topic queries score ~0.52, relevant clauses ~0.67+


@dataclass
class Clause:
    clause_id: str
    title: str
    source: str
    text: str


def load_clauses(kb_dir: Path | None = None) -> list[Clause]:
    kb_dir = kb_dir or config.KNOWLEDGE_BASE_DIR
    clauses: list[Clause] = []
    for path in sorted(kb_dir.glob("*.md")):
        current: Clause | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            match = HEADING.match(line)
            if match:
                if current:
                    clauses.append(current)
                current = Clause(match.group(1), match.group(2).strip(), path.name, "")
            elif current is not None and line.strip():
                current.text = f"{current.text} {line.strip()}".strip()
        if current:
            clauses.append(current)
    return clauses


class PolicyRetriever:
    def __init__(self, kb_dir: Path | None = None, embedder: GeminiEmbedder | None | bool = True,
                 min_dense: float = MIN_DENSE):
        self.min_dense = min_dense  # cosine floor; depends on the embedding model's score scale
        self.clauses = load_clauses(kb_dir)
        if not self.clauses:
            raise ValueError("Knowledge base is empty - add markdown files with '## ID Title' headings.")
        self._by_id = {c.clause_id: c for c in self.clauses}
        corpus = [f"{c.title}. {c.title}. {c.text}" for c in self.clauses]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

        self.embedder = get_embedder() if embedder is True else (embedder or None)
        self.dense_matrix = None
        if self.embedder is not None:
            self.dense_matrix = self.embedder.embed([f"{c.title}: {c.text}" for c in self.clauses])

    @property
    def clause_ids(self) -> set[str]:
        return set(self._by_id)

    @property
    def default_mode(self) -> str:
        # Benchmarked in scripts/evaluate_retrieval.py: dense alone beat equal-weight hybrid on
        # this corpus (lexical noise pulled paraphrase recall down), so "auto" means dense.
        if self.dense_matrix is None:
            return "lexical"
        return "dense" if config.RETRIEVAL_MODE == "auto" else config.RETRIEVAL_MODE

    def lexical_scores(self, query: str) -> np.ndarray:
        return cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]

    def dense_scores(self, query: str) -> np.ndarray:
        if self.dense_matrix is None:
            raise RuntimeError("Dense retrieval needs an embedder (set GEMINI_API_KEY).")
        q = self.embedder.embed([query], task="RETRIEVAL_QUERY")[0]
        return self.dense_matrix @ q

    def search(self, query: str, k: int = 3, mode: str | None = None) -> list[RetrievedClause]:
        mode = mode or self.default_mode
        if mode == "lexical":
            scores = self.lexical_scores(query)
            ranked = [i for i in np.argsort(-scores)[:k] if scores[i] >= MIN_LEXICAL]
            return [self._to_result(self.clauses[i], float(scores[i])) for i in ranked]
        if mode == "dense":
            scores = self.dense_scores(query)
            ranked = [i for i in np.argsort(-scores)[:k] if scores[i] >= self.min_dense]
            return [self._to_result(self.clauses[i], float(scores[i])) for i in ranked]

        lexical, dense = self.lexical_scores(query), self.dense_scores(query)
        fused = np.zeros(len(self.clauses))
        for scores in (lexical, dense):
            for rank, i in enumerate(np.argsort(-scores)):
                fused[i] += 1.0 / (RRF_K + rank + 1)
        # A clause must be at least weakly relevant to one retriever to be returned.
        eligible = (lexical >= MIN_LEXICAL) | (dense >= self.min_dense)
        ranked = [i for i in np.argsort(-fused) if eligible[i]][:k]
        return [self._to_result(self.clauses[i], float(fused[i])) for i in ranked]

    def get(self, clause_id: str) -> RetrievedClause | None:
        clause = self._by_id.get(clause_id)
        return self._to_result(clause, 1.0) if clause else None

    @staticmethod
    def _to_result(c: Clause, score: float) -> RetrievedClause:
        return RetrievedClause(clause_id=c.clause_id, title=c.title, source=c.source, text=c.text, score=round(score, 4))
