"""Clause-level retrieval over the policy knowledge base.

Documents are split on their ``## <CLAUSE-ID> Title`` headings so every retrieved chunk maps
to exactly one citable clause. Ranking uses TF-IDF with bigrams, which needs no external
service and is fully reproducible; the interface allows swapping in a vector store.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .schemas import RetrievedClause

HEADING = re.compile(r"^##\s+([A-Z]{2,4}-\d+)\s+(.*)$")
CLAUSE_ID = re.compile(r"\b(?:RLP|MRM|KYC|AML)-\d+\b")


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
    def __init__(self, kb_dir: Path | None = None):
        self.clauses = load_clauses(kb_dir)
        if not self.clauses:
            raise ValueError("Knowledge base is empty - add markdown files with '## ID Title' headings.")
        self._by_id = {c.clause_id: c for c in self.clauses}
        corpus = [f"{c.title}. {c.title}. {c.text}" for c in self.clauses]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    @property
    def clause_ids(self) -> set[str]:
        return set(self._by_id)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[RetrievedClause]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [self._to_result(self.clauses[i], float(scores[i])) for i in ranked if scores[i] >= min_score]

    def get(self, clause_id: str) -> RetrievedClause | None:
        clause = self._by_id.get(clause_id)
        return self._to_result(clause, 1.0) if clause else None

    @staticmethod
    def _to_result(c: Clause, score: float) -> RetrievedClause:
        return RetrievedClause(clause_id=c.clause_id, title=c.title, source=c.source, text=c.text, score=round(score, 4))
