"""Small local lexical retriever for the synthetic RAG experiment."""

from dataclasses import dataclass
from .rag_corpus import load_corpus, tokenize

@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    document_type: str
    topic: str
    text: str
    score: float

class LexicalRetriever:
    def __init__(self, documents=None):
        self.documents = list(documents or load_corpus())
        self._terms = {d.document_id: tokenize(f"{d.topic} {d.text}") for d in self.documents}

    def retrieve(self, query, top_k=3):
        query_terms = tokenize(query)
        ranked = []
        for document in self.documents:
            overlap = query_terms & self._terms[document.document_id]
            score = len(overlap) / max(1, len(query_terms))
            ranked.append(RetrievedDocument(document.document_id, document.document_type, document.topic, document.text, round(score, 4)))
        return sorted(ranked, key=lambda d: (-d.score, d.document_id))[:max(1, top_k)]
