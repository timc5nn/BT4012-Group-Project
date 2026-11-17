"""Retrievers for the synthetic RAG experiment."""

from dataclasses import dataclass

import numpy as np
from openai import APIConnectionError, APIError

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


class EmbeddingRetriever:
    """Retrieve documents with Ollama embeddings, falling back to lexical search."""

    def __init__(self, documents=None, model="nomic-embed-text", client=None):
        self.documents = list(documents or load_corpus())
        self.model = model
        self.client = client
        self.fallback = LexicalRetriever(self.documents)
        self.last_error = None
        self._document_vectors = None

    def _client(self):
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        return self.client

    def _embed(self, texts):
        response = self._client().embeddings.create(model=self.model, input=texts)
        data = sorted(response.data, key=lambda item: item.index)
        vectors = np.asarray([item.embedding for item in data], dtype=float)
        if len(vectors) != len(texts) or vectors.ndim != 2:
            raise ValueError("embedding service returned an invalid response")
        return vectors

    def _documents(self):
        if self._document_vectors is None:
            self._document_vectors = self._embed([
                f"{document.topic}\n{document.text}" for document in self.documents
            ])
        return self._document_vectors

    def retrieve(self, query, top_k=3):
        try:
            document_vectors = self._documents()
            query_vector = self._embed([query])[0]
            document_norms = np.linalg.norm(document_vectors, axis=1)
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0 or np.any(document_norms == 0):
                raise ValueError("embedding service returned a zero-length vector")
            scores = document_vectors @ query_vector / (document_norms * query_norm)
            ranked = [
                RetrievedDocument(
                    document.document_id,
                    document.document_type,
                    document.topic,
                    document.text,
                    round(float(score), 4),
                )
                for document, score in zip(self.documents, scores)
            ]
            self.last_error = None
            return sorted(ranked, key=lambda d: (-d.score, d.document_id))[:max(1, top_k)]
        except (APIConnectionError, APIError, ImportError, OSError, ValueError) as error:
            self.last_error = str(error)
            return self.fallback.retrieve(query, top_k)
