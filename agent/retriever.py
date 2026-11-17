"""Retrievers for the synthetic RAG experiment."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from openai import APIConnectionError, APIError

from .rag_corpus import load_corpus, tokenize

DEFAULT_EMBEDDING_CACHE_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "agent_runs" / "rag_embeddings"
)

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

    def __init__(self, documents=None, model="nomic-embed-text", client=None, cache_dir=None):
        self.documents = list(documents or load_corpus())
        self.model = model
        self.client = client
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_EMBEDDING_CACHE_DIR
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

    def _cache_path(self):
        corpus = [
            {
                "document_id": document.document_id,
                "document_type": document.document_type,
                "topic": document.topic,
                "text": document.text,
            }
            for document in self.documents
        ]
        cache_key = hashlib.sha256(
            json.dumps(
                {"model": self.model, "documents": corpus},
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        model_name = "".join(character if character.isalnum() or character in "-._" else "_"
                             for character in self.model)
        return self.cache_dir / f"{model_name}-{cache_key}.npz"

    def _load_cached_documents(self):
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            with np.load(path) as cached:
                vectors = cached["vectors"]
        except (OSError, ValueError, KeyError):
            return None
        if vectors.ndim != 2 or vectors.shape[0] != len(self.documents):
            return None
        return vectors

    def _save_cached_documents(self, vectors):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path()
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, vectors=vectors)
        temporary.replace(path)

    def _documents(self):
        if self._document_vectors is None:
            self._document_vectors = self._load_cached_documents()
            if self._document_vectors is None:
                self._document_vectors = self._embed([
                    f"{document.topic}\n{document.text}" for document in self.documents
                ])
                try:
                    self._save_cached_documents(self._document_vectors)
                except OSError:
                    pass
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
