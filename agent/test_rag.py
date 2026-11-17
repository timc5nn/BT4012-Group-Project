from agent.rag_corpus import DOCUMENTS, load_corpus, write_corpus
from agent.retriever import EmbeddingRetriever, LexicalRetriever


def test_default_corpus_is_label_free():
    text = " ".join(document.text.lower() for document in DOCUMENTS)
    assert "is_fraud" not in text
    assert "trans_num" not in text
    assert len(DOCUMENTS) >= 10


def test_retriever_returns_relevant_guidance():
    results = LexicalRetriever().retrieve("many transactions in one hour transaction velocity", top_k=3)
    assert results[0].document_id == "pattern_velocity"
    assert 0 <= results[0].score <= 1


class FakeEmbeddingClient:
    class Embeddings:
        @staticmethod
        def create(model, input):
            vectors = []
            for text in input:
                vectors.append(type("Embedding", (), {
                    "index": 0,
                    "embedding": [1.0, 0.0] if "velocity" in text else [0.0, 1.0],
                })())
            return type("Response", (), {"data": vectors})()

    embeddings = Embeddings()


def test_embedding_retriever_ranks_by_cosine_similarity(tmp_path):
    documents = [
        type("Document", (), {
            "document_id": "velocity",
            "document_type": "pattern",
            "topic": "velocity",
            "text": "transaction velocity",
        })(),
        type("Document", (), {
            "document_id": "amount",
            "document_type": "pattern",
            "topic": "amount",
            "text": "unusual amount",
        })(),
    ]
    retriever = EmbeddingRetriever(documents, client=FakeEmbeddingClient(), cache_dir=tmp_path)
    assert retriever.retrieve("velocity", top_k=1)[0].document_id == "velocity"


def test_embedding_retriever_falls_back_when_service_is_unavailable(tmp_path):
    class UnavailableClient:
        class Embeddings:
            @staticmethod
            def create(model, input):
                raise OSError("Ollama is unavailable")

        embeddings = Embeddings()

    retriever = EmbeddingRetriever(client=UnavailableClient(), cache_dir=tmp_path)
    results = retriever.retrieve("many transactions in one hour transaction velocity")
    assert results[0].document_id == "pattern_velocity"
    assert retriever.last_error == "Ollama is unavailable"


def test_embedding_retriever_reuses_cached_document_vectors(tmp_path):
    calls = []

    class CountingClient:
        class Embeddings:
            @staticmethod
            def create(model, input):
                calls.append(input)
                return FakeEmbeddingClient.Embeddings.create(model, input)

        embeddings = Embeddings()

    documents = [
        type("Document", (), {
            "document_id": "velocity",
            "document_type": "pattern",
            "topic": "velocity",
            "text": "transaction velocity",
        })(),
    ]
    first = EmbeddingRetriever(documents, client=CountingClient(), cache_dir=tmp_path)
    first.retrieve("velocity")
    second = EmbeddingRetriever(documents, client=CountingClient(), cache_dir=tmp_path)
    second._documents()

    assert len(calls) == 2


def test_rag_context_sent_to_model_has_no_ids_or_scores(monkeypatch):
    """The model once cited a retrieval score as case evidence, so only guidance text is sent."""
    from types import SimpleNamespace

    from agent.llm import LLMReviewer
    from agent.reviewers import RAGReviewer

    captured = {}
    monkeypatch.setattr(LLMReviewer, "review", lambda self, case, tools: captured.setdefault("context", self.additional_context))
    reviewer = RAGReviewer(SimpleNamespace(provider="scripted", model="test-model"))
    case = SimpleNamespace(category="shopping_net", merchant="m", amount=500, xgb_score=0.9)
    try:
        reviewer.review(case, None)
    except StopIteration:  # no decision event in the transcript of the stubbed review
        pass

    context = captured["context"]
    assert all(document.text in context for document in reviewer.last_retrieval)
    assert "retrieval_score" not in context
    assert not any(document.document_id in context for document in reviewer.last_retrieval)


def test_corpus_can_be_written_and_loaded(tmp_path):
    path = write_corpus(tmp_path / "corpus")
    loaded = load_corpus(path)
    assert len(loaded) == len(DOCUMENTS)
    assert loaded[0].document_id == DOCUMENTS[0].document_id
