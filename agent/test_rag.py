from agent.rag_corpus import DOCUMENTS, load_corpus, write_corpus
from agent.retriever import LexicalRetriever


def test_default_corpus_is_label_free():
    text = " ".join(document.text.lower() for document in DOCUMENTS)
    assert "is_fraud" not in text
    assert "trans_num" not in text
    assert len(DOCUMENTS) >= 10


def test_retriever_returns_relevant_guidance():
    results = LexicalRetriever().retrieve("many transactions in one hour transaction velocity", top_k=3)
    assert results[0].document_id == "pattern_velocity"
    assert 0 <= results[0].score <= 1


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
