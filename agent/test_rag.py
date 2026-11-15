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


def test_corpus_can_be_written_and_loaded(tmp_path):
    path = write_corpus(tmp_path / "corpus")
    loaded = load_corpus(path)
    assert len(loaded) == len(DOCUMENTS)
    assert loaded[0].document_id == DOCUMENTS[0].document_id
