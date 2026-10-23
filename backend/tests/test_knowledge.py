"""Tests for document RAG: parsing, chunking, store, hybrid retrieval, cited
answers and definition grounding. Runs offline (local embedding + reranker
models, no LLM)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.config import get_settings
from app.knowledge import definitions
from app.knowledge.answer import NOT_FOUND, answer
from app.knowledge.chunking import chunk_sections
from app.knowledge.parsing import Section, UnsupportedDocument, parse_document
from app.knowledge.retrieval import MODES, search
from app.knowledge.store import DocumentStore, reset_document_store, tokenize

DOCS = Path(__file__).resolve().parents[2] / "sample_data" / "docs"


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("name", ["metric_definitions.md", "data_dictionary.md", "business_glossary.docx", "reporting_policy.pdf"])
def test_every_sample_format_parses(name):
    sections = parse_document(name, (DOCS / name).read_bytes())
    assert sections and all(s.text.strip() for s in sections)


def test_markdown_keeps_heading_path():
    md = b"# Metrics\n\nintro\n\n## Active customer\n\nBought in 90 days.\n\n```\n# not a heading\n```\n"
    sections = parse_document("m.md", md)
    assert [s.heading for s in sections] == ["Metrics", "Metrics > Active customer"]
    assert "# not a heading" in sections[1].text


def test_pdf_sections_carry_page_numbers():
    sections = parse_document("reporting_policy.pdf", (DOCS / "reporting_policy.pdf").read_bytes())
    assert [s.page for s in sections] == [1, 2, 3]
    assert "fiscal year runs from 1 April" in sections[0].text


def test_unsupported_and_empty_documents_rejected():
    with pytest.raises(UnsupportedDocument):
        parse_document("data.xlsx", b"x")
    with pytest.raises(UnsupportedDocument):
        parse_document("empty.md", b"   \n\n")


# ----------------------------------------------------------------- chunking
def test_chunks_respect_sections_and_overlap():
    long = " ".join(f"Sentence number {i} explains the policy in detail." for i in range(60))
    sections = [Section(text=long, heading="Policy"), Section(text="Short.", heading="Other")]
    chunks = chunk_sections(sections, "d1", "doc.md", max_chars=300, overlap=80)
    policy = [c for c in chunks if c.heading == "Policy"]
    assert len(policy) > 3 and all(len(c.text) <= 300 for c in policy)
    assert all(c.heading in ("Policy", "Other") for c in chunks)          # never crosses a section
    assert policy[0].text[-40:].split()[-1] in policy[1].text            # overlap carried over
    assert chunks[-1].citation() == "[doc.md, § Other]"


def test_chunk_ids_are_content_hashes():
    a = chunk_sections([Section(text="Same text here.", heading="H")], "d1", "x.md")
    b = chunk_sections([Section(text="Same text here.", heading="H")], "d1", "x.md")
    c = chunk_sections([Section(text="Changed text here.", heading="H")], "d1", "x.md")
    assert a[0].chunk_id == b[0].chunk_id != c[0].chunk_id


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The exported reports") == ["export", "report"]


# -------------------------------------------------------------------- store
def test_reupload_only_embeds_changed_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "enable_docs", True)
    store = DocumentStore(root=tmp_path / "s")
    v1 = b"# A\n\nAlpha section text.\n\n# B\n\nBeta section text.\n"
    first = store.add("notes.md", v1)
    assert (first["embedded"], first["reused"]) == (2, 0)
    same = store.add("notes.md", v1)
    assert (same["embedded"], same["reused"], same["removed"]) == (0, 2, 0)
    edited = store.add("notes.md", v1.replace(b"Beta section", b"Gamma section"))
    assert (edited["embedded"], edited["reused"], edited["removed"]) == (1, 1, 1)
    assert len(store.chunks()) == 2


def test_store_persists_and_deletes(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "enable_docs", True)
    store = DocumentStore(root=tmp_path / "s")
    meta = store.add("notes.md", b"# A\n\nAlpha.\n")
    reopened = DocumentStore(root=tmp_path / "s")
    assert [d.name for d in reopened.list()] == ["notes.md"] and len(reopened.chunks()) == 1
    assert reopened.delete(meta["doc_id"]) and reopened.chunks() == {}


# ---------------------------------------------------------------- retrieval
@pytest.mark.parametrize("mode", MODES)
def test_each_mode_finds_the_defining_passage(docs_store, mode):
    result = search("What counts as a high-value transaction?", k=3, mode=mode)
    assert result.hits[0].chunk.heading.endswith("High-value transaction")


def test_reranker_gates_out_of_scope_questions(docs_store):
    assert search("When does the fiscal year start?").covered
    off_topic = search("Who won the football world cup in 2022?")
    assert not off_topic.covered and off_topic.hits[0].rerank_score < 0


def test_hybrid_reports_both_retrievers(docs_store):
    hit = search("UPI payment method", k=1, mode="hybrid").hits[0]
    assert hit.dense_rank is not None and hit.bm25_rank is not None


def test_search_scoped_to_documents(docs_store):
    glossary = next(d for d in docs_store.list() if d.name == "business_glossary.docx")
    result = search("high-value transaction", doc_ids=[glossary.doc_id])
    assert all(h.chunk.doc_id == glossary.doc_id for h in result.hits)


# ------------------------------------------------------------------ answers
def test_extractive_answer_is_cited(docs_store):
    out = answer("What is the target conversion rate?", search("What is the target conversion rate?"))
    assert out["found"] and "3.5 percent" in out["answer"]
    assert "[reporting_policy.pdf, p.2" in out["answer"]
    assert out["citations"][0]["page"] == 2


def test_not_found_when_documents_do_not_cover(docs_store):
    out = answer("What is the capital of France?", search("What is the capital of France?"))
    assert out == {"answer": NOT_FOUND, "citations": [], "found": False, "mode": "none", "passages": []}


class _FakeLLM:
    available = True

    def __init__(self, reply):
        self.reply, self.messages = reply, None

    def complete(self, messages, max_tokens=None, temperature=None):
        self.messages = messages
        return self.reply


def test_llm_answer_maps_citations_and_drops_invented_ones(docs_store):
    llm = _FakeLLM("It is an order total above 10,000 rupees [1]. Reviewed by fraud [1][9].")
    out = answer("high-value transaction?", search("What counts as a high-value transaction?"), llm=llm)
    assert "[metric_definitions.md, § High-value transaction]" in out["answer"]
    assert "[9]" not in out["answer"] and len(out["citations"]) == 1
    # passages are fenced as data and the model is told to ignore instructions in them
    assert "<<<" in llm.messages[1]["content"] and "untrusted data" in llm.messages[0]["content"]


def test_llm_not_found_is_respected(docs_store):
    out = answer("q", search("When does the fiscal year start?"), llm=_FakeLLM("NOT_FOUND"))
    assert not out["found"] and out["answer"] == NOT_FOUND


# -------------------------------------------------------------- definitions
@pytest.fixture()
def retail_record():
    from app.core.registry import DataFrameRegistry

    df = pd.DataFrame({"total": [500.0, 15000.0, 22000.0], "rating": [5, 1, 4], "city": ["Pune", "Delhi", "Pune"]})
    return DataFrameRegistry().register(df, "retail_transactions.csv", "t")


def test_definitions_parsed_from_docs(docs_store):
    terms = {(d.term, d.kind) for d in definitions.all_definitions()}
    assert ("High-value transaction", "filter") in terms and ("Gross margin", "expression") in terms


def test_definition_matches_plural_term_for_its_dataset(docs_store, retail_record):
    found = definitions.match("How many high value transactions were there?", retail_record)
    assert [d.sql for d in found] == ['"total" > 10000']
    assert found[0].citation == "[metric_definitions.md, § High-value transaction]"
    assert definitions.match("bulk orders", retail_record) == []   # applies to sales_data.csv only


def test_unsafe_or_mismatched_definition_sql_rejected(retail_record):
    assert not definitions._check('"total" > 1; DROP TABLE x', retail_record)
    assert not definitions._check("\"total\" > (SELECT 1)", retail_record)
    assert not definitions._check("read_text('/etc/passwd') IS NOT NULL", retail_record)
    assert not definitions._check('"no_such_column" > 5', retail_record)
    assert definitions._check('"total" > 10000', retail_record)


# ---------------------------------------------------------------------- API
def test_documents_api_roundtrip(docs_store):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    listing = client.get("/api/documents").json()
    assert listing["count"] == 4 and len(listing["definitions"]) >= 10
    bad = client.post("/api/documents", files=[("files", ("x.exe", b"MZ"))])
    assert bad.status_code == 400
    res = client.post("/api/documents/search", json={"query": "What is UPI?", "answer": True}).json()
    assert res["covered"] and "[business_glossary.docx" in res["answer"]["answer"]
    doc_id = listing["documents"][0]["doc_id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.get("/api/documents").json()["count"] == 3


def test_sample_documents_endpoint(docs_store):
    from fastapi.testclient import TestClient

    from app.main import app

    res = TestClient(app).post("/api/documents/samples").json()
    assert res["count"] == 4 and all(d["embedded"] == 0 for d in res["imported"])  # already indexed: all reused


def test_llm_native_citation_format_is_mapped(docs_store):
    # gpt-oss cites as 【1†L1-L4】 rather than [1]
    llm = _FakeLLM("Order total above 10,000 rupees【1†L1-L4】.")
    out = answer("high value?", search("What counts as a high-value transaction?"), llm=llm)
    assert out["found"] and "【" not in out["answer"]
    assert "[metric_definitions.md, § High-value transaction]" in out["answer"]
