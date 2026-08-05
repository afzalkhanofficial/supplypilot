"""
Unit and integration tests for Phase 8: Supplier Document Intelligence (RAG).

Test suites
-----------
1. TestChunker   — text windowing, overlap, boundary snapping, edge cases
2. TestEmbedder  — model singleton, embedding shape (384,), L2-normalisation
3. TestIngestor  — document ingestion, SHA-256 deduplication, vector search
4. TestAgentTools — LangChain tool wrappers for RAG search and document listing
5. TestRAGApi    — REST API endpoints (/documents, /documents/search, /documents/ingest)
"""

import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from database.db import SessionLocal, Document
from rag.chunker import chunk_text
from rag.embedder import embed_one, embed_texts, get_embedder
from rag.ingestor import ingest_document
from rag.search import list_documents, search_documents
from agent.tools import list_supplier_documents, search_supplier_docs


# ---------------------------------------------------------------------------
# 1. Chunker Unit Tests
# ---------------------------------------------------------------------------

class TestChunker:
    def test_chunk_text_basic(self):
        sample = "The quick brown fox jumps over the lazy dog. " * 30
        chunks = chunk_text(sample, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        assert all(len(c) <= 350 for c in chunks)

    def test_chunk_text_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n   ") == []

    def test_chunk_text_invalid_params(self):
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            chunk_text("test", chunk_size=0)

        with pytest.raises(ValueError, match="overlap must be strictly less than chunk_size"):
            chunk_text("test", chunk_size=100, overlap=100)

    def test_chunk_text_boundary_snap(self):
        sample = "WordOne WordTwo WordThree WordFour WordFive WordSix WordSeven"
        chunks = chunk_text(sample, chunk_size=25, overlap=5)
        assert len(chunks) >= 2
        # Verify no chunk starts or ends with a partial word if snap succeeded
        for c in chunks:
            assert not c.startswith("ord")
            assert not c.endswith("Wor")


# ---------------------------------------------------------------------------
# 2. Embedder Unit Tests
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_get_embedder_singleton(self):
        model1 = get_embedder()
        model2 = get_embedder()
        assert model1 is model2

    def test_embed_texts_shape_and_norm(self):
        texts = ["Supply chain optimization", "Prophet demand forecast"]
        vecs = embed_texts(texts)
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (2, 384)
        assert vecs.dtype == np.float32

        # Check L2-normalization (length == 1.0 within floating point precision)
        norm1 = np.linalg.norm(vecs[0])
        norm2 = np.linalg.norm(vecs[1])
        assert np.isclose(norm1, 1.0, atol=1e-4)
        assert np.isclose(norm2, 1.0, atol=1e-4)

    def test_embed_one_shape(self):
        v = embed_one("Single test query")
        assert v.shape == (384,)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-4)

    def test_embed_texts_empty_raises(self):
        with pytest.raises(ValueError, match="at least one text string"):
            embed_texts([])


# ---------------------------------------------------------------------------
# 3. Ingestor & Search Integration Tests
# ---------------------------------------------------------------------------

class TestIngestorAndSearch:
    def test_ingest_duplicate_and_search_flow(self):
        test_filename = "pytest_temp_sla_contract.txt"
        test_supplier = "PyTest Logistics Inc"
        test_content = (
            "PYTEST TEMPORARY CONTRACT AGREEMENT.\n"
            "Clause 1.1: Standard delivery lead time is three (3) business days.\n"
            "Clause 1.2: Minimum order fill rate is set to 96% measured monthly.\n"
            "Clause 1.3: Penalty for fill rate below 90% is 5% credit on invoice value.\n"
            "Clause 1.4: Emergency dispatch surcharge is 10% of order value.\n"
        ).encode("utf-8")

        db = SessionLocal()
        try:
            # Clean up prior test run if any
            db.query(Document).filter(Document.filename == test_filename).delete()
            db.commit()

            # 1. Ingest document
            raw_res = ingest_document(
                file_bytes=test_content,
                filename=test_filename,
                supplier_name=test_supplier,
                doc_type="sla",
            )
            res = json.loads(raw_res)
            assert res["status"] == "ok", res
            assert res["filename"] == test_filename
            assert res["supplier_name"] == test_supplier
            assert res["doc_type"] == "sla"
            assert res["chunks_stored"] >= 1
            doc_id = res["document_id"]

            # 2. Duplicate ingestion check
            dup_raw = ingest_document(
                file_bytes=test_content,
                filename=test_filename,
                supplier_name=test_supplier,
                doc_type="sla",
            )
            dup_res = json.loads(dup_raw)
            assert dup_res["status"] == "duplicate", dup_res

            # 3. Search document vector database
            search_raw = search_documents(
                query="what is the penalty for fill rate below 90%",
                supplier_name="PyTest Logistics",
                top_k=3,
            )
            search_res = json.loads(search_raw)
            assert search_res["status"] == "ok", search_res
            assert len(search_res["results"]) >= 1
            top_match = search_res["results"][0]
            assert top_match["supplier_name"] == test_supplier
            assert "Clause 1.3" in top_match["chunk_text"] or "Penalty" in top_match["chunk_text"]

            # 4. List documents check
            list_raw = list_documents(supplier_name="PyTest Logistics")
            list_res = json.loads(list_raw)
            assert list_res["status"] == "ok", list_res
            assert list_res["count"] == 1
            assert list_res["documents"][0]["id"] == doc_id

        finally:
            # Clean up test rows
            db.query(Document).filter(Document.filename == test_filename).delete()
            db.commit()
            db.close()


# ---------------------------------------------------------------------------
# 4. Agent Tools Integration Tests
# ---------------------------------------------------------------------------

class TestAgentTools:
    def test_list_supplier_documents_tool(self):
        tool_out = list_supplier_documents.invoke({"supplier_name": ""})
        data = json.loads(tool_out)
        assert data["status"] == "ok"
        assert "count" in data
        assert data["count"] >= 5

    def test_search_supplier_docs_tool(self):
        tool_out = search_supplier_docs.invoke({
            "query": "lead time for standard orders",
            "supplier_name": "Apex Supply Co",
        })
        data = json.loads(tool_out)
        assert data["status"] == "ok"
        assert len(data["results"]) >= 1
        assert data["results"][0]["supplier_name"].startswith("Apex")


# ---------------------------------------------------------------------------
# 5. REST API Endpoint Integration Tests
# ---------------------------------------------------------------------------

class TestRAGApi:
    def test_get_documents_endpoint(self, client: TestClient):
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["count"] >= 5

    def test_get_documents_search_endpoint(self, client: TestClient):
        response = client.get(
            "/documents/search",
            params={"q": "fill rate penalty", "top_k": 3, "supplier_name": "Apex"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert len(data["results"]) >= 1

    def test_post_documents_ingest_endpoint(self, client: TestClient):
        test_file = ("api_test_contract.txt", b"API TEST CONTRACT TEXT FOR PYTEST", "text/plain")
        db = SessionLocal()
        try:
            db.query(Document).filter(Document.filename == "api_test_contract.txt").delete()
            db.commit()

            response = client.post(
                "/documents/ingest",
                files={"file": test_file},
                data={"supplier_name": "API Test Supplier", "doc_type": "contract"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ("ok", "duplicate")
        finally:
            db.query(Document).filter(Document.filename == "api_test_contract.txt").delete()
            db.commit()
            db.close()
