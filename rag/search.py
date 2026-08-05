"""
rag.search — cosine-similarity search over stored document chunks.

The query vector is produced by the same all-MiniLM-L6-v2 model used
during ingestion, so query and chunk embeddings live in the same vector
space and cosine similarity is well-defined.

pgvector's ``<=>`` operator computes cosine *distance* (1 − similarity),
so lower scores mean higher relevance.  Results are returned in ascending
distance order.

The IVFFlat index on ``document_chunks.embedding`` (lists=50) means ANN
search scales to tens of thousands of chunks without a full table scan.
For exact search (small corpus), drop the index and pgvector falls back to
a sequential scan automatically.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from database.db import SessionLocal
from rag.embedder import embed_one

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.0  # cosine similarity floor
# Note: cosine similarity ranges from -1.0 to +1.0. A floor of 0.0 ensures
# all non-negative similarity matches are returned, sorted by relevance.


def search_documents(
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    supplier_name: str | None = None,
    doc_type: str | None = None,
    min_similarity: float = _DEFAULT_MIN_SCORE,
) -> str:
    """
    Find the *top_k* document chunks most similar to *query*.

    Args:
        query: Natural-language question or keyword string.
        top_k: Maximum number of chunks to return (default 5).
        supplier_name: Optional filter — restrict results to a specific
            supplier.  Case-insensitive prefix match.
        doc_type: Optional filter — one of ``'sla'``, ``'contract'``,
            ``'policy'``.  Exact match.
        min_similarity: Discard chunks whose cosine similarity to the query
            is below this threshold (default 0.70).  Set to 0.0 to disable.

    Returns:
        A JSON string. On success::

            {
                "status": "ok",
                "query": "...",
                "results": [
                    {
                        "rank": 1,
                        "similarity": 0.87,
                        "chunk_text": "...",
                        "document_id": 3,
                        "filename": "acme_sla.pdf",
                        "supplier_name": "Acme Corp",
                        "doc_type": "sla",
                        "chunk_index": 4
                    },
                    ...
                ]
            }

        On error or no results::

            {"status": "no_results", "query": "...", "message": "..."}
            {"status": "error", "message": "..."}
    """
    try:
        query = query.strip()
        if not query:
            return json.dumps(
                {"status": "error", "message": "query string must not be empty."}
            )

        if top_k < 1 or top_k > 50:
            top_k = max(1, min(top_k, 50))

        # Embed the query with the same model used during ingestion.
        query_vec = embed_one(query).tolist()

        # Build the WHERE clause conditionally so we can apply optional filters.
        # pgvector's <=> operator returns cosine distance; convert to similarity.
        conditions: list[str] = ["1 - (dc.embedding <=> CAST(:qvec AS vector)) >= :min_sim"]
        params: dict[str, Any] = {
            "qvec": str(query_vec),
            "min_sim": min_similarity,
            "top_k": top_k,
        }

        if supplier_name:
            conditions.append("LOWER(d.supplier_name) LIKE LOWER(:supplier_name)")
            params["supplier_name"] = f"{supplier_name.strip()}%"

        if doc_type:
            conditions.append("d.doc_type = :doc_type")
            params["doc_type"] = doc_type.lower().strip()

        where_clause = " AND ".join(conditions)

        sql = text(
            f"""
            SELECT
                dc.id            AS chunk_id,
                dc.chunk_index,
                dc.chunk_text,
                dc.document_id,
                d.filename,
                d.supplier_name,
                d.doc_type,
                1 - (dc.embedding <=> CAST(:qvec AS vector)) AS similarity
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE {where_clause}
            ORDER BY dc.embedding <=> CAST(:qvec AS vector)
            LIMIT :top_k
            """
        )

        db = SessionLocal()
        try:
            rows = db.execute(sql, params).fetchall()
        finally:
            db.close()

        if not rows:
            return json.dumps(
                {
                    "status": "no_results",
                    "query": query,
                    "message": (
                        "No document chunks matched the query with the current "
                        f"filters (min_similarity={min_similarity})."
                    ),
                }
            )

        results = [
            {
                "rank": rank,
                "similarity": round(float(row.similarity), 4),
                "chunk_text": row.chunk_text,
                "document_id": row.document_id,
                "filename": row.filename,
                "supplier_name": row.supplier_name,
                "doc_type": row.doc_type,
                "chunk_index": row.chunk_index,
            }
            for rank, row in enumerate(rows, start=1)
        ]

        logger.info(
            "search_documents: query='%s' returned %d results (top sim=%.3f).",
            query[:60],
            len(results),
            results[0]["similarity"] if results else 0.0,
        )

        return json.dumps({"status": "ok", "query": query, "results": results})

    except Exception as exc:
        logger.exception("search_documents failed.")
        return json.dumps({"status": "error", "message": str(exc)})


def list_documents(
    supplier_name: str | None = None,
    doc_type: str | None = None,
) -> str:
    """
    List all ingested documents, with optional filters.

    Args:
        supplier_name: Optional supplier name prefix filter (case-insensitive).
        doc_type: Optional exact doc_type filter.

    Returns:
        A JSON string::

            {
                "status": "ok",
                "count": 3,
                "documents": [
                    {
                        "id": 1,
                        "filename": "acme_sla.pdf",
                        "supplier_name": "Acme Corp",
                        "doc_type": "sla",
                        "page_count": 5,
                        "uploaded_at": "2024-01-15T10:30:00+00:00"
                    },
                    ...
                ]
            }
    """
    try:
        conditions: list[str] = []
        params: dict[str, Any] = {}

        if supplier_name:
            conditions.append("LOWER(supplier_name) LIKE LOWER(:supplier_name)")
            params["supplier_name"] = f"{supplier_name.strip()}%"

        if doc_type:
            conditions.append("doc_type = :doc_type")
            params["doc_type"] = doc_type.lower().strip()

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = text(
            f"""
            SELECT id, filename, supplier_name, doc_type, page_count, uploaded_at
            FROM documents
            {where}
            ORDER BY uploaded_at DESC
            """
        )

        db = SessionLocal()
        try:
            rows = db.execute(sql, params).fetchall()
        finally:
            db.close()

        documents = [
            {
                "id": row.id,
                "filename": row.filename,
                "supplier_name": row.supplier_name,
                "doc_type": row.doc_type,
                "page_count": row.page_count,
                "uploaded_at": row.uploaded_at.isoformat() if row.uploaded_at else None,
            }
            for row in rows
        ]

        return json.dumps({"status": "ok", "count": len(documents), "documents": documents})

    except Exception as exc:
        logger.exception("list_documents failed.")
        return json.dumps({"status": "error", "message": str(exc)})
