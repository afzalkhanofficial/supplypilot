"""
Supplier document management and RAG search routes.

Endpoints
---------
POST /documents/ingest — Upload and ingest a supplier document (PDF or TXT).
GET /documents/search — Vector search across supplier document chunks.
GET /documents        — List indexed supplier documents.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from api.schemas import (
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentSearchResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    summary="Upload and ingest a supplier document",
)
async def ingest_document_endpoint(
    file: UploadFile = File(..., description="Document file (.pdf or .txt)"),
    supplier_name: str = Form(..., description="Name of the supplier"),
    doc_type: str = Form(
        "contract",
        description="Document type: 'sla', 'contract', or 'policy'",
    ),
):
    """
    Upload and ingest a supplier SLA, contract, or policy document.

    Extracts text, splits into overlapping chunks, generates 384-dim vector
    embeddings, and stores them in PostgreSQL via pgvector.
    Duplicate uploads (matching SHA-256) are rejected without re-embedding.
    """
    from rag.ingestor import ingest_document

    doc_type_clean = doc_type.lower().strip()
    if doc_type_clean not in ("sla", "contract", "policy"):
        raise HTTPException(
            status_code=422,
            detail="doc_type must be one of: 'sla', 'contract', 'policy'",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        filename = file.filename or "uploaded_doc.txt"
        raw_res = ingest_document(
            file_bytes=content,
            filename=filename,
            supplier_name=supplier_name.strip(),
            doc_type=doc_type_clean,
        )

        res = json.loads(raw_res)
        if res.get("status") == "error":
            raise HTTPException(
                status_code=400,
                detail=res.get("message", "Document ingestion failed."),
            )

        return res

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ingest_document_endpoint failed for '%s'", file.filename)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/search",
    response_model=DocumentSearchResponse,
    summary="Search supplier document vector database",
)
def search_documents_endpoint(
    q: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(5, ge=1, le=50, description="Max results to return"),
    supplier_name: Optional[str] = Query(None, description="Optional supplier filter"),
    doc_type: Optional[str] = Query(None, description="Optional doc type filter: sla | contract | policy"),
):
    """
    Perform semantic vector search across all stored supplier document chunks.

    Returns relevant text passages ranked by cosine similarity.
    """
    from rag.search import search_documents

    try:
        raw_res = search_documents(
            query=q,
            top_k=top_k,
            supplier_name=supplier_name.strip() if supplier_name else None,
            doc_type=doc_type.strip() if doc_type else None,
        )
        res = json.loads(raw_res)
        if res.get("status") == "error":
            raise HTTPException(status_code=400, detail=res.get("message", "Search failed."))
        return res
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("search_documents_endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all indexed supplier documents",
)
def list_documents_endpoint(
    supplier_name: Optional[str] = Query(None, description="Optional supplier filter"),
    doc_type: Optional[str] = Query(None, description="Optional doc type filter: sla | contract | policy"),
):
    """
    Return a list of all indexed supplier documents with metadata.
    """
    from rag.search import list_documents

    try:
        raw_res = list_documents(
            supplier_name=supplier_name.strip() if supplier_name else None,
            doc_type=doc_type.strip() if doc_type else None,
        )
        res = json.loads(raw_res)
        if res.get("status") == "error":
            raise HTTPException(status_code=400, detail=res.get("message", "Listing failed."))
        return res
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("list_documents_endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc))
