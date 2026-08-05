"""
rag.ingestor — end-to-end pipeline for ingesting a supplier document.

Pipeline
--------
1. Compute SHA-256 of the raw bytes — reject if already present (dedup).
2. Extract text:
   - PDF bytes → pypdf PdfReader → concatenated page text.
   - Plain bytes → UTF-8 decode (fallback latin-1).
3. Split text into overlapping chunks via rag.chunker.
4. Embed all chunks in one batch via rag.embedder.
5. INSERT a row into ``documents``, then bulk-INSERT all chunks.
6. Return a JSON string with the outcome so callers always get a string.

Error handling
--------------
Every public function catches all exceptions and returns a JSON error
string instead of raising, so the LangChain agent tool never crashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from typing import Literal

from sqlalchemy.orm import Session

from database.db import Document, DocumentChunk, SessionLocal
from rag.chunker import chunk_text
from rag.embedder import embed_texts

logger = logging.getLogger(__name__)

DocType = Literal["sla", "contract", "policy"]


def _sha256_hex(data: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _extract_text_from_pdf(data: bytes) -> tuple[str, int]:
    """
    Extract plain text from PDF bytes using pypdf.

    Args:
        data: Raw PDF file bytes.

    Returns:
        A (text, page_count) tuple.  *text* is page texts joined by newlines.
        *page_count* is the number of pages in the PDF.

    Raises:
        Exception: Propagates pypdf errors so the caller can wrap them.
    """
    from pypdf import PdfReader  # local import — keeps startup fast

    reader = PdfReader(io.BytesIO(data))
    page_count = len(reader.pages)
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        pages.append(extracted)
    return "\n".join(pages), page_count


def _extract_text_from_bytes(data: bytes) -> tuple[str, None]:
    """
    Decode raw bytes as UTF-8 (with latin-1 fallback) for plain-text files.

    Args:
        data: Raw file bytes.

    Returns:
        A (text, None) tuple — page_count is None for non-PDF documents.
    """
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace"), None


def ingest_document(
    file_bytes: bytes,
    filename: str,
    supplier_name: str,
    doc_type: DocType,
) -> str:
    """
    Ingest a supplier document into the vector store.

    Accepts PDF or plain-text bytes.  The file type is inferred from the
    filename extension: ``.pdf`` triggers PDF extraction, everything else
    is treated as plain text.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used for display and type detection).
        supplier_name: Name of the supplier this document belongs to.
        doc_type: One of ``'sla'``, ``'contract'``, or ``'policy'``.

    Returns:
        A JSON string with one of two shapes:

        Success::

            {
                "status": "ok",
                "document_id": 42,
                "filename": "acme_sla.pdf",
                "supplier_name": "Acme Corp",
                "doc_type": "sla",
                "chunks_stored": 17,
                "page_count": 5
            }

        Duplicate::

            {"status": "duplicate", "message": "...", "sha256_hex": "..."}

        Error::

            {"status": "error", "message": "..."}
    """
    try:
        if not file_bytes:
            return json.dumps({"status": "error", "message": "file_bytes is empty."})

        sha = _sha256_hex(file_bytes)

        db: Session = SessionLocal()
        try:
            # --- Deduplication check -------------------------------------------
            existing = db.query(Document).filter(Document.sha256_hex == sha).first()
            if existing is not None:
                return json.dumps(
                    {
                        "status": "duplicate",
                        "message": (
                            f"Document '{filename}' is identical to already-stored "
                            f"document id={existing.id} ('{existing.filename}'). "
                            "No new rows were inserted."
                        ),
                        "sha256_hex": sha,
                    }
                )

            # --- Text extraction -----------------------------------------------
            is_pdf = filename.lower().endswith(".pdf")
            if is_pdf:
                try:
                    raw_text, page_count = _extract_text_from_pdf(file_bytes)
                except Exception as exc:
                    return json.dumps(
                        {
                            "status": "error",
                            "message": f"PDF extraction failed for '{filename}': {exc}",
                        }
                    )
            else:
                raw_text, page_count = _extract_text_from_bytes(file_bytes)

            if not raw_text.strip():
                return json.dumps(
                    {
                        "status": "error",
                        "message": (
                            f"No readable text could be extracted from '{filename}'. "
                            "The file may be image-only or corrupted."
                        ),
                    }
                )

            # --- Chunking -------------------------------------------------------
            chunks = chunk_text(raw_text)
            if not chunks:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Chunking produced 0 chunks for '{filename}'.",
                    }
                )

            logger.info(
                "Ingesting '%s': %d chunks from %d chars of text.",
                filename,
                len(chunks),
                len(raw_text),
            )

            # --- Embedding (batch) ---------------------------------------------
            embeddings = embed_texts(chunks)  # shape: (n_chunks, 384)

            # --- Database writes -----------------------------------------------
            doc = Document(
                filename=filename,
                supplier_name=supplier_name,
                doc_type=doc_type,
                sha256_hex=sha,
                page_count=page_count,
            )
            db.add(doc)
            db.flush()  # populate doc.id before inserting chunks

            chunk_rows = [
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=idx,
                    chunk_text=text,
                    embedding=embeddings[idx].tolist(),
                )
                for idx, text in enumerate(chunks)
            ]
            db.bulk_save_objects(chunk_rows)
            db.commit()

            logger.info(
                "Ingested document id=%d ('%s') — %d chunks stored.",
                doc.id,
                filename,
                len(chunks),
            )

            return json.dumps(
                {
                    "status": "ok",
                    "document_id": doc.id,
                    "filename": filename,
                    "supplier_name": supplier_name,
                    "doc_type": doc_type,
                    "chunks_stored": len(chunks),
                    "page_count": page_count,
                }
            )

        except Exception as exc:
            db.rollback()
            logger.exception("ingest_document failed for '%s'.", filename)
            return json.dumps({"status": "error", "message": str(exc)})
        finally:
            db.close()

    except Exception as exc:
        logger.exception("Unexpected error in ingest_document.")
        return json.dumps({"status": "error", "message": str(exc)})
