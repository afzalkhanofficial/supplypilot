"""
rag — Supplier Document Intelligence package.

Exposes the four primary callables used by the rest of the codebase:

    from rag import ingest_document, search_documents, chunk_text, get_embedder

Sub-modules
-----------
chunker  : splits raw text into overlapping character windows
embedder : singleton wrapper around all-MiniLM-L6-v2
ingestor : end-to-end pipeline — bytes → chunks → vectors → Postgres
search   : cosine-similarity ANN query returning cited passages
"""

from rag.chunker import chunk_text
from rag.embedder import get_embedder
from rag.ingestor import ingest_document
from rag.search import search_documents

__all__ = [
    "chunk_text",
    "get_embedder",
    "ingest_document",
    "search_documents",
]
