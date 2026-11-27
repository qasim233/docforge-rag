"""Compatibility exports for chunk preparation APIs."""

from ingestion.chunk import CHUNKING_STRATEGY, split_documents
from ingestion.embed import deduplicate_chunks, indexable_chunks
from storage.vector_store import write_bm25_snapshot

__all__ = [
    "CHUNKING_STRATEGY",
    "deduplicate_chunks",
    "indexable_chunks",
    "split_documents",
    "write_bm25_snapshot",
]
