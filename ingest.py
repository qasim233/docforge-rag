"""Compatibility CLI for the modular document ingestion pipeline."""

from ingestion.chunk import CHUNKING_STRATEGY, split_documents
from ingestion.extract import load_documents, load_pdfs
from ingestion.worker import create_indexes, create_vectorstore, run_ingestion

__all__ = [
    "CHUNKING_STRATEGY",
    "create_indexes",
    "create_vectorstore",
    "load_documents",
    "load_pdfs",
    "run_ingestion",
    "split_documents",
]


if __name__ == "__main__":
    run_ingestion()
