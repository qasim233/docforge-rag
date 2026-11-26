"""Document ingestion entry point: load, chunk, and build synchronized indexes."""

import json
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from chunking import (
    CHUNKING_STRATEGY,
    EMBEDDING_MODEL,
    deduplicate_chunks,
    indexable_chunks,
    split_documents,
    write_bm25_snapshot,
)
from document_loader import DATA_DIR, load_documents, load_pdfs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma")
BM25_DIR = os.path.join(BASE_DIR, "bm25")
INDEX_MANIFEST_PATH = os.path.join(BASE_DIR, "index_manifest.json")


def _write_index_manifest(manifest: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=BASE_DIR, delete=False
    ) as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
        temp_path = manifest_file.name
    os.replace(temp_path, INDEX_MANIFEST_PATH)


def create_indexes(chunks: list[Document]) -> None:
    """Build matching Chroma and BM25 indexes concurrently, then activate them."""
    if not chunks:
        raise ValueError("Cannot build indexes without document chunks")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    unique_chunks = deduplicate_chunks(chunks, embedding_model)
    indexed_chunks, chunk_ids, records, fingerprint = indexable_chunks(unique_chunks)
    index_version = uuid.uuid4().hex
    collection_name = f"docforge_{index_version}"
    bm25_snapshot = os.path.join(BM25_DIR, f"{index_version}.jsonl")

    with ThreadPoolExecutor(max_workers=2) as executor:
        chroma_future = executor.submit(
            Chroma.from_documents,
            documents=indexed_chunks,
            embedding=embedding_model,
            ids=chunk_ids,
            collection_name=collection_name,
            persist_directory=CHROMA_DIR,
        )
        bm25_future = executor.submit(
            write_bm25_snapshot, records, indexed_chunks, bm25_snapshot
        )
        chroma_store = chroma_future.result()
        bm25_future.result()

    stored_ids = set(chroma_store._collection.get(include=["metadatas"])["ids"])
    if stored_ids != set(chunk_ids):
        raise RuntimeError("Chroma and BM25 chunks differ; refusing to activate indexes")

    _write_index_manifest(
        {
            "version": index_version,
            "collection_name": collection_name,
            "bm25_snapshot": os.path.relpath(bm25_snapshot, BASE_DIR),
            "embedding_model": EMBEDDING_MODEL,
            "chunk_count": len(chunk_ids),
            "fingerprint": fingerprint,
        }
    )
    print(f"Chroma and BM25 indexes created for {len(chunk_ids)} chunks")


def create_vectorstore(chunks: list[Document]) -> None:
    """Backward-compatible name; creates both the Chroma and BM25 indexes."""
    create_indexes(chunks)


if __name__ == "__main__":
    raw_docs = load_documents(DATA_DIR)
    chunks = split_documents(raw_docs, strategy=CHUNKING_STRATEGY)
    create_indexes(chunks)
    print("Ingestion complete! Chroma and BM25 indexes are ready.")
