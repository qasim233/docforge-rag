"""Synchronous ingestion worker that creates a consistent Chroma/BM25 index set."""

import json
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from ingestion.chunk import split_documents
from ingestion.embed import deduplicate_chunks, get_embedding_model, indexable_chunks
from ingestion.extract import load_documents
from storage.vector_store import create_vector_collection, get_collection_ids
from storage.vector_store import write_bm25_snapshot


def _write_index_manifest(manifest: dict[str, Any]) -> None:
    settings.index_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=settings.index_manifest_path.parent,
        delete=False,
    ) as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
        temp_path = manifest_file.name
    os.replace(temp_path, settings.index_manifest_path)


def create_indexes(chunks: list[Document]) -> None:
    """Build matching Chroma and BM25 indexes concurrently, then activate them."""
    if not chunks:
        raise ValueError("Cannot build indexes without document chunks")

    embedding_model = get_embedding_model()
    unique_chunks = deduplicate_chunks(chunks, embedding_model)
    indexed_chunks, chunk_ids, records, fingerprint = indexable_chunks(unique_chunks)
    index_version = uuid.uuid4().hex
    collection_name = f"docforge_{index_version}"
    bm25_snapshot = settings.bm25_dir / f"{index_version}.jsonl"

    with ThreadPoolExecutor(max_workers=2) as executor:
        chroma_future = executor.submit(
            create_vector_collection,
            documents=indexed_chunks,
            embedding=embedding_model,
            ids=chunk_ids,
            collection_name=collection_name,
            persist_directory=settings.chroma_dir,
        )
        bm25_future = executor.submit(
            write_bm25_snapshot, records, indexed_chunks, bm25_snapshot
        )
        chroma_store = chroma_future.result()
        bm25_future.result()

    if get_collection_ids(chroma_store) != set(chunk_ids):
        raise RuntimeError("Chroma and BM25 chunks differ; refusing to activate indexes")

    _write_index_manifest(
        {
            "version": index_version,
            "collection_name": collection_name,
            "distance_metric": "cosine",
            "bm25_snapshot": os.path.relpath(bm25_snapshot, settings.index_manifest_path.parent),
            "embedding_model": settings.embedding_model,
            "chunk_count": len(chunk_ids),
            "fingerprint": fingerprint,
        }
    )
    print(f"Chroma and BM25 indexes created for {len(chunk_ids)} chunks")


def run_ingestion() -> None:
    documents = load_documents(str(settings.data_dir))
    chunks = split_documents(
        documents,
        strategy=settings.chunking_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        similarity_threshold=settings.semantic_similarity_threshold,
    )
    create_indexes(chunks)
    print(f"Ingestion complete; indexed chunks are stored under {settings.chroma_dir}")
