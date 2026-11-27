"""Chroma vector-store backend and persisted BM25 corpus snapshots."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def create_vector_collection(
    documents: list[Document],
    embedding: Any,
    ids: list[str],
    collection_name: str,
    persist_directory: Path,
) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        ids=ids,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vector_collection(
    collection_name: str,
    embedding: Any,
    persist_directory: Path,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=str(persist_directory),
    )


def get_collection_ids(vector_store: Chroma) -> set[str]:
    return set(vector_store._collection.get(include=["metadatas"])["ids"])


def write_bm25_snapshot(
    records: list[dict[str, Any]],
    documents: list[Document],
    path: Path,
) -> None:
    """Validate a BM25 corpus and atomically persist its rebuildable records."""
    BM25Retriever.from_documents(documents)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as snapshot:
        for record in records:
            snapshot.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path = snapshot.name
    os.replace(temp_path, path)
