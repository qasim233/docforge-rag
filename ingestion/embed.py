"""Embedding model access, chunk metadata, and near-duplicate filtering."""

import hashlib
import json
import math
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.deps import get_embeddings
from ingestion.chunk import CHUNKING_STRATEGY


def get_embedding_model() -> HuggingFaceEmbeddings:
    return get_embeddings()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def indexable_chunks(
    chunks: list[Document],
) -> tuple[list[Document], list[str], list[dict[str, Any]], str]:
    """Prepare consistent metadata, IDs, and a fingerprint for both indexes."""
    if not chunks:
        raise ValueError("Cannot build indexes without document chunks")

    indexed_chunks: list[Document] = []
    chunk_ids: list[str] = []
    records: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source") or "unknown")
        metadata: dict[str, Any] = {
            key: value
            for key, value in chunk.metadata.items()
            if value is not None
            and isinstance(value, (str, int, float, bool))
        }
        metadata.update(
            {
                "source_document": source,
                "chunk_index": chunk_index,
                "section_heading": str(chunk.metadata.get("section_heading") or ""),
                "chunking_strategy": str(
                    chunk.metadata.get("chunking_strategy") or CHUNKING_STRATEGY
                ),
                "character_count": len(chunk.page_content),
            }
        )
        chunk_id = hashlib.sha256(
            f"{source}\0{chunk_index}\0{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        metadata["chunk_id"] = chunk_id
        indexed_chunk = Document(page_content=chunk.page_content, metadata=metadata)
        indexed_chunks.append(indexed_chunk)
        chunk_ids.append(chunk_id)
        records.append(
            {
                "id": chunk_id,
                "page_content": indexed_chunk.page_content,
                "metadata": metadata,
            }
        )

    fingerprint = hashlib.sha256(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            for record in records
        ).encode("utf-8")
    ).hexdigest()
    return indexed_chunks, chunk_ids, records, fingerprint


def deduplicate_chunks(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings | None = None,
    similarity_threshold: float | None = None,
) -> list[Document]:
    """Keep the first chunk in each cosine-similar group and report skipped copies."""
    threshold = (
        settings.dedup_similarity_threshold
        if similarity_threshold is None
        else similarity_threshold
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0")
    if not chunks:
        return []

    embedding_model = embedding_model or get_embedding_model()
    vectors = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding model returned a different number of vectors than chunks")

    unique_chunks: list[Document] = []
    unique_vectors: list[list[float]] = []
    duplicates = 0
    for chunk, vector in zip(chunks, vectors):
        match_index = next(
            (
                index
                for index, existing_vector in enumerate(unique_vectors)
                if _cosine_similarity(vector, existing_vector) > threshold
            ),
            None,
        )
        if match_index is not None:
            duplicates += 1
            duplicate_source = chunk.metadata.get("source", "unknown")
            existing_source = unique_chunks[match_index].metadata.get("source", "unknown")
            similarity = _cosine_similarity(vector, unique_vectors[match_index])
            print(
                f"Skipping near-duplicate chunk from {duplicate_source!r}: "
                f"cosine similarity {similarity:.4f} with {existing_source!r}"
            )
            continue
        unique_chunks.append(chunk)
        unique_vectors.append(vector)

    print(f"Skipped {duplicates} near-duplicate chunks")
    return unique_chunks
