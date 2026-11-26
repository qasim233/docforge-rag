"""Chunking strategies, duplicate filtering, and BM25 corpus preparation."""

import hashlib
import json
import math
import os
import re
import tempfile
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKING_STRATEGY = os.getenv("DOCFORGE_CHUNKING_STRATEGY", "fixed").lower()
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEAR_DUPLICATE_THRESHOLD = 0.95


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _semantic_split(
    document: Document,
    embedding_model: HuggingFaceEmbeddings,
    chunk_size: int,
    similarity_threshold: float,
) -> list[Document]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", document.page_content)
        if sentence.strip()
    ]
    if not sentences:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    units = [
        part
        for sentence in sentences
        for part in text_splitter.split_text(sentence)
        if part.strip()
    ]
    if len(units) == 1:
        groups = [units]
    else:
        vectors = embedding_model.embed_documents(units)
        groups: list[list[str]] = [[units[0]]]
        for index, unit in enumerate(units[1:], start=1):
            previous_size = sum(len(part) for part in groups[-1])
            boundary = (
                _cosine_similarity(vectors[index - 1], vectors[index])
                < similarity_threshold
                or previous_size + len(unit) > chunk_size
            )
            if boundary:
                groups.append([unit])
            else:
                groups[-1].append(unit)

    chunks: list[Document] = []
    for group in groups:
        metadata = dict(document.metadata)
        metadata["chunking_strategy"] = "semantic"
        chunks.append(Document(page_content=" ".join(group), metadata=metadata))
    return chunks


def split_documents(
    documents: list[Document],
    strategy: str = CHUNKING_STRATEGY,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    similarity_threshold: float = 0.55,
    embedding_model: HuggingFaceEmbeddings | None = None,
) -> list[Document]:
    """
    Split documents with one of three strategies:
    - fixed: fixed-size chunks with overlap, the fast and predictable baseline.
    - structure: recursive character splitting that respects paragraph/newline
      boundaries; each input document is already bounded by its section header.
    - semantic: split sentence units where adjacent embedding similarity falls
      below the threshold, producing chunks around topic changes.
    """
    strategy = strategy.lower()
    if strategy not in {"fixed", "structure", "semantic"}:
        raise ValueError(
            "strategy must be 'fixed', 'structure', or 'semantic'; "
            f"got {strategy!r}"
        )
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap smaller than chunk_size")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1.0 and 1.0")

    if strategy == "semantic":
        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        chunks = [
            chunk
            for document in documents
            for chunk in _semantic_split(
                document, embedding_model, chunk_size, similarity_threshold
            )
        ]
    else:
        separators = (
            ["\n\n", "\n", ". ", " ", ""]
            if strategy == "structure"
            else ["\n\n", "\n", " ", ""]
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap if strategy == "fixed" else 0,
            separators=separators,
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = strategy
    return chunks


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
    embedding_model: HuggingFaceEmbeddings,
    similarity_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[Document]:
    """Keep the first chunk in each cosine-similar group and report skipped copies."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0")
    if not chunks:
        return []

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
                if _cosine_similarity(vector, existing_vector) > similarity_threshold
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


def write_bm25_snapshot(
    records: list[dict[str, Any]],
    documents: list[Document],
    path: str,
) -> None:
    """Build BM25 from the indexed documents and persist its rebuildable corpus."""
    BM25Retriever.from_documents(documents)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=os.path.dirname(path), delete=False
    ) as snapshot:
        for record in records:
            snapshot.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path = snapshot.name
    os.replace(temp_path, path)
