"""Chunking strategies, duplicate filtering, and BM25 corpus preparation."""

import math
import re

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

CHUNKING_STRATEGY = settings.chunking_strategy
EMBEDDING_MODEL = settings.embedding_model


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
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    similarity_threshold: float | None = None,
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
    strategy = (strategy or settings.chunking_strategy).lower()
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    similarity_threshold = (
        settings.semantic_similarity_threshold
        if similarity_threshold is None
        else similarity_threshold
    )
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
            from ingestion.embed import get_embedding_model

            embedding_model = get_embedding_model()
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
