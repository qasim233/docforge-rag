"""Synchronized dense, BM25, weighted-RRF, and reranked search."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.config import settings
from app.deps import get_embeddings, get_reranker
from retrieval.rerank import CrossEncoderReranker
from storage.vector_store import get_collection_ids, load_vector_collection


def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    rrf_constant: int | None = None,
    candidate_k: int | None = None,
) -> list[Document]:
    """Merge ranked result lists by weighted reciprocal-rank contribution."""
    dense_weight = settings.dense_weight if dense_weight is None else dense_weight
    sparse_weight = settings.sparse_weight if sparse_weight is None else sparse_weight
    rrf_constant = settings.rrf_constant if rrf_constant is None else rrf_constant
    candidate_k = settings.rrf_candidate_k if candidate_k is None else candidate_k
    if dense_weight < 0 or sparse_weight < 0 or dense_weight + sparse_weight <= 0:
        raise ValueError("RRF weights must be non-negative and not both zero")
    if rrf_constant < 0 or candidate_k <= 0:
        raise ValueError("rrf_constant must be non-negative and candidate_k positive")

    fused: dict[str, tuple[Document, float]] = {}
    for results, weight in ((dense_results, dense_weight), (sparse_results, sparse_weight)):
        for rank, document in enumerate(results, start=1):
            key = str(
                document.metadata.get("chunk_id")
                or hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
            )
            previous = fused.get(key)
            score = weight / (rrf_constant + rank)
            fused[key] = (
                document if previous is None else previous[0],
                score if previous is None else previous[1] + score,
            )

    ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)
    candidates = []
    for document, score in ranked[:candidate_k]:
        metadata = dict(document.metadata)
        metadata["rrf_score"] = score
        candidates.append(Document(page_content=document.page_content, metadata=metadata))
    return candidates


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_retriever: Any
    sparse_retriever: Any
    reranker: CrossEncoderReranker

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        dense_results = self.dense_retriever.invoke(query)
        sparse_results = self.sparse_retriever.invoke(query)
        candidates = reciprocal_rank_fusion(dense_results, sparse_results)
        return self.reranker.rerank(query, candidates, settings.reranked_top_k)


def _read_manifest_and_records() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not settings.index_manifest_path.is_file():
        raise FileNotFoundError(
            f"Document indexes are missing; run ingest.py to build Chroma and BM25 "
            f"(manifest: {settings.index_manifest_path})."
        )

    with settings.index_manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest["embedding_model"] != settings.embedding_model:
        raise RuntimeError(
            "The stored index uses a different embedding model; run ingest.py to rebuild it."
        )
    if manifest.get("distance_metric") != "cosine":
        raise RuntimeError("Chroma index is not configured for cosine distance; rebuild it.")

    snapshot_path = (settings.index_manifest_path.parent / manifest["bm25_snapshot"]).resolve()
    if not snapshot_path.is_relative_to(settings.bm25_dir.resolve()):
        raise ValueError("BM25 snapshot path must be inside the configured BM25 directory")
    with snapshot_path.open(encoding="utf-8") as snapshot:
        records = [json.loads(line) for line in snapshot if line.strip()]

    fingerprint = hashlib.sha256(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            for record in records
        ).encode("utf-8")
    ).hexdigest()
    if len(records) != manifest["chunk_count"] or fingerprint != manifest["fingerprint"]:
        raise RuntimeError("The BM25 snapshot does not match the active index manifest")
    return manifest, records


@lru_cache(maxsize=1)
def get_retrievers() -> tuple[Any, Any, HybridRetriever]:
    manifest, records = _read_manifest_and_records()
    documents = [
        Document(page_content=record["page_content"], metadata=record["metadata"])
        for record in records
    ]
    vector_store = load_vector_collection(
        manifest["collection_name"],
        get_embeddings(),
        settings.chroma_dir,
    )
    corpus_ids = {record["id"] for record in records}
    if get_collection_ids(vector_store) != corpus_ids:
        raise RuntimeError("Chroma and BM25 indexes are out of sync; rebuild with ingest.py")

    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.dense_top_k},
    )
    sparse_retriever = BM25Retriever.from_documents(
        documents,
        k=settings.sparse_top_k,
    )
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=get_reranker(),
    )
    return dense_retriever, sparse_retriever, hybrid_retriever
