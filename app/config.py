"""Environment-backed application settings and generated-data paths."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = PROJECT_ROOT / "data"
    raw_documents_dir: Path = PROJECT_ROOT / "data" / "raw_documents"
    processed_documents_path: Path = PROJECT_ROOT / "data" / "processed_documents.jsonl"
    chroma_dir: Path = PROJECT_ROOT / "chroma"
    bm25_dir: Path = PROJECT_ROOT / "bm25"
    memory_dir: Path = PROJECT_ROOT / "memory"
    index_manifest_path: Path = PROJECT_ROOT / "index_manifest.json"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: str = "ollama"
    llm_model: str = "gemma2:9b"
    chunking_strategy: str = "fixed"
    chunk_size: int = 600
    chunk_overlap: int = 120
    semantic_similarity_threshold: float = 0.55
    dedup_similarity_threshold: float = 0.95
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_candidate_k: int = 20
    reranked_top_k: int = 5
    rrf_constant: int = 60
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    retrieval_confidence_threshold: float = 0.35
    confidence_retrieval_weight: float = 0.4
    confidence_citation_weight: float = 0.35
    confidence_completeness_weight: float = 0.25


def load_settings() -> Settings:
    """Read settings from environment variables, retaining documented defaults."""
    settings = Settings(
        data_dir=Path(os.getenv("DOCFORGE_DATA_DIR", str(PROJECT_ROOT / "data"))),
        raw_documents_dir=Path(
            os.getenv("DOCFORGE_RAW_DOCUMENTS_DIR", str(PROJECT_ROOT / "data" / "raw_documents"))
        ),
        processed_documents_path=Path(
            os.getenv(
                "DOCFORGE_PROCESSED_DOCUMENTS_PATH",
                str(PROJECT_ROOT / "data" / "processed_documents.jsonl"),
            )
        ),
        chroma_dir=Path(os.getenv("DOCFORGE_CHROMA_DIR", str(PROJECT_ROOT / "chroma"))),
        bm25_dir=Path(os.getenv("DOCFORGE_BM25_DIR", str(PROJECT_ROOT / "bm25"))),
        memory_dir=Path(os.getenv("DOCFORGE_MEMORY_DIR", str(PROJECT_ROOT / "memory"))),
        index_manifest_path=Path(
            os.getenv("DOCFORGE_INDEX_MANIFEST_PATH", str(PROJECT_ROOT / "index_manifest.json"))
        ),
        embedding_model=os.getenv("DOCFORGE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        reranker_model=os.getenv(
            "DOCFORGE_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        llm_provider=os.getenv("DOCFORGE_LLM_PROVIDER", "ollama").lower(),
        llm_model=os.getenv("DOCFORGE_LLM_MODEL", "gemma2:9b"),
        chunking_strategy=os.getenv("DOCFORGE_CHUNKING_STRATEGY", "fixed").lower(),
        chunk_size=_env_int("DOCFORGE_CHUNK_SIZE", 600),
        chunk_overlap=_env_int("DOCFORGE_CHUNK_OVERLAP", 120),
        semantic_similarity_threshold=_env_float(
            "DOCFORGE_SEMANTIC_SIMILARITY_THRESHOLD", 0.55
        ),
        dedup_similarity_threshold=_env_float("DOCFORGE_DEDUP_THRESHOLD", 0.95),
        dense_top_k=_env_int("DOCFORGE_DENSE_TOP_K", 10),
        sparse_top_k=_env_int("DOCFORGE_SPARSE_TOP_K", 10),
        rrf_candidate_k=_env_int("DOCFORGE_RRF_CANDIDATE_K", 20),
        reranked_top_k=_env_int("DOCFORGE_RERANKED_TOP_K", 5),
        rrf_constant=_env_int("DOCFORGE_RRF_CONSTANT", 60),
        dense_weight=_env_float("DOCFORGE_DENSE_WEIGHT", 0.7),
        sparse_weight=_env_float("DOCFORGE_SPARSE_WEIGHT", 0.3),
        retrieval_confidence_threshold=_env_float(
            "DOCFORGE_RETRIEVAL_CONFIDENCE_THRESHOLD", 0.35
        ),
        confidence_retrieval_weight=_env_float("DOCFORGE_CONFIDENCE_RETRIEVAL_WEIGHT", 0.4),
        confidence_citation_weight=_env_float("DOCFORGE_CONFIDENCE_CITATION_WEIGHT", 0.35),
        confidence_completeness_weight=_env_float(
            "DOCFORGE_CONFIDENCE_COMPLETENESS_WEIGHT", 0.25
        ),
    )
    if settings.chunk_size <= 0:
        raise ValueError("DOCFORGE_CHUNK_SIZE must be positive")
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("DOCFORGE_CHUNK_OVERLAP must be non-negative and smaller than chunk size")
    if settings.dense_weight < 0 or settings.sparse_weight < 0:
        raise ValueError("Dense and sparse retrieval weights must be non-negative")
    if settings.dense_weight + settings.sparse_weight <= 0:
        raise ValueError("At least one dense/sparse retrieval weight must be positive")
    if not 0 <= settings.retrieval_confidence_threshold <= 1:
        raise ValueError("DOCFORGE_RETRIEVAL_CONFIDENCE_THRESHOLD must be between 0 and 1")
    if settings.reranked_top_k <= 0 or settings.rrf_candidate_k <= 0:
        raise ValueError("Candidate and reranked result limits must be positive")
    confidence_weights = (
        settings.confidence_retrieval_weight,
        settings.confidence_citation_weight,
        settings.confidence_completeness_weight,
    )
    if any(weight < 0 for weight in confidence_weights) or sum(confidence_weights) <= 0:
        raise ValueError("Confidence weights must be non-negative and not all zero")
    return settings


settings = load_settings()
