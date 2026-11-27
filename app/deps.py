"""Cached factories for models and application services."""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from generation.llm import build_llm
from retrieval.rerank import CrossEncoderReranker


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_llm():
    return build_llm()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=settings.reranker_model)


@lru_cache(maxsize=1)
def get_memory_manager():
    from memory_manager import MemoryManager

    return MemoryManager(llm=get_llm(), embeddings=get_embeddings())
