"""Cross-encoder reranking of fused retrieval candidates."""

import threading

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import settings


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        if not documents:
            return []
        scores = self._get_model().predict(
            [(query, document.page_content) for document in documents]
        )
        if len(scores) != len(documents):
            raise RuntimeError("Cross-encoder returned a different number of scores than documents")
        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        results = []
        for document, score in ranked[:top_k]:
            metadata = dict(document.metadata)
            metadata["rerank_score"] = float(score)
            results.append(Document(page_content=document.page_content, metadata=metadata))
        return results
