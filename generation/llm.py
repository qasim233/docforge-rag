"""LLM provider construction used by answers and conversation memory."""

from app.config import settings


def build_llm():
    if settings.llm_provider == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(model=settings.llm_model)
    raise ValueError(f"Unsupported DOCFORGE_LLM_PROVIDER: {settings.llm_provider!r}")
