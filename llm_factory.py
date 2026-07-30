# ============================================================
# llm_factory.py — Shared LLM provider selection
# ============================================================
# Default to local Ollama for offline use.
# If you want a hosted provider instead, comment out the
# Ollama block below and uncomment one of the example blocks.
# ============================================================

import os


def build_llm():
    """
    Build the active LLM client used by the project.

    Default: local Ollama
    Swap-in examples are left commented so other users can
    replace the provider without changing the rest of the code.
    """
    # --- Default: Ollama (local) ---
    from langchain_ollama import OllamaLLM

    return OllamaLLM(model="gemma2:9b")

    # --- Example: Grok / xAI (OpenAI-compatible) ---
    # from langchain_openai import ChatOpenAI
    # return ChatOpenAI(
    #     model="grok-2-latest",
    #     api_key=os.getenv("XAI_API_KEY"),
    #     base_url="https://api.x.ai/v1",
    # )

    # --- Example: OpenAI ---
    # from langchain_openai import ChatOpenAI
    # return ChatOpenAI(
    #     model="gpt-4o-mini",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # )

    # --- Example: Anthropic ---
    # from langchain_anthropic import ChatAnthropic
    # return ChatAnthropic(
    #     model="claude-3-5-sonnet-latest",
    #     api_key=os.getenv("ANTHROPIC_API_KEY"),
    # )