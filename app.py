# ============================================================
# app.py — Gradio Chat UI
# ============================================================
# This file creates a web-based chat interface using Gradio.
# It connects the UI to the RAG pipeline so users can:
#   • Ask questions about their documents
#   • Get AI-powered answers grounded in retrieved document chunks
#   • Have a multi-turn conversation with memory
#
# Run with:  python app.py
# Then open http://localhost:7860 in your browser.
# ============================================================

import gradio as gr                         # web UI framework

from generation.answer import ask, reset_memory


# ── CHAT HANDLER ─────────────────────────────────────────────
def chat_handler(message: str, history: list) -> str:
    """
    Callback invoked by Gradio every time the user sends a message.

    Args:
        message: The user's latest message (string).
        history: A list of [user_msg, bot_msg] pairs representing
                 the conversation so far (managed by Gradio).

    Returns:
        The assistant's response string, which Gradio displays
        in the chat window.
    """
    # Delegate to the RAG pipeline's ask() function.
    # ask() handles retrieval, memory, and LLM invocation.
    response = ask(message)
    return response


# ── GRADIO UI ────────────────────────────────────────────────
gr.ChatInterface(
    fn=chat_handler,
    title="DocForge Document Assistant",
    description="Ask questions about your documents. Answers include evidence citations and confidence scores.",
).launch()