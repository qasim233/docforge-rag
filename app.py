# ============================================================
# app.py — Gradio Chat UI
# ============================================================
# This file creates a web-based chat interface using Gradio.
# It connects the UI to the RAG pipeline so users can:
#   • Ask questions about their Aerux annual reports
#   • Get AI-powered answers grounded in the PDF documents
#   • Have a multi-turn conversation with memory
#
# Run with:  python app.py
# Then open http://localhost:7860 in your browser.
# ============================================================

import gradio as gr                         # web UI framework

# Import the public API from our RAG pipeline module
from rag_pipeline import ask, reset_memory


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
    title="Aerux Annual Report Chatbot",
    description="Ask questions about Aerux's annual reports. The AI will provide answers grounded in the PDF documents.",
).launch()