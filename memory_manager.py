# ============================================================
# memory_manager.py — Unified Memory Manager (Modern API)
# ============================================================
# This file manages THREE distinct types of conversational memory
# WITHOUT using the deprecated langchain.memory module.
#
# Instead we use:
#   • ChatMessageHistory   (langchain_core)  — raw message buffer
#   • Manual LLM summary   (LCEL chain)      — compressed history
#   • FAISS direct          (langchain_community) — episodic vector store
#
# ┌──────────────────────────────────────────────────────────┐
# │ 1. CONVERSATION BUFFER  (a.k.a. "Context Memory")       │
# │    ──────────────────────────────────────────────────     │
# │    WHAT:  Stores the raw chat messages (human + AI)      │
# │           word-for-word in a sliding window of the last  │
# │           k exchanges.                                   │
# │    WHY:   So the bot "remembers" what was just said and  │
# │           can follow up naturally ("you said X earlier"). │
# │    WHEN:  Use for SHORT conversations (< ~10 turns).     │
# │           Cheap, fast, no LLM calls needed.              │
# │    HOW:   ChatMessageHistory from langchain_core stores  │
# │           messages in a plain list.  We manually slice   │
# │           to the last k turns in get_chat_messages().    │
# │                                                          │
# │ 2. SUMMARY MEMORY  (a.k.a. "Semantic Memory")           │
# │    ──────────────────────────────────────────────────     │
# │    WHAT:  After the conversation exceeds a threshold the │
# │           LLM compresses older turns into a concise      │
# │           running summary.                               │
# │    WHY:   Long conversations would overflow the LLM's    │
# │           context window if kept raw.  A summary retains │
# │           the key facts while using far fewer tokens.    │
# │    WHEN:  Auto-activate after conversation exceeds a     │
# │           configurable threshold (e.g. 10 turns).        │
# │    HOW:   A simple LCEL chain (prompt → LLM → parser)   │
# │           generates the summary on demand.  No special   │
# │           LangChain memory class needed.                 │
# │                                                          │
# │ 3. EPISODIC MEMORY  (a.k.a. "Long-term / Vector Memory")│
# │    ──────────────────────────────────────────────────     │
# │    WHAT:  Each exchange is embedded and stored in a FAISS│
# │           index, so past conversations can be retrieved  │
# │           by semantic similarity.                        │
# │    WHY:   Enables questions like "what did I ask about   │
# │           revenue earlier?" — even from many turns ago.  │
# │    WHEN:  Always running in background; every turn is    │
# │           silently stored.  Retrieved on demand.         │
# │    HOW:   We manage a dedicated FAISS index in memory/,  │
# │           storing each exchange as a Document and using  │
# │           similarity_search() to retrieve.               │
# └──────────────────────────────────────────────────────────┘
#
# PUBLIC API (used by rag_pipeline.py and app.py):
#   • get_chat_messages()          → list of BaseMessage objects
#   • get_summary()                → str (compressed history)
#   • get_episodic_context(query)  → str (relevant past exchanges)
#   • record_exchange(human, ai)   → None (saves to all stores)
#   • reset()                      → None (clears session state)
# ============================================================

import os  # file-system helpers for persistence paths

# --- LangChain core: modern message & history primitives ---
from langchain_core.chat_history import InMemoryChatMessageHistory  # simple in-memory message store
from langchain_core.messages import HumanMessage, AIMessage         # typed message wrappers
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- LangChain integrations ---
from langchain_huggingface import HuggingFaceEmbeddings    # sentence-transformer embeddings
from langchain_community.vectorstores import FAISS                  # FAISS vector store wrapper                        # universal document wrapper

# --- Shared LLM factory ---
from generation.llm import build_llm
from app.config import settings


# ── PATHS ────────────────────────────────────────────────────
# Dedicated folder for the episodic-memory FAISS index
# (separate from the document vectorstore in vectorstore/)
MEMORY_DIR = str(settings.memory_dir)


class MemoryManager:
    """
    Central hub that owns all three memory backends and exposes
    a clean public API used by rag_pipeline.py and app.py.

    No deprecated langchain.memory imports — everything is built
    from ChatMessageHistory, plain LCEL chains, and direct FAISS.
    """

    # Number of conversation turns before we switch from raw
    # buffer messages to LLM-compressed summaries.
    SUMMARY_THRESHOLD = 10

    # How many recent exchanges the buffer window keeps.
    # Each "exchange" = 1 HumanMessage + 1 AIMessage = 2 messages,
    # so BUFFER_K=5 means the last 10 messages are retained.
    BUFFER_K = 5

    def __init__(self, llm=None, embeddings=None):
        """
        Initialize all memory backends.
        The active LLM is needed by summary memory to generate
        compressed summaries.
        """
        # Ensure the memory persistence directory exists
        os.makedirs(MEMORY_DIR, exist_ok=True)

        # --- Shared resources needed by multiple memory types ---

        # LLM used by summary memory to compress conversations.
        # Keep this in sync with the main chain by using the
        # shared provider factory.
        self.llm = llm or build_llm()

        # Embedding model used by episodic memory to encode exchanges.
        # Same model as ingest.py so representations are compatible.
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
        )

        # --- Internal turn counter ---
        # Tracks how many exchanges have happened in this session.
        # Used by get_chat_messages() to decide buffer vs summary.
        self._turn_count = 0

        # ── Initialize the 3 memory backends ────────────────
        self.chat_history = self._init_chat_history()
        self._summary_cache = ""  # stores the latest compressed summary
        self.episodic_store = self._init_episodic_store()

    # ─────────────────────────────────────────────────────────
    # MEMORY 1: Conversation Buffer (ChatMessageHistory)
    # ─────────────────────────────────────────────────────────
    def _init_chat_history(self):
        """
        Create an InMemoryChatMessageHistory instance.
        This is a simple list-backed store of BaseMessage objects —
        no legacy wrappers, no memory_key gymnastics.
        """
        return InMemoryChatMessageHistory()

    # ─────────────────────────────────────────────────────────
    # MEMORY 2: Summary Memory (LCEL chain, no legacy class)
    # ─────────────────────────────────────────────────────────
    def _generate_summary(self, conversation_text: str) -> str:
        """
        Use the LLM to compress `conversation_text` into a
        concise running summary.

        Args:
            conversation_text: A formatted string of the full
                conversation so far (Human: … / AI: … pairs).

        Returns:
            A short summary string produced by the LLM.
        """
        prompt = PromptTemplate(
            template="Summarize the following conversation without excluding any important details:\n{conversation_text}"
        )
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"conversation_text": conversation_text})
        return summary

    # ─────────────────────────────────────────────────────────
    # MEMORY 3: Episodic Memory (FAISS vector store)
    # ─────────────────────────────────────────────────────────
    def _init_episodic_store(self):
        """
        Load an existing FAISS index from the memory/ folder, or
        create a brand-new empty one if none exists yet.

        Returns a LangChain FAISS vectorstore instance.
        """
        index_file = os.path.join(MEMORY_DIR, "index.faiss")
        store_file = os.path.join(MEMORY_DIR, "index.pkl")

        if os.path.exists(index_file) and os.path.exists(store_file):
            return FAISS.load_local(
                MEMORY_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            placeholder_doc = Document(page_content="This is a placeholder for the episodic memory.")
            vectorstore = FAISS.from_documents([placeholder_doc], self.embeddings)
            vectorstore.save_local(MEMORY_DIR)
            return vectorstore

    # ═════════════════════════════════════════════════════════
    #  PUBLIC API — used by rag_pipeline.py and app.py
    # ═════════════════════════════════════════════════════════

    def get_chat_messages(self) -> list:
        """
        Return the chat messages that should be injected into
        the prompt's {chat_history} placeholder.

        Behavior:
        - If turn_count < SUMMARY_THRESHOLD: return the last
          BUFFER_K exchanges (2*K messages) from chat_history.
        - If turn_count >= SUMMARY_THRESHOLD: return a single
          SystemMessage containing the compressed summary so
          the raw history doesn't overflow the context window.

        Returns:
            A list of BaseMessage objects (HumanMessage,
            AIMessage, or SystemMessage).
        """
        if self._turn_count < self.SUMMARY_THRESHOLD:
            # --- Buffer mode: return last K exchanges ---
            # Each exchange is 2 messages (Human + AI), so
            # we slice the last 2*K messages from the list.
            all_messages = self.chat_history.messages
            window_size = self.BUFFER_K * 2  # 2 messages per exchange
            return all_messages[-window_size:]  # slice is safe even if list is shorter
        else:
            # --- Summary mode: return compressed summary ---
            print(
                f"Conversation exceeded {self.SUMMARY_THRESHOLD} turns. "
                "Returning summary instead of raw history."
            )
            # Import here to avoid circular issues at module level
            from langchain_core.messages import SystemMessage
            return [SystemMessage(content=f"Conversation summary so far:\n{self._summary_cache}")]

    def get_episodic_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve the k most semantically relevant past exchanges
        from the episodic (vector) memory, given a query.

        This lets the bot answer questions like "what did I ask
        about revenue earlier?" by searching across ALL past turns
        (not just the current buffer window).

        Args:
            query: The search string (usually the user's question).
            k:     Number of past exchanges to retrieve.

        Returns:
            A formatted string of relevant past exchanges,
            or an empty string if episodic memory isn't ready.
        """
        try:
            # similarity_search returns a list of Document objects
            # whose page_content holds the stored exchange text.
            docs = self.episodic_store.similarity_search(query, k=k)

            # Filter out the placeholder seed document (if present)
            relevant = [d for d in docs if d.page_content != "[episodic memory initialized]"]

            if not relevant:
                return ""

            # Format the retrieved exchanges into a readable block
            return "\n---\n".join(doc.page_content for doc in relevant)

        except (NotImplementedError, AttributeError):
            print(
                "Episodic memory not implemented yet — skipping retrieval. "
                "Complete the placeholder in _init_episodic_store()."
            )
            return ""

    def record_exchange(self, human_input: str, ai_output: str) -> None:
        """
        Record a single Q&A exchange into ALL memory backends.
        This is the ONE method that rag_pipeline.ask() calls after
        every turn — it replaces the old save_to_vector_memory()
        and manual save_context() calls.

        Steps:
          1. Append HumanMessage + AIMessage to chat_history
          2. Increment turn counter
          3. If over threshold, regenerate the running summary
          4. Embed the exchange into episodic FAISS store

        Args:
            human_input: The user's message.
            ai_output:   The assistant's response.
        """
        # --- 1. Append to the raw chat history buffer ---
        try:
            self.chat_history.add_message(HumanMessage(content=human_input))
            self.chat_history.add_message(AIMessage(content=ai_output))
        except NotImplementedError:
            print(
                "Chat history not implemented yet — skipping buffer save. "
                "Complete the placeholder in _init_chat_history()."
            )

        # --- 2. Increment turn counter ---
        self._turn_count += 1

        # --- 3. Update summary if we've crossed the threshold ---
        if self._turn_count >= self.SUMMARY_THRESHOLD:
            try:
                # Build a plain-text transcript from all messages
                all_msgs = self.chat_history.messages
                transcript = "\n".join(
                    f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                    for m in all_msgs
                )
                self._summary_cache = self._generate_summary(transcript)
            except NotImplementedError:
                print(
                    "Summary generation not implemented yet — skipping. "
                    "Complete the placeholder in _generate_summary()."
                )

        # --- 4. Store in episodic (vector) memory ---
        try:
            # Combine human + AI into a single Document for embedding.
            # This way, similarity search can match on either the
            # question or the answer text.
            exchange_text = f"Human: {human_input}\nAI: {ai_output}"
            doc = Document(
                page_content=exchange_text,
                metadata={"turn": self._turn_count},
            )

            # Add the document to the existing FAISS index
            self.episodic_store.add_documents([doc])

            # Persist the updated index to disk so it survives restarts
            self.episodic_store.save_local(MEMORY_DIR)
        except NotImplementedError:
            print(
                "Episodic memory not implemented yet — skipping save. "
                "Complete the placeholder in _init_episodic_store()."
            )

    # ── CONVENIENCE ──────────────────────────────────────────
    def get_turn_count(self) -> int:
        """Return the number of exchanges so far in this session."""
        return self._turn_count

    def reset(self) -> None:
        """
        Clear all in-memory state for a fresh session.
        (Episodic memory on disk is NOT erased — that's long-term.)
        """
        self._turn_count = 0
        self._summary_cache = ""

        # Clear the in-memory chat history buffer
        try:
            self.chat_history.clear()
        except (NotImplementedError, AttributeError):
            pass

        print("Session memory cleared.")
