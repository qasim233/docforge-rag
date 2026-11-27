# ============================================================
# rag_pipeline.py — Core RAG Logic
# ============================================================
# This file wires together:
#   • Hybrid retrieval    (Chroma vectors + BM25 lexical search)
#   • The active LLM provider (default: local Ollama/gemma2)
#   • The MemoryManager    (buffer / summary / episodic)
#   • A prompt template    (YOUR TASK)
#   • An LCEL chain        (YOUR TASK)
#   • A LangChain agent with DuckDuckGo search  (YOUR TASK)
#
# Memory integration uses the NEW public API:
#   memory_manager.get_chat_messages()         → list[BaseMessage]
#   memory_manager.get_episodic_context(query)  → str
#   memory_manager.record_exchange(human, ai)   → None
#
# It exposes a single `ask(question)` function used by app.py.
# ============================================================

import hashlib
import json
import os
import threading
from typing import Any

# --- LangChain core ---
from langchain_core.runnables import RunnablePassthrough      # passthrough in LCEL pipes
from langchain_core.output_parsers import StrOutputParser      # extracts plain-text from LLM output
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import PrivateAttr

# --- LangChain agents & tools ---
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferMemory

# --- LangChain integrations ---
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from sentence_transformers import CrossEncoder

# --- Shared LLM factory ---
from llm_factory import build_llm

# --- Memory manager (our custom module) ---
from memory_manager import MemoryManager


# ── PATHS ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DIR = os.path.join(BASE_DIR, "chroma")
INDEX_MANIFEST_PATH = os.path.join(BASE_DIR, "index_manifest.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
RRF_CANDIDATE_K = 20
RERANKED_TOP_K = 5
RRF_CONSTANT = 60
RERANKER_MODEL = os.getenv(
    "DOCFORGE_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


def _read_weight(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


DENSE_WEIGHT = _read_weight("DOCFORGE_DENSE_WEIGHT", 0.7)
SPARSE_WEIGHT = _read_weight("DOCFORGE_SPARSE_WEIGHT", 0.3)
if DENSE_WEIGHT + SPARSE_WEIGHT <= 0:
    raise ValueError("At least one RRF retrieval weight must be greater than zero")


def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
    rrf_constant: int = RRF_CONSTANT,
    candidate_k: int = RRF_CANDIDATE_K,
) -> list[Document]:
    """Merge ranked result lists by weighted reciprocal-rank contribution."""
    if dense_weight < 0 or sparse_weight < 0 or dense_weight + sparse_weight <= 0:
        raise ValueError("RRF weights must be non-negative and not both zero")
    if rrf_constant < 0 or candidate_k <= 0:
        raise ValueError("rrf_constant must be non-negative and candidate_k positive")

    fused: dict[str, tuple[Document, float]] = {}
    for results, weight in (
        (dense_results, dense_weight),
        (sparse_results, sparse_weight),
    ):
        for rank, document in enumerate(results, start=1):
            key = str(
                document.metadata.get("chunk_id")
                or hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
            )
            previous = fused.get(key)
            score = weight / (rrf_constant + rank)
            if previous is None:
                fused[key] = (document, score)
            else:
                fused[key] = (previous[0], previous[1] + score)

    ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)
    candidates: list[Document] = []
    for document, score in ranked[:candidate_k]:
        metadata = dict(document.metadata)
        metadata["rrf_score"] = score
        candidates.append(Document(page_content=document.page_content, metadata=metadata))
    return candidates


class HybridRerankerRetriever(BaseRetriever):
    """Fuse dense and sparse rankings, then rerank their candidates for the query."""

    dense_retriever: Any
    sparse_retriever: Any
    dense_weight: float = DENSE_WEIGHT
    sparse_weight: float = SPARSE_WEIGHT
    candidate_k: int = RRF_CANDIDATE_K
    final_k: int = RERANKED_TOP_K
    reranker_model: str = RERANKER_MODEL

    _cross_encoder: CrossEncoder | None = PrivateAttr(default=None)
    _reranker_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            with self._reranker_lock:
                if self._cross_encoder is None:
                    self._cross_encoder = CrossEncoder(self.reranker_model)
        return self._cross_encoder

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        dense_results = self.dense_retriever.invoke(query)
        sparse_results = self.sparse_retriever.invoke(query)
        candidates = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            candidate_k=self.candidate_k,
        )
        if not candidates:
            return []

        model = self._get_cross_encoder()
        scores = model.predict(
            [(query, document.page_content) for document in candidates]
        )
        reranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        final_documents: list[Document] = []
        for document, score in reranked[: self.final_k]:
            metadata = dict(document.metadata)
            metadata["rerank_score"] = float(score)
            final_documents.append(
                Document(page_content=document.page_content, metadata=metadata)
            )
        return final_documents


# ── LOAD SHARED RESOURCES (run once at import time) ──────────

def _load_hybrid_retrievers():
    if not os.path.isfile(INDEX_MANIFEST_PATH):
        raise FileNotFoundError(
            "Document indexes are missing; run ingest.py to build Chroma and BM25."
        )

    with open(INDEX_MANIFEST_PATH, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest["embedding_model"] != EMBEDDING_MODEL:
        raise RuntimeError(
            "The stored index uses a different embedding model; run ingest.py to rebuild it."
        )
    if manifest.get("distance_metric") != "cosine":
        raise RuntimeError(
            "The stored Chroma index is not configured for cosine distance; "
            "run ingest.py to rebuild it."
        )
    snapshot_path = os.path.join(BASE_DIR, manifest["bm25_snapshot"])
    records = []
    with open(snapshot_path, encoding="utf-8") as snapshot:
        for line in snapshot:
            if line.strip():
                records.append(json.loads(line))

    fingerprint = hashlib.sha256(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            for record in records
        ).encode("utf-8")
    ).hexdigest()
    if len(records) != manifest["chunk_count"] or fingerprint != manifest["fingerprint"]:
        raise RuntimeError("The BM25 snapshot does not match the active index manifest")

    documents = [
        Document(page_content=record["page_content"], metadata=record["metadata"])
        for record in records
    ]
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=manifest["collection_name"],
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    chroma_ids = set(vectorstore._collection.get(include=["metadatas"])["ids"])
    bm25_ids = {record["id"] for record in records}
    if chroma_ids != bm25_ids:
        raise RuntimeError("Chroma and BM25 indexes are out of sync; rebuild with ingest.py")

    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": DENSE_TOP_K},
    )
    sparse_retriever = BM25Retriever.from_documents(documents, k=SPARSE_TOP_K)
    hybrid_retriever = HybridRerankerRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=DENSE_WEIGHT,
        sparse_weight=SPARSE_WEIGHT,
        candidate_k=RRF_CANDIDATE_K,
        final_k=RERANKED_TOP_K,
        reranker_model=RERANKER_MODEL,
    )
    return dense_retriever, sparse_retriever, hybrid_retriever


dense_retriever, sparse_retriever, retriever = _load_hybrid_retrievers()

# Instantiate the active LLM provider.
# Default is local Ollama. To switch providers, edit llm_factory.py.
llm = build_llm()

# Instantiate our custom memory manager (buffer + summary + vector)
memory_manager = MemoryManager()


# ── HELPER: format retrieved docs into a single string ───────
def format_docs(docs) -> str:
    """
    Take a list of LangChain Document objects returned by the
    retriever and concatenate their page_content fields into
    one big context string, separated by double newlines.

    This string is injected into the prompt as {context}.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# ── STEP 1: PROMPT TEMPLATE ─────────────────────────────────
def get_prompt_template():
    """
    Build and return a ChatPromptTemplate that tells the LLM:
      • Its role (system message)
      • The retrieved document context  → {context}
      • The conversation history        → {chat_history}  (list of BaseMessages)
      • Relevant past exchanges         → {episodic_context}  (plain string)
      • The user's current question     → {question}

    The template must accept these 4 variables because
    build_chain() will assign them via RunnablePassthrough.

    Returns a ChatPromptTemplate instance.
    """
    template = ChatPromptTemplate.from_messages([
        ("system", "Retrieved context:\n{context}\n\nRelevant past exchanges:\n{episodic_context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    return template


# ── STEP 2: LCEL CONVERSATIONAL RETRIEVAL CHAIN ─────────────
def build_chain():
    """
    Construct the LCEL (LangChain Expression Language) chain that:
      1. Retrieves relevant document chunks  → {context}
      2. Fetches chat messages from memory   → {chat_history}
      3. Fetches episodic context            → {episodic_context}
      4. Sends everything through prompt → LLM → output parser

    The chain expects input: {"question": "..."}.
    Returns an LCEL Runnable.
    """
    run = RunnablePassthrough()
    run = run.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"])),
        chat_history=lambda x: memory_manager.get_chat_messages(),
        episodic_context=lambda x: memory_manager.get_episodic_context(x["question"]),
    )
    
    chain = run | get_prompt_template() | llm | StrOutputParser()
    return chain


# ── STEP 3: AGENT WITH DUCKDUCKGO TOOL ──────────────────────
def build_agent():
    """
    Create a LangChain agent that has access to:
      • The RAG retriever  (searches your documents)
      • DuckDuckGo web search  (for questions not in your PDFs)
      • Conversation memory

    Returns an AgentExecutor instance.
    """
    retriever_tool = create_retriever_tool(
        retriever,
        "search_documents",
        (
            "Search the uploaded documents for relevant excerpts. Use this first "
            "when the question may be answered from the user's documents. "
            "Input should be a search query."
        ),
    )

    web_search = DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Search the web for up-to-date information not found in the "
            "uploaded documents. Input should be a search query."
        ),
    )

    tools = [retriever_tool, web_search]

    react_prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, react_prompt)

    memory = ConversationBufferMemory(
        chat_memory=memory_manager.chat_history,
        memory_key="chat_history",
        return_messages=True,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor


# ── PUBLIC API ───────────────────────────────────────────────
def ask(question: str) -> str:
    """
    High-level entry point called by app.py.

    Sends the user's question through the RAG chain (or agent),
    records the exchange in ALL memory backends via the unified
    record_exchange() method, and returns the answer.

    Args:
        question: The user's natural-language question.

    Returns:
        The assistant's answer string.
    """
    try:
        # --- Try the LCEL chain first ---
        chain = build_chain()

        # Invoke the chain with the user's question.
        # The chain internally fetches context, chat history,
        # and episodic context via the MemoryManager.
        answer = chain.invoke({"question": question})

    except NotImplementedError:
        # If the chain isn't implemented yet, fall back to a
        # raw LLM call so the app still "works" while you're
        # building out the placeholders.
        print(
            "LCEL chain not implemented — falling back to raw LLM. "
            "Complete the placeholders in rag_pipeline.py."
        )
        answer = llm.invoke(question)

    # --- Record the exchange in ALL memory backends at once ---
    # record_exchange() handles:
    #   1. Appending HumanMessage + AIMessage to chat_history buffer
    #   2. Incrementing the turn counter
    #   3. Regenerating the summary if over threshold
    #   4. Embedding the exchange into episodic FAISS store
    memory_manager.record_exchange(question, answer)

    return answer


def reset_memory() -> None:
    """
    Clear all session memory.  Called by app.py when the user
    clicks "New Chat" or refreshes the page.
    """
    memory_manager.reset()