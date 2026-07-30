# ============================================================
# rag_pipeline.py — Core RAG Logic
# ============================================================
# This file wires together:
#   • The FAISS retriever  (loads the index built by ingest.py)
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

import os  # file-system path helpers

# --- LangChain core ---
from langchain_core.runnables import RunnablePassthrough      # passthrough in LCEL pipes
from langchain_core.output_parsers import StrOutputParser      # extracts plain-text from LLM output
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools.retriever import create_retriever_tool

# --- LangChain agents & tools ---
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferMemory

# --- LangChain integrations ---
from langchain_community.vectorstores import FAISS             # FAISS vector store wrapper
from langchain_huggingface import HuggingFaceEmbeddings        # same embedder as ingest.py
from langchain_community.tools import DuckDuckGoSearchRun

# --- Shared LLM factory ---
from llm_factory import build_llm

# --- Memory manager (our custom module) ---
from memory_manager import MemoryManager


# ── PATHS ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Must match the path used in ingest.py so we load the correct index
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")


# ── LOAD SHARED RESOURCES (run once at import time) ──────────

# Embedding model — must be the SAME model used during ingestion,
# otherwise the vector dimensions won't match and retrieval will
# return garbage.
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
)

# Load the FAISS index that ingest.py created and saved to disk.
# allow_dangerous_deserialization=True is required because FAISS
# indexes are stored as pickle files — LangChain asks you to
# explicitly opt in to loading them.
vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embedding_model,
    allow_dangerous_deserialization=True,
)

# Build a retriever from the vector store.
# search_kwargs={"k": 4} means "return the 4 most similar chunks"
# for every query.  Tune k up for broader context, down for speed.
retriever = vectorstore.as_retriever(
    search_type="similarity",       # cosine similarity (default)
    search_kwargs={"k": 4},         # number of chunks to retrieve
)

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
            "Search the uploaded PDF documents (Aerux annual reports) for "
            "relevant excerpts. Use this first when the question may be "
            "answered from the user's documents. Input should be a search query."
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