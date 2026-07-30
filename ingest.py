# ============================================================
# ingest.py — Document Ingestion Pipeline
# ============================================================
# This file handles the full ingestion workflow:
#   1. Load PDF files from the data/ folder using pdfplumber
#   2. Split them into smaller chunks  (YOUR TASK)
#   3. Embed chunks and store them in a FAISS vector store  (YOUR TASK)
#
# Run this file once (or whenever you add new PDFs):
#     python ingest.py
# ============================================================

import os                              # file-system path helpers
import pdfplumber                      # PDF text extraction library
from langchain_core.documents import Document  # LangChain's universal document wrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- Community integrations for embeddings + vector store ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── PATHS ────────────────────────────────────────────────────
# __file__ resolves to this script's location so paths work
# regardless of where you run the command from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder that holds your Aerux reports PDFs
DATA_DIR = os.path.join(BASE_DIR, "data")

# Folder where the FAISS index will be persisted to disk
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")


# ── STEP 1: PDF LOADING ─────────────────────────────────────
def load_pdfs(data_dir: str) -> list[Document]:
    """
    Walk through every PDF in `data_dir`, extract text page-by-page
    using pdfplumber, and wrap each page's text in a LangChain
    Document object with metadata (source filename + page number).

    Returns a flat list of Document objects — one per page.
    """
    documents = []  # accumulator for all page-level Documents

    # os.listdir returns every file name in the directory
    for filename in os.listdir(data_dir):

        # Skip non-PDF files (e.g. .DS_Store, desktop.ini, etc.)
        if not filename.lower().endswith(".pdf"):
            continue

        # Build absolute path to the PDF
        filepath = os.path.join(data_dir, filename)
        print(f"Loading: {filename}")

        # pdfplumber.open() gives a context manager for safe file handling
        with pdfplumber.open(filepath) as pdf:

            # Iterate over every page in the PDF
            for page_num, page in enumerate(pdf.pages, start=1):

                # .extract_text() pulls all the text from one page
                text = page.extract_text()

                # Some pages may be pure images/charts — skip blanks
                if text and text.strip():
                    # Wrap the text in a LangChain Document, attaching
                    # metadata so we can cite sources later.
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": filename,       # which PDF
                            "page": page_num,         # which page
                        },
                    )
                    documents.append(doc)

    print(f"Loaded {len(documents)} pages from {data_dir}")
    return documents


# ── STEP 2: TEXT SPLITTING / CHUNKING ────────────────────────
def split_documents(documents: list[Document]) -> list[Document]:
    """
    Take the page-level Documents and split them into smaller,
    overlapping chunks suitable for embedding and retrieval.

    Returns a list of chunked Documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = text_splitter.split_documents(documents)
    return chunks


# ── STEP 3: EMBEDDING + VECTOR STORE CREATION ───────────────
def create_vectorstore(chunks: list[Document]) -> None:
    """
    Embed every chunk using the all-MiniLM-L6-v2 sentence-transformer
    and persist the resulting FAISS index to the vectorstore/ folder.
    """
    # Instantiate the HuggingFace embedding model.
    # "all-MiniLM-L6-v2" is a lightweight, fast, high-quality model
    # that produces 384-dimensional embeddings.
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(VECTORSTORE_DIR)
    print(f"vector store created and saved to {VECTORSTORE_DIR}")
    return vector_store


# ── MAIN — run the full pipeline ─────────────────────────────
if __name__ == "__main__":
    # Ensure the output directory exists before we try to write there
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    # Step 1 — load raw pages from PDFs
    raw_docs = load_pdfs(DATA_DIR)

    # Step 2 — split pages into smaller chunks
    chunks = split_documents(raw_docs)

    # Step 3 — embed chunks and save the FAISS index
    create_vectorstore(chunks)

    print("Ingestion complete! Vector store saved to:", VECTORSTORE_DIR)
