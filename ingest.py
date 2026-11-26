# ============================================================
# ingest.py — Document Ingestion Pipeline
# ============================================================

import json
import hashlib
import math
import os
import re
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Any

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DOCUMENTS_DIR = os.path.join(DATA_DIR, "raw_documents")
PROCESSED_DOCUMENTS_PATH = os.path.join(DATA_DIR, "processed_documents.jsonl")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma")
BM25_DIR = os.path.join(BASE_DIR, "bm25")
INDEX_MANIFEST_PATH = os.path.join(BASE_DIR, "index_manifest.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}
CHUNKING_STRATEGY = os.getenv("DOCFORGE_CHUNKING_STRATEGY", "fixed").lower()


def _clean_text(text: str) -> str:
    """Normalize whitespace and HTML entities without losing paragraph breaks."""
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Split text into lines, collapse multiple spaces/tabs into one, and trim leading/trailing whitespace
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Collapse 3 or more consecutive newlines into a single blank line (2 newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_markdown(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # Replace Markdown images with their alt text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # Replace Markdown links with their link text
    text = re.sub(r"`{1,3}", "", text)  # Remove inline code and code block backticks
    text = re.sub(r"(?<!\w)(\*\*|__|~~|\*|_)(.*?)\1(?!\w)", r"\2", text)  # Strip bold, italic, and strikethrough markers
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)  # Remove Markdown heading hashes (# through ######)
    text = re.sub(r"(?m)^\s*>\s?", "", text)  # Remove blockquote markers (>)
    text = re.sub(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+", "", text)  # Strip bullet and numbered list prefixes
    return _clean_text(text)


def _markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    content: list[str] = []

    def flush() -> None:
        body = _clean_markdown("\n".join(content))
        if body:
            sections.append((heading, body))
        content.clear()

    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            flush()
            heading = _clean_markdown(match.group(1))
        else:
            content.append(line)
    flush()
    return sections


def _is_text_heading(lines: list[str], index: int) -> bool:
    line = lines[index].strip()
    if not line or len(line) > 100 or re.search(r"[.!?]$", line):
        return False
    if index > 0 and lines[index - 1].strip():
        return False
    if index + 1 < len(lines) and lines[index + 1].strip():
        return False
    return line.isupper() or line.istitle()


def _plain_text_sections(text: str) -> list[tuple[str | None, str]]:
    lines = text.splitlines()
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    content: list[str] = []

    def flush() -> None:
        body = _clean_text("\n".join(content))
        if body:
            sections.append((heading, body))
        content.clear()

    for index, line in enumerate(lines):
        if _is_text_heading(lines, index):
            flush()
            heading = line.strip()
        content.append(line)
    flush()
    return sections


class _HTMLTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
        "dt", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
        "li", "main", "ol", "p", "section", "table", "td", "th", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[tuple[str, str]] = []
        self.heading_tag: str | None = None
        self.ignored_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.ignored_tag = tag
            return
        if self.ignored_tag:
            return
        if tag in self._BLOCK_TAGS:
            self.events.append(("text", "\n"))
        if re.fullmatch(r"h[1-6]", tag):
            self.heading_tag = tag
            self.events.append(("heading_start", ""))

    def handle_endtag(self, tag: str) -> None:
        if self.ignored_tag:
            if tag == self.ignored_tag:
                self.ignored_tag = None
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.events.append(("heading_end", ""))
            self.heading_tag = None
        if tag in self._BLOCK_TAGS:
            self.events.append(("text", "\n"))

    def handle_data(self, data: str) -> None:
        if not self.ignored_tag:
            self.events.append(("text", data))


def _html_sections(text: str) -> list[tuple[str | None, str]]:
    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()

    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    content: list[str] = []
    heading_content: list[str] | None = None

    def flush() -> None:
        body = _clean_text("".join(content))
        if body:
            sections.append((heading, body))
        content.clear()

    for event, value in parser.events:
        if event == "heading_start":
            flush()
            heading_content = []
        elif event == "heading_end":
            heading = _clean_text("".join(heading_content or [])) or None
            heading_content = None
        elif heading_content is not None:
            heading_content.append(value)
        else:
            content.append(value)
    flush()
    return sections


def _make_document(
    text: str,
    source: str,
    source_format: str,
    raw_source: str,
    section_heading: str | None = None,
    page: int | None = None,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "source_format": source_format,
            "section_heading": section_heading,
            "page": page,
            "raw_source": raw_source,
        },
    )


def _documents_from_file(filepath: str, source: str, raw_source: str) -> list[Document]:
    extension = os.path.splitext(filepath)[1].lower()
    documents: list[Document] = []

    if extension == ".pdf":
        with pdfplumber.open(filepath) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = _clean_text(page.extract_text() or "")
                if text:
                    first_line = text.splitlines()[0]
                    heading = first_line if len(first_line) <= 120 else None
                    documents.append(
                        _make_document(
                            text, source, extension[1:], raw_source, heading, page_number
                        )
                    )
        return documents

    with open(filepath, "r", encoding="utf-8-sig") as file:
        text = file.read()

    if extension in {".md", ".markdown"}:
        sections = _markdown_sections(text)
    elif extension in {".html", ".htm"}:
        sections = _html_sections(text)
    else:
        sections = _plain_text_sections(text)

    return [
        _make_document(body, source, extension[1:], raw_source, heading)
        for heading, body in sections
    ]


def load_documents(data_dir: str) -> list[Document]:
    """Load supported files, archive originals, and persist normalized documents."""
    documents: list[Document] = []
    os.makedirs(RAW_DOCUMENTS_DIR, exist_ok=True)

    for entry in sorted(os.scandir(data_dir), key=lambda item: item.name.lower()):
        if not entry.is_file():
            continue
        extension = os.path.splitext(entry.name)[1].lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        source = os.path.relpath(entry.path, data_dir).replace(os.sep, "/")
        raw_path = os.path.join(RAW_DOCUMENTS_DIR, source)
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        if os.path.realpath(entry.path) != os.path.realpath(raw_path):
            shutil.copy2(entry.path, raw_path)
        raw_source = os.path.relpath(raw_path, DATA_DIR).replace(os.sep, "/")

        print(f"Loading: {source}")
        documents.extend(_documents_from_file(entry.path, source, raw_source))

    os.makedirs(os.path.dirname(PROCESSED_DOCUMENTS_PATH), exist_ok=True)
    with open(PROCESSED_DOCUMENTS_PATH, "w", encoding="utf-8") as file:
        for document in documents:
            file.write(
                json.dumps(
                    {"page_content": document.page_content, "metadata": document.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Loaded {len(documents)} document sections from {data_dir}")
    print(f"Raw files archived in {RAW_DOCUMENTS_DIR}")
    print(f"Processed documents saved to {PROCESSED_DOCUMENTS_PATH}")
    return documents


def load_pdfs(data_dir: str) -> list[Document]:
    """Backward-compatible loader name; now loads every supported document format."""
    return load_documents(data_dir)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _semantic_split(
    document: Document,
    embedding_model: HuggingFaceEmbeddings,
    chunk_size: int,
    similarity_threshold: float,
) -> list[Document]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", document.page_content)
        if sentence.strip()
    ]
    if not sentences:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    units = [
        part
        for sentence in sentences
        for part in text_splitter.split_text(sentence)
        if part.strip()
    ]
    if len(units) == 1:
        groups = [units]
    else:
        vectors = embedding_model.embed_documents(units)
        groups: list[list[str]] = [[units[0]]]
        for index, unit in enumerate(units[1:], start=1):
            previous_size = sum(len(part) for part in groups[-1])
            boundary = (
                _cosine_similarity(vectors[index - 1], vectors[index])
                < similarity_threshold
                or previous_size + len(unit) > chunk_size
            )
            if boundary:
                groups.append([unit])
            else:
                groups[-1].append(unit)

    chunks: list[Document] = []
    for group in groups:
        metadata = dict(document.metadata)
        metadata["chunking_strategy"] = "semantic"
        chunks.append(Document(page_content=" ".join(group), metadata=metadata))
    return chunks


def split_documents(
    documents: list[Document],
    strategy: str = CHUNKING_STRATEGY,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    similarity_threshold: float = 0.55,
    embedding_model: HuggingFaceEmbeddings | None = None,
) -> list[Document]:
    """
    Split documents with one of three strategies:
    - fixed: fixed-size chunks with overlap, the fast and predictable baseline.
    - structure: recursive character splitting that respects paragraph/newline
      boundaries; each input document is already bounded by its section header.
    - semantic: split sentence units where adjacent embedding similarity falls
      below the threshold, producing chunks around topic changes.
    """
    strategy = strategy.lower()
    if strategy not in {"fixed", "structure", "semantic"}:
        raise ValueError(
            "strategy must be 'fixed', 'structure', or 'semantic'; "
            f"got {strategy!r}"
        )
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap smaller than chunk_size")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1.0 and 1.0")

    if strategy == "semantic":
        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chunks = [
            chunk
            for document in documents
            for chunk in _semantic_split(
                document, embedding_model, chunk_size, similarity_threshold
            )
        ]
    else:
        separators = (
            ["\n\n", "\n", ". ", " ", ""]
            if strategy == "structure"
            else ["\n\n", "\n", " ", ""]
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap if strategy == "fixed" else 0,
            separators=separators,
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = strategy
    return chunks


def _indexable_chunks(
    chunks: list[Document],
) -> tuple[list[Document], list[str], list[dict[str, Any]], str]:
    if not chunks:
        raise ValueError("Cannot build indexes without document chunks")

    indexed_chunks: list[Document] = []
    chunk_ids: list[str] = []
    records: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source") or "unknown")
        metadata: dict[str, Any] = {
            key: value
            for key, value in chunk.metadata.items()
            if value is not None
            and isinstance(value, (str, int, float, bool))
        }
        metadata.update(
            {
                "source_document": source,
                "chunk_index": chunk_index,
                "section_heading": str(chunk.metadata.get("section_heading") or ""),
                "chunking_strategy": str(
                    chunk.metadata.get("chunking_strategy") or CHUNKING_STRATEGY
                ),
                "character_count": len(chunk.page_content),
            }
        )
        chunk_id = hashlib.sha256(
            f"{source}\0{chunk_index}\0{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        metadata["chunk_id"] = chunk_id
        indexed_chunk = Document(page_content=chunk.page_content, metadata=metadata)
        indexed_chunks.append(indexed_chunk)
        chunk_ids.append(chunk_id)
        records.append(
            {
                "id": chunk_id,
                "page_content": indexed_chunk.page_content,
                "metadata": metadata,
            }
        )

    fingerprint = hashlib.sha256(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            for record in records
        ).encode("utf-8")
    ).hexdigest()
    return indexed_chunks, chunk_ids, records, fingerprint


def _write_bm25_snapshot(
    records: list[dict[str, Any]],
    documents: list[Document],
    path: str,
) -> None:
    # Build BM25 from the same enriched chunks and persist its rebuildable corpus.
    BM25Retriever.from_documents(documents)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=os.path.dirname(path), delete=False
    ) as snapshot:
        for record in records:
            snapshot.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path = snapshot.name
    os.replace(temp_path, path)


def _write_index_manifest(manifest: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=BASE_DIR, delete=False
    ) as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
        temp_path = manifest_file.name
    os.replace(temp_path, INDEX_MANIFEST_PATH)


def create_indexes(chunks: list[Document]) -> None:
    """Build matching Chroma and BM25 indexes concurrently, then activate them."""
    indexed_chunks, chunk_ids, records, fingerprint = _indexable_chunks(chunks)
    index_version = uuid.uuid4().hex
    collection_name = f"docforge_{index_version}"
    bm25_snapshot = os.path.join(BM25_DIR, f"{index_version}.jsonl")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    with ThreadPoolExecutor(max_workers=2) as executor:
        chroma_future = executor.submit(
            Chroma.from_documents,
            documents=indexed_chunks,
            embedding=embedding_model,
            ids=chunk_ids,
            collection_name=collection_name,
            persist_directory=CHROMA_DIR,
        )
        bm25_future = executor.submit(
            _write_bm25_snapshot, records, indexed_chunks, bm25_snapshot
        )
        chroma_store = chroma_future.result()
        bm25_future.result()

    stored_ids = set(chroma_store._collection.get(include=["metadatas"])["ids"])
    if stored_ids != set(chunk_ids):
        raise RuntimeError("Chroma and BM25 chunks differ; refusing to activate indexes")

    _write_index_manifest(
        {
            "version": index_version,
            "collection_name": collection_name,
            "bm25_snapshot": os.path.relpath(bm25_snapshot, BASE_DIR),
            "embedding_model": EMBEDDING_MODEL,
            "chunk_count": len(chunk_ids),
            "fingerprint": fingerprint,
        }
    )
    print(f"Chroma and BM25 indexes created for {len(chunk_ids)} chunks")


def create_vectorstore(chunks: list[Document]) -> None:
    """Backward-compatible name; creates both the Chroma and BM25 indexes."""
    create_indexes(chunks)


if __name__ == "__main__":
    raw_docs = load_documents(DATA_DIR)
    chunks = split_documents(raw_docs, strategy=CHUNKING_STRATEGY)
    create_indexes(chunks)
    print("Ingestion complete! Chroma and BM25 indexes are ready.")