"""Multi-format extraction and raw/processed corpus persistence."""

import json
import os
import shutil

from langchain_core.documents import Document

from app.config import settings
from ingestion.extract.html import extract_html_sections
from ingestion.extract.markdown import extract_markdown_sections
from ingestion.extract.pdf import extract_pdf_pages
from ingestion.extract.plain_text import extract_text_sections

DATA_DIR = str(settings.data_dir)
RAW_DOCUMENTS_DIR = str(settings.raw_documents_dir)
PROCESSED_DOCUMENTS_PATH = str(settings.processed_documents_path)
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}


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
    if extension == ".pdf":
        return [
            _make_document(text, source, extension[1:], raw_source, heading, page)
            for text, page, heading in extract_pdf_pages(filepath)
        ]

    with open(filepath, "r", encoding="utf-8-sig") as file:
        text = file.read()
    if extension in {".md", ".markdown"}:
        sections = extract_markdown_sections(text)
    elif extension in {".html", ".htm"}:
        sections = extract_html_sections(text)
    else:
        sections = extract_text_sections(text)
    return [
        _make_document(body, source, extension[1:], raw_source, heading)
        for heading, body in sections
    ]


def load_documents(data_dir: str | os.PathLike[str] | None = None) -> list[Document]:
    """Load supported files, archive originals, and persist normalized documents."""
    data_dir = os.fspath(data_dir or settings.data_dir)
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


def load_pdfs(data_dir: str | os.PathLike[str] | None = None) -> list[Document]:
    """Backward-compatible loader name; now loads every supported document format."""
    return load_documents(data_dir)
