"""Multi-format document extraction and raw/processed corpus persistence."""

import json
import os
import re
import shutil
from html.parser import HTMLParser

import pdfplumber
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DOCUMENTS_DIR = os.path.join(DATA_DIR, "raw_documents")
PROCESSED_DOCUMENTS_PATH = os.path.join(DATA_DIR, "processed_documents.jsonl")
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}


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
            self.events.append(("heading_start", ""))

    def handle_endtag(self, tag: str) -> None:
        if self.ignored_tag:
            if tag == self.ignored_tag:
                self.ignored_tag = None
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.events.append(("heading_end", ""))
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
    if extension == ".pdf":
        documents = []
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
