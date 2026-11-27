"""Plain-text cleanup and likely standalone heading detection."""

import re

from ingestion.extract.common import clean_text


def _is_text_heading(lines: list[str], index: int) -> bool:
    line = lines[index].strip()
    if not line or len(line) > 100 or re.search(r"[.!?]$", line):
        return False
    if index > 0 and lines[index - 1].strip():
        return False
    if index + 1 < len(lines) and lines[index + 1].strip():
        return False
    return line.isupper() or line.istitle()


def extract_text_sections(text: str) -> list[tuple[str | None, str]]:
    lines = text.splitlines()
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    content: list[str] = []

    def flush() -> None:
        body = clean_text("\n".join(content))
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
