"""Markdown cleanup and heading-based section extraction."""

import re

from ingestion.extract.common import clean_text


def _clean_markdown(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"(?<!\w)(\*\*|__|~~|\*|_)(.*?)\1(?!\w)", r"\2", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+", "", text)
    return clean_text(text)


def extract_markdown_sections(text: str) -> list[tuple[str | None, str]]:
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
