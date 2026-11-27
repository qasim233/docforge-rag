"""HTML text extraction, excluding non-content elements."""

import re
from html.parser import HTMLParser

from ingestion.extract.common import clean_text


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


def extract_html_sections(text: str) -> list[tuple[str | None, str]]:
    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()

    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    content: list[str] = []
    heading_content: list[str] | None = None

    def flush() -> None:
        body = clean_text("".join(content))
        if body:
            sections.append((heading, body))
        content.clear()

    for event, value in parser.events:
        if event == "heading_start":
            flush()
            heading_content = []
        elif event == "heading_end":
            heading = clean_text("".join(heading_content or [])) or None
            heading_content = None
        elif heading_content is not None:
            heading_content.append(value)
        else:
            content.append(value)
    flush()
    return sections
