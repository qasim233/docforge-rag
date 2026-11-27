"""Page-level PDF text extraction."""

import pdfplumber

from ingestion.extract.common import clean_text


def extract_pdf_pages(
    filepath: str,
) -> list[tuple[str, int, str | None]]:
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                first_line = text.splitlines()[0]
                heading = first_line if len(first_line) <= 120 else None
                pages.append((text, page_number, heading))
    return pages
