"""Retrieved document formatting for grounded generation."""

from langchain_core.documents import Document


def format_context(documents: list[Document]) -> str:
    """Render chunks as numbered evidence blocks with source and section labels."""
    blocks = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source_document") or document.metadata.get(
            "source", "unknown"
        )
        heading = document.metadata.get("section_heading") or "Not specified"
        blocks.append(
            f"[{index}]\n"
            f"Source: {source}\n"
            f"Section: {heading}\n"
            f"Content:\n{document.page_content}"
        )
    return "\n\n".join(blocks)
