"""Pure functions for vector store file operations."""

import io
from typing import Tuple

from pypdf import PdfReader
from pypdf.errors import PdfReadError

PDF_MAGIC_BYTES = b"%PDF-"


class EmptyFileError(ValueError):
    """File has no content."""


def decode_file_content(content: bytes, filename: str) -> Tuple[str, bool]:
    """Pure function to decode file content with PDF and text support."""
    if not content:
        raise EmptyFileError(f"File {filename} has no content")

    if content.startswith(PDF_MAGIC_BYTES):
        try:
            # Create BytesIO wrapper for pypdf (content already in memory from upload)
            reader = PdfReader(io.BytesIO(content))

            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            text_content = "\n\n".join(text_parts)
            if not text_content.strip():
                raise ValueError("No text content found in PDF")

            return text_content, False

        except (PdfReadError, ValueError) as e:
            raise ValueError(f"Failed to process PDF file: {str(e)}")

    # Handle text files
    try:
        text_content = content.decode("utf-8")
        decode_errors = False
    except UnicodeDecodeError:
        text_content = content.decode("utf-8", errors="replace")
        decode_errors = True

    return text_content, decode_errors
