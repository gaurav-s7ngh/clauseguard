import io
from pathlib import Path
from typing import Tuple


def extract_text_from_file(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Extracts raw plain text from PDF, DOCX, or TXT without mutating character offsets.
    Returns (raw_text, file_type).
    """
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        file_type = "txt"
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Failed to decode plain text document: {e}")

    elif ext == ".pdf":
        file_type = "pdf"
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text("text"))
            doc.close()
            text = "\n\n".join(pages_text)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF document: {e}")

    elif ext == ".docx":
        file_type = "docx"
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX document: {e}")

    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Only PDF (.pdf), Word (.docx), and plain text (.txt) are supported."
        )

    # Normalize carriage returns to standard newlines for consistent offset counting
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text, file_type

