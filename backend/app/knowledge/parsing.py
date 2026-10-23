"""Document parsing: PDF / Markdown / plain text / DOCX -> sections.

A *section* is a run of text under one heading on one page. Sections carry the
page number and heading path so every chunk cut from them can be cited
precisely ("[metric_definitions.md, § Active customer]", "[policy.pdf, p.2]").
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".docx"}

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Section:
    text: str
    page: int | None = None      # 1-based, PDFs only
    heading: str = ""            # "Metrics > Active customer"


class UnsupportedDocument(ValueError):
    pass


def extension(filename: str) -> str:
    m = re.search(r"(\.[A-Za-z0-9]+)$", filename or "")
    return m.group(1).lower() if m else ""


def parse_document(filename: str, data: bytes) -> list[Section]:
    ext = extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocument(
            f"Unsupported document type '{ext or filename}'. Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if ext == ".pdf":
        sections = _parse_pdf(data)
    elif ext == ".docx":
        sections = _parse_docx(data)
    else:
        text = _decode(data)
        sections = _parse_markdown(text) if ext in (".md", ".markdown") else _parse_text(text)
    sections = [s for s in sections if s.text.strip()]
    if not sections:
        raise UnsupportedDocument(f"No extractable text found in '{filename}'")
    return sections


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _parse_markdown(text: str) -> list[Section]:
    """Split on headings, keeping the heading path (H1 > H2 > H3)."""
    sections: list[Section] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(text=body, heading=" > ".join(path)))
        buf.clear()

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _MD_HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
            continue
        buf.append(line)
    flush()
    return sections


def _parse_text(text: str) -> list[Section]:
    """Plain text: treat short standalone lines followed by a blank line or
    underline (===/---) as headings; otherwise one section."""
    lines = text.splitlines()
    sections: list[Section] = []
    heading = ""
    buf: list[str] = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if re.fullmatch(r"\s*[=\-]{3,}\s*", nxt or "") and line.strip() and len(line) < 80:
            if buf:
                sections.append(Section(text="\n".join(buf).strip(), heading=heading))
                buf = []
            heading = line.strip()
            continue
        if re.fullmatch(r"\s*[=\-]{3,}\s*", line):
            continue
        buf.append(line)
    if buf:
        sections.append(Section(text="\n".join(buf).strip(), heading=heading))
    return sections


def _parse_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # malformed PDF
        raise UnsupportedDocument(f"Could not read PDF: {exc}") from exc
    sections: list[Section] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        # PDFs have no reliable heading markup; a short Title-Case first line is
        # used as the section heading so citations still point somewhere useful.
        first, _, rest = text.partition("\n")
        heading = first.strip() if len(first.strip()) < 70 and rest else ""
        sections.append(Section(text=text, page=page_no, heading=heading))
    return sections


def _parse_docx(data: bytes) -> list[Section]:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise UnsupportedDocument(f"Could not read DOCX: {exc}") from exc
    sections: list[Section] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(text=body, heading=" > ".join(path)))
        buf.clear()

    for para in document.paragraphs:
        style = (para.style.name if para.style is not None else "") or ""
        m = re.match(r"Heading\s*(\d)", style)
        if m and para.text.strip():
            flush()
            level = int(m.group(1))
            path[:] = path[: level - 1] + [para.text.strip()]
        elif style == "Title" and para.text.strip():
            flush()
            path[:] = [para.text.strip()]
        else:
            buf.append(para.text)
    flush()
    # tables (data dictionaries are often tables in Word)
    for table in document.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        body = "\n".join(r for r in rows if r.strip(" |"))
        if body:
            sections.append(Section(text=body, heading=" > ".join(path) or "Table"))
    return sections
