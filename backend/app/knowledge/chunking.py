"""Heading-aware chunking with overlap.

Chunks never cross a section boundary, so each one belongs to exactly one
heading/page and can be cited. Long sections are split on paragraph, then
sentence boundaries into ~``max_chars`` windows with ``overlap`` characters
carried over, so a definition split across a boundary is still retrievable.
The heading path is prepended to the indexed text (not the displayed text):
"Active customer" in a heading is strong retrieval signal for its body.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .parsing import Section

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    chunk_id: str           # sha256(doc_id + text): stable, so unchanged chunks are not re-embedded
    doc_id: str
    doc_name: str
    text: str               # what is shown and cited
    index_text: str         # what is embedded / keyword-indexed (heading + text)
    heading: str
    page: int | None
    position: int           # order within the document

    def citation(self) -> str:
        where = []
        if self.page:
            where.append(f"p.{self.page}")
        if self.heading:
            where.append(f"§ {self.heading.split(' > ')[-1]}")
        return f"[{self.doc_name}{', ' + ', '.join(where) if where else ''}]"

    def metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "heading": self.heading,
            "page": self.page or 0,
            "position": self.position,
        }


def chunk_sections(sections: list[Section], doc_id: str, doc_name: str,
                   max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        for piece in _split(section.text.strip(), max_chars, overlap):
            index_text = f"{section.heading}\n{piece}" if section.heading else piece
            chunk_id = hashlib.sha256(f"{doc_id}\x00{index_text}".encode("utf-8")).hexdigest()[:32]
            chunks.append(Chunk(
                chunk_id=chunk_id, doc_id=doc_id, doc_name=doc_name, text=piece,
                index_text=index_text, heading=section.heading, page=section.page,
                position=len(chunks),
            ))
    # identical text repeated in one document (headers/footers) is indexed once
    seen: set[str] = set()
    unique = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique.append(c)
    return unique


def _split(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    # paragraphs, then sentences, as the atomic units
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(s.strip() for s in _SENTENCE.split(para) if s.strip())
    pieces: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:  # a single enormous sentence: hard-wrap it
            for i in range(0, len(unit), max_chars - overlap):
                pieces.append(unit[i:i + max_chars])
            current = ""
            continue
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        pieces.append(current)
        tail = current[-overlap:] if overlap else ""
        # start the overlap at a word boundary
        tail = tail[tail.find(" ") + 1:] if " " in tail else tail
        current = f"{tail} {unit}".strip() if tail else unit
    if current:
        pieces.append(current)
    return pieces
