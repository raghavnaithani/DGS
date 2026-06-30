from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from ..models.knowledge import ChunkDocument

MIN_CHUNK_SIZE = 400
MAX_CHUNK_SIZE = 700


@dataclass(slots=True)
class _Block:
    text: str
    kind: str


@dataclass(slots=True)
class _Section:
    title: str
    blocks: list[_Block]


def _normalize_markdown(markdown: str) -> str:
    return markdown.replace("\r\n", "\n").strip()


def _classify_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "blank"
    if stripped.startswith("#"):
        return "heading"
    if stripped.startswith("|"):
        return "table"
    if re.match(r"^(?:[-*+]|\d+[.)])\s+", stripped):
        return "list"
    return "text"


def _split_blocks(markdown: str) -> list[_Block]:
    blocks: list[_Block] = []
    current_lines: list[str] = []
    current_kind = "text"

    def flush() -> None:
        nonlocal current_lines, current_kind
        if current_lines:
            blocks.append(_Block(text="\n".join(current_lines).strip(), kind=current_kind))
            current_lines = []
            current_kind = "text"

    for line in _normalize_markdown(markdown).split("\n"):
        kind = _classify_line(line)
        if kind == "blank":
            flush()
            continue
        if kind == "heading":
            flush()
            blocks.append(_Block(text=line.strip(), kind=kind))
            continue
        if not current_lines:
            current_kind = kind
        elif current_kind != kind and current_kind not in {"list", "table"}:
            flush()
            current_kind = kind
        current_lines.append(line.rstrip())

    flush()
    return [block for block in blocks if block.text]


def _split_sections(blocks: list[_Block]) -> list[_Section]:
    sections: list[_Section] = []
    current_title = "root"
    current_blocks: list[_Block] = []

    for block in blocks:
        if block.kind == "heading":
            if current_blocks:
                sections.append(_Section(title=current_title, blocks=current_blocks))
            current_title = block.text.lstrip("#").strip() or "section"
            current_blocks = [block]
            continue
        current_blocks.append(block)

    if current_blocks:
        sections.append(_Section(title=current_title, blocks=current_blocks))
    return sections


def _pack_section(section: _Section) -> list[str]:
    chunks: list[str] = []
    current_blocks: list[_Block] = []
    current_length = 0

    def flush() -> None:
        nonlocal current_blocks, current_length
        if current_blocks:
            chunks.append("\n\n".join(block.text for block in current_blocks).strip())
            current_blocks = []
            current_length = 0

    for block in section.blocks:
        candidate_length = current_length + len(block.text) + (2 if current_blocks else 0)
        if current_blocks and candidate_length > MAX_CHUNK_SIZE and current_length >= MIN_CHUNK_SIZE:
            flush()
        if len(block.text) > MAX_CHUNK_SIZE and block.kind == "text":
            if current_blocks:
                flush()
            words = block.text.split()
            current_text: list[str] = []
            current_text_length = 0
            for word in words:
                next_length = current_text_length + len(word) + (1 if current_text else 0)
                if current_text and next_length > MAX_CHUNK_SIZE:
                    chunks.append(" ".join(current_text).strip())
                    current_text = [word]
                    current_text_length = len(word)
                    continue
                current_text.append(word)
                current_text_length = next_length
            if current_text:
                chunks.append(" ".join(current_text).strip())
            continue
        current_blocks.append(block)
        current_length += len(block.text) + (2 if current_length else 0)

    flush()
    return [chunk for chunk in chunks if chunk]


def _calculate_actionability(text: str) -> float:
    score = 0.0
    if re.search(r'\b(apply|submit|enroll|register|download|install|create|build|join|attend)\b', text, re.IGNORECASE):
        score += 0.5
    if re.search(r'(\$|€|£|\d+\s*(days|weeks|months|years|hours))', text, re.IGNORECASE):
        score += 0.5
    return min(1.0, score)


def chunk_markdown(
    markdown: str,
    *,
    source_url: str,
    source_title: str | None = None,
    ttl_days: int = 30,
) -> list[ChunkDocument]:
    blocks = _split_blocks(markdown)
    sections = _split_sections(blocks)
    created_at = datetime.now(timezone.utc)
    chunks: list[ChunkDocument] = []
    chunk_index = 0

    for section in sections:
        parent_id = str(uuid4())
        parent_content = "\n\n".join(block.text for block in section.blocks).strip()
        for chunk_text in _pack_section(section):
            chunks.append(
                ChunkDocument(
                    id=str(uuid4()),
                    content=chunk_text,
                    source_url=source_url,
                    source_title=source_title,
                    chunk_index=chunk_index,
                    parent_id=parent_id,
                    parent_content=parent_content,
                    section_title=section.title,
                    embedding=[],
                    created_at=created_at,
                    ttl_days=ttl_days,
                    verification_status="unverified",
                    actionability_score=_calculate_actionability(chunk_text),
                )
            )
            chunk_index += 1

    return chunks
