from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ParsedDocument:
    title: str
    text: str
    metadata: Dict[str, Any]


def parse_document(path: str | Path) -> ParsedDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        title = _markdown_title(text) or path.stem
        return ParsedDocument(title=title, text=text, metadata={"format": suffix[1:]})
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            title = str(payload.get("title") or path.stem)
            text = str(payload.get("content") or json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            title = path.stem
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        return ParsedDocument(title=title, text=text, metadata={"format": "json"})
    raise ValueError(f"Unsupported knowledge document type: {suffix}")


def chunk_document(text: str, max_chars: int = 1200, overlap_chars: int = 160) -> List[str]:
    if max_chars <= overlap_chars:
        raise ValueError("max_chars must be greater than overlap_chars.")
    sections = re.split(r"(?m)(?=^#{1,6}\s+)", text)
    chunks: List[str] = []
    current = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        paragraphs = re.split(r"\n\s*\n", section)
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(current) + len(paragraph) + 2 <= max_chars:
                current = f"{current}\n\n{paragraph}".strip()
                continue
            if current:
                chunks.append(current)
            if len(paragraph) <= max_chars:
                current = paragraph
                continue
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end])
                if end == len(paragraph):
                    break
                start = end - overlap_chars
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _markdown_title(text: str) -> str | None:
    match = re.search(r"(?m)^#\s+(.+)$", text)
    return match.group(1).strip() if match else None

