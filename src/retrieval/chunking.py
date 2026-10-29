"""Structure-aware chunking of markdown KB documents.

A Track A KB doc is YAML front matter + "# Title" + "## Section" blocks.
Each "## Section" becomes one chunk with ID "<doc_id>#<slug(section)>", so gold
labels can name chunks without depending on token counts. Text before the
first "##" (if any) becomes "<doc_id>#overview".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def slugify(heading: str) -> str:
    """'EU and UK right of withdrawal' -> 'eu-and-uk-right-of-withdrawal'."""
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Split '---\\n<yaml>\\n---\\n<body>' into (metadata, body)."""
    if not raw.startswith("---"):
        return {}, raw
    _, fm, body = raw.split("---", 2)
    return yaml.safe_load(fm) or {}, body.strip()


def chunk_markdown_doc(path: Path) -> list[Chunk]:
    """One chunk per '## ' section of a KB markdown file."""
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    doc_id = meta.get("doc_id", path.stem)
    title = meta.get("title", doc_id)
    sections: list[tuple[str, list[str]]] = [("overview", [])]
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif line.startswith("# "):
            continue  # document title, already in metadata
        else:
            sections[-1][1].append(line)

    chunks = []
    doc_meta = {k: v for k, v in meta.items() if k not in ("doc_id", "title")}
    for heading, lines in sections:
        text = "\n".join(lines).strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=f"{doc_id}#{slugify(heading)}", doc_id=doc_id, title=title, section=heading,
            text=f"{title}: {heading}\n\n{text}", metadata=doc_meta,
        ))
    return chunks


def chunk_kb_dir(kb_dir: Path) -> list[Chunk]:
    """Chunk every markdown doc in a directory (README.md excluded)."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name.lower() != "readme.md":
            chunks += chunk_markdown_doc(path)
    return chunks
