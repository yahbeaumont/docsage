"""Chunker Markdown consciente de headings, code fences e frontmatter."""

import hashlib
import re
from pathlib import Path

import tiktoken

from docsage.models import Chunk

ENC = tiktoken.get_encoding("cl100k_base")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :].lstrip("\n")
    return text


def chunk_id(doc_id: str, heading: str, ordinal: int) -> str:
    raw = f"{doc_id}|{heading}|{ordinal}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def split_sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Divide por headings ATX, mantendo a trilha de headings como contexto."""
    sections: list[tuple[tuple[str, ...], str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((tuple(h for _, h in stack), body))
        buf.clear()

    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        match = HEADING.match(line) if not in_fence else None
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return sections


def window(text: str, size: int, overlap: int) -> list[str]:
    """Janela por tokens com sobreposição, cortando em fronteira de parágrafo quando possível."""
    if count_tokens(text) <= size:
        return [text]
    paras = [p for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for para in paras:
        pn = count_tokens(para)
        if cur_n + pn > size and cur:
            out.append("\n\n".join(cur))
            tail: list[str] = []
            tail_n = 0
            for prev in reversed(cur):
                pn_prev = count_tokens(prev)
                if tail_n + pn_prev > overlap:
                    break
                tail.insert(0, prev)
                tail_n += pn_prev
            cur, cur_n = tail, tail_n
        cur.append(para)
        cur_n += pn
    if cur:
        out.append("\n\n".join(cur))
    return out


def parse_document(path: Path, root: Path, size: int = 350, overlap: int = 60) -> list[Chunk]:
    rel = path.relative_to(root).as_posix()
    doc_id = hashlib.sha1(rel.encode()).hexdigest()[:16]
    text = strip_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
    chunks: list[Chunk] = []
    ordinal = 0
    for heading_path, body in split_sections(text):
        heading = " > ".join(heading_path) or "(raiz)"
        for piece in window(body, size, overlap):
            piece = piece.strip()
            if count_tokens(piece) < 25:
                continue
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(doc_id, heading, ordinal),
                    doc_id=doc_id,
                    path=rel,
                    heading_path=heading_path,
                    text=f"{heading}\n\n{piece}",
                    n_tokens=count_tokens(piece),
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return chunks
