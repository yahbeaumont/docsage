import re

from docsage.models import CitationReport

# Aceita qualquer id alfanumérico: um marcador inventado com formato errado
# também precisa contar como órfão. Com regex só de hex, [[nao-existe]] passava
# batido e a taxa de citação válida ficava 1.0 sem ser.
CITE = re.compile(r"\[\[([A-Za-z0-9_.:-]{2,64})\]\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def extract_markers(text: str) -> list[str]:
    return CITE.findall(text)


def strip_invalid(text: str, allowed: set[str]) -> tuple[str, int]:
    orphan = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal orphan
        if match.group(1) in allowed:
            return match.group(0)
        orphan += 1
        return ""

    cleaned = CITE.sub(repl, text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, orphan


def validate(answer: str, allowed: set[str]) -> tuple[str, CitationReport]:
    total = len(extract_markers(answer))
    cleaned, orphan = strip_invalid(answer, allowed)

    sentences = [s for s in SENTENCE.split(cleaned) if s.strip()]
    supported = sum(1 for s in sentences if CITE.search(s))

    return cleaned, CitationReport(
        total_markers=total,
        valid_markers=total - orphan,
        orphan_markers=orphan,
        sentences=len(sentences),
        supported_sentences=supported,
    )
