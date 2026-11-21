"""Gera candidatos de golden set a partir de chunks e verifica a resolução dos existentes."""

import asyncio
import json
import random
from pathlib import Path

import typer

from docsage.eval import dataset
from docsage.llm import get_client
from docsage.logging import configure, get_logger
from docsage.models import Chunk

configure()
log = get_logger(__name__)
app = typer.Typer(add_completion=False)

PROMPT = """A partir do trecho de documentação abaixo, escreva UMA pergunta que só possa ser
respondida com ele e a resposta curta correspondente.
Responda só com JSON: {{"question": "...", "answer": "..."}}

Trecho:
{text}"""


async def _candidate(chunk: Chunk) -> dict | None:
    client = get_client()
    try:
        data = await client.complete_json(PROMPT.format(text=chunk.text[:2500]))
    except Exception as exc:
        log.warning("golden.generate_failed", chunk=chunk.chunk_id, error=str(exc))
        return None
    if not data.get("question") or not data.get("answer"):
        return None
    needle = max(chunk.text.split("\n\n")[-1].split(". "), key=len)[:40].strip()
    return {
        "category": "factual",
        "answerable": True,
        "question": data["question"],
        "answer": data["answer"],
        "gold_paths": [chunk.path],
        "gold_text_contains": [needle],
    }


async def _generate(chunks_path: Path, count: int, seed: int) -> None:
    chunks = [Chunk.model_validate_json(line)
              for line in chunks_path.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(chunks)
    picked = chunks[: count * 2]

    results = await asyncio.gather(*(_candidate(c) for c in picked))
    items = [r for r in results if r][:count]

    for i, item in enumerate(items, start=1):
        item["id"] = f"g{i:03d}"

    out = Path("eval/generated.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    typer.echo(f"{len(items)} candidatos -> {out} (revisar antes de mesclar em golden.jsonl)")


@app.command()
def generate(chunks: Path = Path("data/chunks.jsonl"),
             count: int = 20, seed: int = 1337) -> None:
    asyncio.run(_generate(chunks, count, seed))


@app.command()
def verify(golden: Path = Path("eval/golden.jsonl"),
           chunks: Path = Path("data/chunks.jsonl")) -> None:
    items = dataset.load(golden, chunks, strict=False)
    answerable = [i for i in items if i.answerable]
    ok = [i for i in answerable if i.gold_chunk_ids]
    typer.echo(f"{len(ok)}/{len(answerable)} itens resolvidos")
    for item in answerable:
        if not item.gold_chunk_ids:
            typer.secho(f"  ÓRFÃO {item.id}: {item.gold_paths} {item.gold_text_contains}",
                        fg="yellow")
    raise typer.Exit(0 if len(ok) == len(answerable) else 1)


if __name__ == "__main__":
    app()
