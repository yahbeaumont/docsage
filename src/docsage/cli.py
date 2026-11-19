import asyncio
from pathlib import Path

import typer

from docsage.corpus.fetch import DEFAULT_COMMIT, DEFAULT_REPO
from docsage.index.build import build_corpus, build_index
from docsage.logging import configure

configure()
app = typer.Typer(add_completion=False)
corpus_app = typer.Typer()
index_app = typer.Typer()
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")


@corpus_app.command("fetch")
def corpus_fetch(
    out: Path = typer.Option(Path("data/corpus"), "--out"),
    repo: str = typer.Option(DEFAULT_REPO, "--repo"),
    commit: str = typer.Option(DEFAULT_COMMIT, "--commit"),
) -> None:
    from docsage.corpus.fetch import fetch

    typer.echo(fetch(repo, commit, out))


@corpus_app.command("parse")
def corpus_parse(
    inp: Path = typer.Option(Path("data/corpus"), "--in"),
    out: Path = typer.Option(Path("data/chunks.jsonl"), "--out"),
) -> None:
    from docsage.corpus.fetch import list_markdown
    from docsage.corpus.parse import parse_document

    total = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for doc in list_markdown(inp):
            for chunk in parse_document(doc, inp):
                fh.write(chunk.model_dump_json() + "\n")
                total += 1
    typer.echo(f"{total} chunks -> {out}")


@index_app.command("build")
def index_build(chunks: Path = typer.Option(Path("data/chunks.jsonl"), "--chunks")) -> None:
    from docsage.index import store

    build_index(chunks)
    # o cliente embutido do Qdrant segura um lock de arquivo; fechar antes de sair
    # evita o ruído de __del__ no shutdown do interpretador.
    store.reset_client()
    typer.echo("índice construído")


@index_app.command("rebuild-corpus")
def rebuild(out: Path = typer.Option(Path("data/corpus"), "--out")) -> None:
    typer.echo(build_corpus(DEFAULT_REPO, DEFAULT_COMMIT, out))


@app.command("query")
def query(
    question: str = typer.Argument(...),
    mode: str = typer.Option("hybrid", "--mode"),
) -> None:
    from docsage.generate.answer import answer
    from docsage.retrieve.pipeline import retrieve

    async def run() -> None:
        result = await retrieve(question, mode=mode)
        response = await answer(question, result)
        typer.echo(response.model_dump_json(indent=2))

    asyncio.run(run())


if __name__ == "__main__":
    app()
