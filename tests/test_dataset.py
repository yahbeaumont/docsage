import json

import pytest

from docsage.eval import dataset
from docsage.models import Chunk


def _write_chunks(path, entries):
    lines = []
    for i, (doc_path, text) in enumerate(entries):
        chunk = Chunk(
            chunk_id=f"{i:016x}",
            doc_id="d1",
            path=doc_path,
            heading_path=("H",),
            text=text,
            n_tokens=5,
            ordinal=i,
        )
        lines.append(chunk.model_dump_json())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def golden(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            ("docs/en/tutorial/dependencies/index.md", "Use Depends para injetar."),
            ("docs/en/tutorial/dependencies/index.md", "Outra seção sem o termo."),
            ("docs/en/tutorial/testing.md", "Use TestClient."),
        ],
    )
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "f001",
                        "category": "factual",
                        "answerable": True,
                        "question": "Como injetar?",
                        "answer": "Com Depends.",
                        "gold_paths": ["docs/en/tutorial/dependencies/index.md"],
                        "gold_text_contains": ["Depends"],
                    }
                ),
                json.dumps(
                    {
                        "id": "u001",
                        "category": "unanswerable",
                        "answerable": False,
                        "question": "Qual o SLA?",
                        "answer": "Não documentado.",
                        "gold_paths": [],
                        "gold_text_contains": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return golden_path, chunks_path


def test_load_resolves_gold_chunk_ids(golden):
    items = dataset.load(*golden)
    assert items[0].gold_chunk_ids == ["0" * 16]
    assert items[0].resolved is True
    assert items[1].resolved is True


def test_load_raises_when_resolution_is_low(golden, tmp_path):
    golden_path, chunks_path = golden
    raw = json.loads(golden_path.read_text().splitlines()[0])
    raw["gold_text_contains"] = ["termo inexistente"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dataset.load(bad, chunks_path)


def test_load_non_strict_logs_instead_of_raising(golden, tmp_path):
    golden_path, chunks_path = golden
    raw = json.loads(golden_path.read_text().splitlines()[0])
    raw["gold_text_contains"] = ["termo inexistente"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    items = dataset.load(bad, chunks_path, strict=False)
    assert items[0].gold_chunk_ids == []


def test_write_roundtrip(tmp_path):
    items = [
        dataset.GoldenItem(
            id="f001",
            question="q",
            answer="a",
            category="factual",
            answerable=True,
            gold_paths=["docs/x.md"],
            gold_text_contains=["x"],
        )
    ]
    out = tmp_path / "g.jsonl"
    dataset.write(out, items)
    payload = json.loads(out.read_text().splitlines()[0])
    assert payload["gold_paths"] == ["docs/x.md"]
